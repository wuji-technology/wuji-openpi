from __future__ import annotations

import collections
import logging
import math
import threading
import time
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from typing import Dict

import numpy as np
import tree
from typing_extensions import override

from openpi_client import base_policy as _base_policy

logger = logging.getLogger(__name__)


def _copy_obs_tree(obs: Dict) -> Dict:
    return tree.map_structure(lambda x: x.copy() if isinstance(x, np.ndarray) else x, obs)


def _extract_anchor_state(obs: Dict, action_dim: int) -> np.ndarray | None:
    for key in ("observation/state", "state"):
        value = obs.get(key)
        if value is None:
            continue

        anchor = np.asarray(value, dtype=np.float32).reshape(-1)
        if anchor.shape[0] == action_dim:
            return anchor

    return None


def build_time_window_old_reference(
    current_chunk: np.ndarray | None,
    start_step: int,
    window_len: int,
) -> np.ndarray | None:
    """Build a future old-trajectory window and extrapolate if it runs out.

    This is closer to the paper's RTG description than a simple tail copy:
    we take the still-relevant future segment of the currently executing
    trajectory, then extrapolate beyond the end using the last observed velocity.
    """

    if current_chunk is None or window_len <= 0:
        return None

    old_chunk = np.asarray(current_chunk, dtype=np.float32)
    if old_chunk.ndim != 2:
        return None

    start = min(max(start_step, 0), old_chunk.shape[0] - 1)
    reference = old_chunk[start : start + window_len].copy()
    if len(reference) == 0:
        return None

    if len(reference) < window_len:
        if old_chunk.shape[0] >= 2:
            velocity = old_chunk[-1] - old_chunk[-2]
        else:
            velocity = np.zeros(old_chunk.shape[1], dtype=np.float32)

        last = reference[-1].copy()
        extra = []
        for _ in range(window_len - len(reference)):
            last = last + velocity
            extra.append(last.copy())
        if extra:
            reference = np.concatenate([reference, np.stack(extra, axis=0)], axis=0)

    return reference


def cubic_smooth_prefix(
    chunk: np.ndarray,
    anchor: np.ndarray,
    guidance_steps: int,
) -> np.ndarray:
    """Smooth the first few actions with cubic Hermite interpolation.

    Inspired by the RTG strategy in arXiv:2507.17141:
    use the current robot state as the anchor, then replace a short prefix of the
    newly predicted chunk with a smooth cubic transition before resuming the
    original trajectory tail.
    """

    chunk_arr = np.asarray(chunk)
    if chunk_arr.ndim != 2:
        raise ValueError(f"Expected chunk with shape [horizon, action_dim], got {chunk_arr.shape}")

    anchor_arr = np.asarray(anchor, dtype=np.float32).reshape(-1)
    if anchor_arr.shape[0] != chunk_arr.shape[1]:
        raise ValueError(f"Anchor dim {anchor_arr.shape[0]} does not match action dim {chunk_arr.shape[1]}")

    prefix_len = max(0, min(int(guidance_steps), chunk_arr.shape[0]))
    if prefix_len == 0:
        return np.array(chunk_arr, copy=True)

    smoothed = np.asarray(chunk_arr, dtype=np.float32).copy()

    if prefix_len == 1:
        smoothed[0] = anchor_arr
        return smoothed.astype(chunk_arr.dtype, copy=False)

    end_idx = prefix_len - 1
    end_pos = smoothed[end_idx].copy()

    if end_idx + 1 < smoothed.shape[0]:
        end_vel = smoothed[end_idx + 1] - end_pos
    elif end_idx > 0:
        end_vel = end_pos - smoothed[end_idx - 1]
    else:
        end_vel = np.zeros_like(end_pos)

    start_vel = np.zeros_like(anchor_arr)
    span = float(end_idx)

    tangent0 = start_vel * span
    tangent1 = end_vel * span

    u = np.linspace(0.0, 1.0, prefix_len, dtype=np.float32)[:, None]
    h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
    h10 = u**3 - 2.0 * u**2 + u
    h01 = -2.0 * u**3 + 3.0 * u**2
    h11 = u**3 - u**2

    smoothed[:prefix_len] = (
        h00 * anchor_arr[None, :] + h10 * tangent0[None, :] + h01 * end_pos[None, :] + h11 * tangent1[None, :]
    )
    smoothed[0] = anchor_arr
    smoothed[end_idx] = end_pos

    return smoothed.astype(chunk_arr.dtype, copy=False)


def qp_smooth_prefix(
    chunk: np.ndarray,
    anchor: np.ndarray,
    guidance_steps: int,
    *,
    start_index: int = 0,
    old_reference: np.ndarray | None = None,
    smoothness_weight: float = 1.0,
    anchor_weight: float = 10.0,
    old_weight: float = 1.0,
    new_weight: float = 1.0,
    velocity_scale: float = 1.25,
    min_step: float = 1e-3,
) -> np.ndarray:
    """QP-style RTG smoothing over the prefix of a chunk.

    This is a closer approximation to the paper than the simplified cubic
    variant:
    - smoothness term via second differences
    - time-varying attraction to the old trajectory
    - time-varying attraction to the new chunk
    - per-step velocity constraints
    """

    try:
        from scipy.optimize import Bounds
        from scipy.optimize import LinearConstraint
        from scipy.optimize import minimize
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("qp_smooth_prefix requires scipy to be installed") from exc

    chunk_arr = np.asarray(chunk)
    if chunk_arr.ndim != 2:
        raise ValueError(f"Expected chunk with shape [horizon, action_dim], got {chunk_arr.shape}")

    anchor_arr = np.asarray(anchor, dtype=np.float32).reshape(-1)
    if anchor_arr.shape[0] != chunk_arr.shape[1]:
        raise ValueError(f"Anchor dim {anchor_arr.shape[0]} does not match action dim {chunk_arr.shape[1]}")

    start_index = min(max(int(start_index), 0), chunk_arr.shape[0] - 1)
    prefix_len = max(0, min(int(guidance_steps), chunk_arr.shape[0] - start_index))
    if prefix_len == 0:
        return np.array(chunk_arr, copy=True)

    smoothed = np.asarray(chunk_arr, dtype=np.float32).copy()
    if prefix_len == 1:
        smoothed[start_index] = anchor_arr
        return smoothed.astype(chunk_arr.dtype, copy=False)

    new_ref = smoothed[start_index : start_index + prefix_len].copy()
    if old_reference is not None:
        old_ref = np.asarray(old_reference, dtype=np.float32)
        if old_ref.shape != new_ref.shape:
            raise ValueError(f"old_reference shape {old_ref.shape} does not match prefix shape {new_ref.shape}")
    else:
        old_ref = new_ref.copy()

    # Paper-inspired time-varying weights: old trajectory weight decays
    # exponentially over the blending window, new chunk weight rises.
    blend = np.linspace(0.0, 1.0, prefix_len, dtype=np.float32)
    decay = np.exp(-3.0 * blend)
    if old_reference is None:
        old_w = np.zeros(prefix_len, dtype=np.float32)
        new_w = np.full(prefix_len, new_weight, dtype=np.float32)
    else:
        old_w = old_weight * decay
        new_w = new_weight * (1.0 - decay)
        old_w[0] = max(old_w[0], old_weight)
        new_w[-1] = max(new_w[-1], new_weight)

    if prefix_len >= 3:
        d2 = np.zeros((prefix_len - 2, prefix_len), dtype=np.float64)
        for i in range(prefix_len - 2):
            d2[i, i : i + 3] = (1.0, -2.0, 1.0)
        smooth_q = d2.T @ d2
    else:
        smooth_q = np.zeros((prefix_len, prefix_len), dtype=np.float64)

    # Compute velocity bound from BOTH old and new references so the QP
    # solver is not locked to the old trajectory's (often tiny) velocity.
    old_stack = np.concatenate(
        [anchor_arr[None, :], old_ref if old_reference is not None else new_ref],
        axis=0,
    )
    new_stack = np.concatenate([anchor_arr[None, :], new_ref], axis=0)
    old_candidates = np.abs(np.diff(old_stack, axis=0))
    new_candidates = np.abs(np.diff(new_stack, axis=0))
    step_candidates = np.maximum(
        np.max(old_candidates, axis=0),
        np.max(new_candidates, axis=0),
    )
    max_step = np.maximum(step_candidates * velocity_scale, min_step).astype(np.float64)

    if prefix_len >= 2:
        diff_a = np.zeros((prefix_len - 1, prefix_len), dtype=np.float64)
        for i in range(prefix_len - 1):
            diff_a[i, i] = -1.0
            diff_a[i, i + 1] = 1.0
    else:
        diff_a = None

    bounds_lb = np.full(prefix_len, -np.inf, dtype=np.float64)
    bounds_ub = np.full(prefix_len, np.inf, dtype=np.float64)
    bounds_lb[0] = float(anchor_arr[0])  # placeholder, overwritten per dim
    bounds_ub[0] = float(anchor_arr[0])

    for dim in range(smoothed.shape[1]):
        old_dim = old_ref[:, dim].astype(np.float64, copy=False)
        new_dim = new_ref[:, dim].astype(np.float64, copy=False)
        anchor_dim = float(anchor_arr[dim])
        constraints = []
        if diff_a is not None:
            max_step_dim = float(max_step[dim])
            constraints.append(LinearConstraint(diff_a, -max_step_dim, max_step_dim))

        def objective(x: np.ndarray) -> float:
            anchor_loss = anchor_weight * (x[0] - anchor_dim) ** 2
            old_loss = np.sum(old_w * (x - old_dim) ** 2)
            new_loss = np.sum(new_w * (x - new_dim) ** 2)
            smooth_loss = smoothness_weight * float(x @ smooth_q @ x)
            return float(anchor_loss + old_loss + new_loss + smooth_loss)

        def gradient(x: np.ndarray) -> np.ndarray:
            grad = 2.0 * smoothness_weight * (smooth_q @ x)
            grad += 2.0 * old_w * (x - old_dim)
            grad += 2.0 * new_w * (x - new_dim)
            grad[0] += 2.0 * anchor_weight * (x[0] - anchor_dim)
            return grad

        x_init = 0.5 * (old_dim + new_dim)
        x_init[0] = anchor_dim

        bounds_lb[0] = anchor_dim
        bounds_ub[0] = anchor_dim

        result = minimize(
            objective,
            x_init,
            method="SLSQP",
            jac=gradient,
            bounds=Bounds(bounds_lb, bounds_ub),
            constraints=constraints,
            options={"ftol": 1e-6, "maxiter": 100, "disp": False},
        )

        if not result.success:
            logger.debug("QP RTG solver failed for dim=%s with message=%s; falling back to cubic", dim, result.message)
            return cubic_smooth_prefix(chunk_arr, anchor_arr, guidance_steps)

        smoothed[start_index : start_index + prefix_len, dim] = result.x.astype(np.float32)

    smoothed[start_index] = anchor_arr
    return smoothed.astype(chunk_arr.dtype, copy=False)


class RTGActionBroker(_base_policy.BasePolicy):
    """Async action broker with RTG-style prefix smoothing.

    This broker overlaps inference with execution, then smooths the first few
    actions of each newly arrived chunk before splicing it into the live control
    stream.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        action_horizon: int,
        control_hz: float = 30.0,
        trigger_fraction: float = 0.5,
        guidance_steps: int = 3,
    ):
        self._policy = policy
        self._action_horizon = action_horizon
        self._control_hz = control_hz
        self._guidance_steps = guidance_steps

        self._trigger_step = min(
            action_horizon - 1,
            max(1, int(action_horizon * trigger_fraction)),
        )

        self._current_chunk: np.ndarray | None = None
        self._last_results: Dict[str, Any] | None = None
        self._cur_step = 0
        self._last_action: np.ndarray | None = None

        self._executor = ThreadPoolExecutor(max_workers=2)
        self._pending_future: Future | None = None
        self._trigger_step_at: int | None = None
        self._latency_window: collections.deque[float] = collections.deque(maxlen=10)
        self._latency_lock = threading.Lock()
        self._lock = threading.Lock()

    def _estimate_inference_delay(self) -> int:
        with self._latency_lock:
            if not self._latency_window:
                return 0
            samples = list(self._latency_window)

        latency = float(np.median(samples))
        return max(0, math.ceil(latency * self._control_hz))

    def _do_inference(self, obs: Dict) -> Dict:
        t0 = time.monotonic()
        result = self._policy.infer(obs)
        elapsed = time.monotonic() - t0
        with self._latency_lock:
            self._latency_window.append(elapsed)
        return result

    def close(self) -> None:
        """Shut down the inference executor. Safe to call multiple times."""
        with self._lock:
            if self._pending_future is not None:
                self._pending_future.cancel()
                self._pending_future = None
        self._executor.shutdown(wait=False, cancel_futures=True)

    def __del__(self) -> None:
        # Best-effort cleanup; we cannot rely on this in normal shutdown paths.
        try:
            self.close()
        except Exception:
            pass

    def _extract_chunk(self, results: Dict) -> np.ndarray:
        actions = results.get("actions")
        if actions is None:
            raise ValueError("Inference result missing 'actions'")

        arr = np.asarray(actions)
        if arr.shape[0] != self._action_horizon:
            raise ValueError(f"Expected horizon {self._action_horizon}, got {arr.shape}")

        return arr

    def _smooth_results(self, results: Dict, obs: Dict) -> tuple[Dict, np.ndarray]:
        chunk = self._extract_chunk(results)

        anchor = _extract_anchor_state(obs, chunk.shape[1])
        if anchor is None and self._last_action is not None and self._last_action.shape[0] == chunk.shape[1]:
            anchor = self._last_action.astype(np.float32, copy=False)

        if anchor is None:
            return results, chunk

        smoothed_chunk = cubic_smooth_prefix(chunk, anchor, self._guidance_steps)
        smoothed_results = dict(results)
        smoothed_results["actions"] = smoothed_chunk
        return smoothed_results, smoothed_chunk

    def _swap_in_ready_chunk(self, new_results: Dict, obs: Dict) -> None:
        smoothed_results, smoothed_chunk = self._smooth_results(new_results, obs)

        steps_elapsed = max(
            0,
            min(
                self._cur_step - (self._trigger_step_at or 0),
                self._action_horizon - 1,
            ),
        )

        self._last_results = smoothed_results
        self._current_chunk = smoothed_chunk
        self._cur_step = steps_elapsed

        logger.debug(
            "RTG splice offset=%s latency_est=%s guidance_steps=%s",
            steps_elapsed,
            self._estimate_inference_delay(),
            self._guidance_steps,
        )

    @override
    def infer(self, obs: Dict) -> Dict:
        with self._lock:
            if self._last_results is None:
                initial_results = self._do_inference(obs)
                self._last_results, self._current_chunk = self._smooth_results(initial_results, obs)
                self._cur_step = 0

            if self._pending_future is not None and self._pending_future.done():
                try:
                    self._swap_in_ready_chunk(self._pending_future.result(), obs)
                except Exception:
                    logger.exception("RTG async inference failed")
                finally:
                    self._pending_future = None
                    self._trigger_step_at = None

            if (
                self._cur_step >= self._trigger_step
                and self._pending_future is None
                and self._current_chunk is not None
            ):
                obs_snapshot = _copy_obs_tree(obs)
                self._trigger_step_at = self._cur_step
                self._pending_future = self._executor.submit(self._do_inference, obs_snapshot)
                logger.debug("RTG trigger step=%s", self._cur_step)

            def slicer(x: Any) -> Any:
                if isinstance(x, np.ndarray) and x.ndim >= 1 and x.shape[0] == self._action_horizon:
                    if self._cur_step >= x.shape[0]:
                        raise RuntimeError("RTG chunk exhausted unexpectedly")
                    return x[self._cur_step]

                return x

            results = tree.map_structure(slicer, self._last_results)

            action = results.get("actions")
            if isinstance(action, np.ndarray):
                self._last_action = action.copy()

            self._cur_step += 1

            if self._cur_step >= self._action_horizon:
                if self._pending_future is not None:
                    logger.debug("RTG waiting for async inference")

                    try:
                        self._swap_in_ready_chunk(self._pending_future.result(), obs)
                    except Exception:
                        logger.exception("RTG async inference failed at boundary")
                        self._last_results = None
                        self._current_chunk = None
                        self._cur_step = 0
                    finally:
                        self._pending_future = None
                        self._trigger_step_at = None
                else:
                    self._last_results = None
                    self._current_chunk = None
                    self._cur_step = 0

            return results

    @override
    def reset(self) -> None:
        with self._lock:
            if self._pending_future is not None:
                self._pending_future.cancel()
                self._pending_future = None

            self._policy.reset()
            self._last_results = None
            self._current_chunk = None
            self._cur_step = 0
            self._last_action = None
            self._trigger_step_at = None


class QPRTGActionBroker(RTGActionBroker):
    """QP-based RTG broker, closer to the paper's trajectory blending idea."""

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        action_horizon: int,
        control_hz: float = 30.0,
        trigger_fraction: float = 0.5,
        guidance_steps: int = 6,
        smoothness_weight: float = 1.0,
        anchor_weight: float = 10.0,
        old_weight: float = 1.0,
        new_weight: float = 1.0,
        velocity_scale: float = 1.25,
        min_step: float = 1e-3,
    ):
        super().__init__(
            policy=policy,
            action_horizon=action_horizon,
            control_hz=control_hz,
            trigger_fraction=trigger_fraction,
            guidance_steps=guidance_steps,
        )
        self._smoothness_weight = smoothness_weight
        self._anchor_weight = anchor_weight
        self._old_weight = old_weight
        self._new_weight = new_weight
        self._velocity_scale = velocity_scale
        self._min_step = min_step

    @override
    def _smooth_results(self, results: Dict, obs: Dict) -> tuple[Dict, np.ndarray]:
        chunk = self._extract_chunk(results)

        anchor = _extract_anchor_state(obs, chunk.shape[1])
        if anchor is None and self._last_action is not None and self._last_action.shape[0] == chunk.shape[1]:
            anchor = self._last_action.astype(np.float32, copy=False)

        if anchor is None:
            return results, chunk

        smoothed_chunk = qp_smooth_prefix(
            chunk,
            anchor,
            self._guidance_steps,
            smoothness_weight=self._smoothness_weight,
            anchor_weight=self._anchor_weight,
            old_weight=self._old_weight,
            new_weight=self._new_weight,
            velocity_scale=self._velocity_scale,
            min_step=self._min_step,
        )
        smoothed_results = dict(results)
        smoothed_results["actions"] = smoothed_chunk
        return smoothed_results, smoothed_chunk

    @override
    def _swap_in_ready_chunk(self, new_results: Dict, obs: Dict) -> None:
        chunk = self._extract_chunk(new_results)

        steps_elapsed = max(
            0,
            min(
                self._cur_step - (self._trigger_step_at or 0),
                self._action_horizon - 1,
            ),
        )

        anchor = _extract_anchor_state(obs, chunk.shape[1])
        if anchor is None and self._last_action is not None and self._last_action.shape[0] == chunk.shape[1]:
            anchor = self._last_action.astype(np.float32, copy=False)

        if anchor is not None:
            blend_start = min(steps_elapsed, chunk.shape[0] - 1)
            window_len = min(self._guidance_steps, chunk.shape[0] - blend_start)
            old_reference = build_time_window_old_reference(
                self._current_chunk,
                self._cur_step,
                window_len,
            )
            smoothed_chunk = qp_smooth_prefix(
                chunk,
                anchor,
                window_len,
                start_index=blend_start,
                old_reference=old_reference,
                smoothness_weight=self._smoothness_weight,
                anchor_weight=self._anchor_weight,
                old_weight=self._old_weight,
                new_weight=self._new_weight,
                velocity_scale=self._velocity_scale,
                min_step=self._min_step,
            )
            smoothed_results = dict(new_results)
            smoothed_results["actions"] = smoothed_chunk
        else:
            smoothed_results = new_results
            smoothed_chunk = chunk

        self._last_results = smoothed_results
        self._current_chunk = smoothed_chunk
        self._cur_step = steps_elapsed

        logger.debug(
            "QP-RTG splice offset=%s latency_est=%s guidance_steps=%s",
            steps_elapsed,
            self._estimate_inference_delay(),
            self._guidance_steps,
        )
