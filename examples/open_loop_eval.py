#!/usr/bin/env python

# Copyright 2024 The OpenPI Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Open-loop testing script for evaluating trained Pi0/Pi0.5 policies on dataset episodes.

This script:
1. Loads a trained policy model
2. Randomly selects episodes from the dataset
3. Runs the model in open-loop mode to predict actions
4. Compares predicted actions with ground truth actions
5. Generates comparison plots for each episode

Usage:
    # For local checkpoint:
    python examples/open_loop_eval.py \
        --checkpoint_dir=outputs/checkpoints/pi05_wuji \
        --config_name=pi05_wuji \
        --num_episodes=5 \
        --output_dir=open_loop_results \
        --device=cuda

    # For Google Cloud Storage checkpoint:
    python examples/open_loop_eval.py \
        --checkpoint_dir=gs://openpi-assets/checkpoints/pi0_fast_droid \
        --config_name=pi0_fast_droid \
        --num_episodes=5 \
        --output_dir=open_loop_results \
        --device=cuda
"""

import argparse
import copy
import logging
from pathlib import Path
import random

import jax
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import matplotlib.pyplot as plt
import numpy as np
from openpi_client.rtg_action_broker import build_time_window_old_reference
from openpi_client.rtg_action_broker import cubic_smooth_prefix
from openpi_client.rtg_action_broker import qp_smooth_prefix
import torch
from tqdm import tqdm

from openpi import transforms as _transforms
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_policy_and_config(checkpoint_dir: str, config_name: str, device: str):
    """Load policy from checkpoint and training config.

    Args:
        checkpoint_dir: Path to the checkpoint directory
        config_name: Name of the training configuration
        device: Device to load the model on ('cuda', 'cpu', etc.)

    Returns:
        policy: Loaded policy model
        train_config: Training configuration
    """
    logging.info(f"Loading policy from: {checkpoint_dir}")
    logging.info(f"Using config: {config_name}")

    # Download checkpoint if it's a GCS path
    checkpoint_dir = download.maybe_download(checkpoint_dir)

    # Load training config
    train_config = _config.get_config(config_name)

    # Create policy from checkpoint
    # Determine PyTorch device based on checkpoint type
    pytorch_device = device if device.startswith("cuda") else "cpu"

    policy = _policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        pytorch_device=pytorch_device,
    )

    logging.info("Policy loaded successfully")
    logging.info(f"Model type: {train_config.model.model_type}")
    logging.info(f"Action dim: {train_config.model.action_dim}")
    logging.info(f"Action horizon: {train_config.model.action_horizon}")

    return policy, train_config


def load_dataset(train_config: _config.TrainConfig, dataset_path: str | None = None):
    """Load the LeRobot dataset used for training.

    Args:
        train_config: Training configuration
        dataset_path: Optional direct path to dataset directory (overrides config)

    Returns:
        dataset: LeRobotDataset instance
        data_config: Data configuration
    """
    logging.info("Loading dataset...")

    # Create data config
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    # Determine the dataset path to use
    if dataset_path:
        # User provided explicit dataset path - use it directly
        dataset_repo_id = dataset_path
        logging.info(f"Using dataset from explicit path: {dataset_path}")
    elif data_config.repo_id:
        # Use repo_id from config (could be absolute path or HF hub name)
        dataset_repo_id = data_config.repo_id
        if dataset_repo_id.startswith("/"):
            logging.info(f"Using dataset from config absolute path: {dataset_repo_id}")
        else:
            logging.info(f"Loading dataset from HuggingFace Hub: {dataset_repo_id}")
    else:
        raise ValueError("Dataset repo_id is required. Cannot use fake data for evaluation.")

    # Load LeRobot dataset
    # NOTE: We don't apply normalization or other transforms here since the policy handles that
    # If repo_id is an absolute path, LeRobotDataset will use it directly
    dataset = lerobot_dataset.LeRobotDataset(repo_id=dataset_repo_id)

    logging.info("Dataset loaded successfully")
    logging.info(f"Total samples: {len(dataset)}")
    logging.info(f"Total episodes: {dataset.num_episodes}")
    logging.info(f"Features: {list(dataset.features.keys())}")

    return dataset, data_config


def get_episode_indices(dataset: lerobot_dataset.LeRobotDataset, num_episodes: int, seed: int = 42):
    """Randomly select episode indices from the dataset.

    Args:
        dataset: LeRobotDataset instance
        num_episodes: Number of episodes to select
        seed: Random seed for reproducibility

    Returns:
        List of selected episode indices
    """
    set_seed(seed)
    total_episodes = dataset.num_episodes

    if num_episodes > total_episodes:
        logging.warning(
            f"Requested {num_episodes} episodes but dataset only has {total_episodes}. " f"Using all episodes."
        )
        num_episodes = total_episodes

    # Randomly sample episode indices
    episode_indices = random.sample(range(total_episodes), num_episodes)
    logging.info(f"Selected {num_episodes} episodes: {episode_indices}")

    return sorted(episode_indices)


def get_episode_frame_indices(dataset: lerobot_dataset.LeRobotDataset, episode_idx: int):
    """Get all frame indices for a specific episode.

    Args:
        dataset: LeRobotDataset instance
        episode_idx: Episode index

    Returns:
        List of frame indices belonging to this episode
    """
    # Get episode data index
    episode_data = dataset.episode_data_index["from"][episode_idx], dataset.episode_data_index["to"][episode_idx]
    return list(range(episode_data[0].item(), episode_data[1].item()))


def _extract_rtg_anchor(
    obs_dict: dict, action_dim: int, fallback_action: np.ndarray | None = None
) -> np.ndarray | None:
    """Extract RTG smoothing anchor from observation state or previous action."""
    for key in ("observation/state", "state"):
        if key not in obs_dict:
            continue
        anchor = np.asarray(obs_dict[key], dtype=np.float32).reshape(-1)
        if anchor.shape[0] == action_dim:
            return anchor

    if fallback_action is not None and fallback_action.shape[0] == action_dim:
        return fallback_action.astype(np.float32, copy=False)

    return None


def predict_episode_actions(
    policy,
    dataset: lerobot_dataset.LeRobotDataset,
    data_config: _config.DataConfig,
    train_config: _config.TrainConfig,
    episode_idx: int,
    default_prompt: str | None = None,
    compare_rtg: bool = False,  # noqa: FBT001, FBT002
    compare_rtg_qp: bool = False,  # noqa: FBT001, FBT002
    rtg_guidance_steps: int = 3,
    rtg_qp_smoothness_weight: float = 1.0,
    rtg_qp_anchor_weight: float = 10.0,
    rtg_qp_old_weight: float = 1.0,
    rtg_qp_new_weight: float = 1.0,
    rtg_qp_velocity_scale: float = 1.25,
    rtg_qp_min_step: float = 1e-3,
):
    """Predict actions for all frames in an episode in open-loop mode.

    Returns:
        predicted_actions: Standard predicted actions [num_frames, action_dim]
        predicted_actions_rtg: RTG predicted actions [num_frames, action_dim] or None
        predicted_actions_rtg_qp: QP-RTG predicted actions [num_frames, action_dim] or None
        ground_truth_actions: Ground truth actions [num_frames, action_dim]
    """
    frame_indices = get_episode_frame_indices(dataset, episode_idx)

    if len(frame_indices) == 0:
        logging.warning(f"Episode {episode_idx} has no frames!")
        return None, None, None, None

    predicted_actions = []
    predicted_actions_rtg = [] if compare_rtg else None
    predicted_actions_rtg_qp = [] if compare_rtg_qp else None
    ground_truth_actions = []

    action_horizon = train_config.model.action_horizon
    action_dim = train_config.model.action_dim
    prev_action_rtg = None
    prev_action_rtg_qp = None
    prev_chunk_rtg_qp = None

    logging.info(f"Predicting actions for episode {episode_idx} ({len(frame_indices)} frames)...")
    logging.info("Running in OPEN-LOOP mode: Using dataset observations, NOT feeding back predicted actions")
    if compare_rtg:
        logging.info(f"  RTG comparison enabled: guidance_steps={rtg_guidance_steps}")
    if compare_rtg_qp:
        logging.info(
            "  QP-RTG comparison enabled: guidance_steps=%d, smooth=%.2f, anchor=%.2f, old=%.2f, new=%.2f, vel_scale=%.2f",
            rtg_guidance_steps,
            rtg_qp_smoothness_weight,
            rtg_qp_anchor_weight,
            rtg_qp_old_weight,
            rtg_qp_new_weight,
            rtg_qp_velocity_scale,
        )

    # Create repack transform to convert dataset format to policy input format
    repack_transform = _transforms.compose(data_config.repack_transforms.inputs)

    for frame_idx in tqdm(frame_indices, desc=f"Episode {episode_idx}"):
        # Get raw data sample from dataset
        sample = dataset[frame_idx]

        # Convert to dict format (dataset returns HF dataset format)
        sample_dict = dict(sample.items())

        # Store ground truth action before transformation
        if "action" in sample_dict:
            gt_action = np.array(sample_dict["action"])
        elif "observation.action" in sample_dict:
            gt_action = np.array(sample_dict["observation.action"])
        else:
            action_keys = [k for k in sample_dict if "action" in k.lower()]
            if action_keys:
                gt_action = np.array(sample_dict[action_keys[0]])
            else:
                logging.warning(f"Could not find action in frame {frame_idx}. Keys: {sample_dict.keys()}")
                continue

        if gt_action.ndim > 1:
            gt_action = gt_action[0]
        gt_action = gt_action[:action_dim]

        # Add prompt if missing
        if "prompt" not in sample_dict:
            if default_prompt:
                sample_dict["prompt"] = default_prompt
            elif "task" in sample_dict:
                sample_dict["prompt"] = sample_dict["task"]
            else:
                sample_dict["prompt"] = "Perform the task."

        obs_dict = repack_transform(sample_dict)

        # Remove action from observation to ensure open-loop evaluation
        if "action" in obs_dict:
            del obs_dict["action"]
        if "actions" in obs_dict:
            del obs_dict["actions"]

        # Generate shared noise for fair comparison (same noise for both paths)
        noise = (
            np.random.randn(action_horizon, action_dim).astype(np.float32)
            if (compare_rtg or compare_rtg_qp)
            else None
        )

        # --- Standard inference ---
        # Use deepcopy when comparing to prevent input transforms from modifying
        # obs_dict's arrays in-place (which would corrupt subsequent calls).
        obs_for_std = copy.deepcopy(obs_dict) if (compare_rtg or compare_rtg_qp) else obs_dict
        result = policy.infer(obs_for_std, noise=noise)
        chunk_original = np.array(result["actions"], dtype=np.float32)
        predicted_action = result["actions"]
        if predicted_action.ndim > 1:
            predicted_action = predicted_action[0]
        predicted_action = predicted_action[:action_dim]
        predicted_actions.append(predicted_action)

        # --- RTG inference ---
        if compare_rtg:
            chunk_rtg = np.array(result["actions"], dtype=np.float32)
            anchor = _extract_rtg_anchor(obs_dict, action_dim, prev_action_rtg)
            if anchor is not None:
                chunk_rtg = cubic_smooth_prefix(chunk_rtg, anchor, rtg_guidance_steps)
            rtg_action = chunk_rtg[0] if chunk_rtg.ndim > 1 else chunk_rtg
            rtg_action = rtg_action[:action_dim]
            predicted_actions_rtg.append(rtg_action)
            prev_action_rtg = rtg_action.copy()

        # --- QP RTG inference ---
        if compare_rtg_qp:
            chunk_rtg_qp = np.array(chunk_original, dtype=np.float32)
            anchor_qp = _extract_rtg_anchor(obs_dict, action_dim, prev_action_rtg_qp)
            old_reference = build_time_window_old_reference(
                prev_chunk_rtg_qp,
                start_step=1,
                window_len=min(rtg_guidance_steps, chunk_rtg_qp.shape[0]),
            )
            if anchor_qp is not None:
                chunk_rtg_qp = qp_smooth_prefix(
                    chunk_rtg_qp,
                    anchor_qp,
                    rtg_guidance_steps,
                    old_reference=old_reference,
                    smoothness_weight=rtg_qp_smoothness_weight,
                    anchor_weight=rtg_qp_anchor_weight,
                    old_weight=rtg_qp_old_weight,
                    new_weight=rtg_qp_new_weight,
                    velocity_scale=rtg_qp_velocity_scale,
                    min_step=rtg_qp_min_step,
                )
            rtg_qp_action = chunk_rtg_qp[0] if chunk_rtg_qp.ndim > 1 else chunk_rtg_qp
            rtg_qp_action = rtg_qp_action[:action_dim]
            predicted_actions_rtg_qp.append(rtg_qp_action)
            prev_action_rtg_qp = rtg_qp_action.copy()
            prev_chunk_rtg_qp = chunk_rtg_qp.copy()

        ground_truth_actions.append(gt_action)

    # Stack into arrays
    predicted_actions = np.stack(predicted_actions)
    ground_truth_actions = np.stack(ground_truth_actions)
    if compare_rtg:
        predicted_actions_rtg = np.stack(predicted_actions_rtg)
    if compare_rtg_qp:
        predicted_actions_rtg_qp = np.stack(predicted_actions_rtg_qp)

    logging.info(f"Predicted actions shape: {predicted_actions.shape}")
    logging.info(f"Ground truth actions shape: {ground_truth_actions.shape}")
    logging.info(f"Predicted actions range: [{predicted_actions.min():.3f}, {predicted_actions.max():.3f}]")
    logging.info(f"Ground truth actions range: [{ground_truth_actions.min():.3f}, {ground_truth_actions.max():.3f}]")
    if compare_rtg:
        logging.info(f"RTG actions range: [{predicted_actions_rtg.min():.3f}, {predicted_actions_rtg.max():.3f}]")
    if compare_rtg_qp:
        logging.info(
            f"QP-RTG actions range: [{predicted_actions_rtg_qp.min():.3f}, {predicted_actions_rtg_qp.max():.3f}]"
        )

    return (
        predicted_actions,
        predicted_actions_rtg,
        predicted_actions_rtg_qp,
        ground_truth_actions,
    )


def compute_smoothness(actions: np.ndarray) -> float:
    """Compute trajectory smoothness as mean absolute inter-step difference."""
    if actions.shape[0] < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(actions, axis=0))))


def plot_episode_comparison(
    predicted_actions: np.ndarray,
    ground_truth_actions: np.ndarray,
    episode_idx: int,
    output_dir: Path,
    config_name: str,
    predicted_actions_rtg: np.ndarray = None,
    predicted_actions_rtg_qp: np.ndarray = None,
):
    """Plot comparison between predicted and ground truth actions.

    When predicted_actions_rtg / predicted_actions_rtg_qp are provided, plots additional
    curves (GT, Standard, RTG-Simple, RTG-QP) and shows comparative metrics including
    smoothness.
    """
    if predicted_actions is None or ground_truth_actions is None:
        logging.warning(f"Skipping plot for episode {episode_idx} - no data")
        return None

    has_rtg = predicted_actions_rtg is not None
    has_rtg_qp = predicted_actions_rtg_qp is not None
    num_frames = predicted_actions.shape[0]
    action_dim = predicted_actions.shape[1]

    # Determine grid layout
    if action_dim <= 2:
        nrows, ncols = 1, action_dim
    elif action_dim <= 4:
        nrows, ncols = 2, 2
    elif action_dim <= 6:
        nrows, ncols = 2, 3
    elif action_dim <= 9:
        nrows, ncols = 3, 3
    elif action_dim <= 12:
        nrows, ncols = 3, 4
    else:
        ncols = 4
        nrows = (action_dim + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))

    axes = [axes] if action_dim == 1 else axes.flatten() if hasattr(axes, "flatten") else axes

    title = f"{config_name} - Episode {episode_idx}: Open-Loop Evaluation"
    if has_rtg and has_rtg_qp:
        title += "\nOriginal vs RTG-Simple vs RTG-QP vs Ground Truth"
    elif has_rtg:
        title += "\nOriginal vs RTG-Simple vs Ground Truth"
    elif has_rtg_qp:
        title += "\nOriginal vs RTG-QP vs Ground Truth"
    else:
        title += "\nPredicted vs Ground Truth Actions"
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)

    # Plot each action dimension
    mse_std_list = []
    mse_rtg_list = []
    mse_rtg_qp_list = []
    for dim_idx in range(action_dim):
        ax = axes[dim_idx]

        # Ground truth
        ax.plot(
            range(num_frames),
            ground_truth_actions[:, dim_idx],
            label="Ground Truth",
            color="blue",
            linewidth=2.5,
            alpha=0.8,
            linestyle="-",
        )

        # Standard prediction
        ax.plot(
            range(num_frames),
            predicted_actions[:, dim_idx],
            label="Original",
            color="#FFD700",
            linewidth=2.0,
            alpha=0.9,
            linestyle="-",
        )

        mse_std = np.mean((predicted_actions[:, dim_idx] - ground_truth_actions[:, dim_idx]) ** 2)
        mse_std_list.append(mse_std)

        dim_title = f"Dim {dim_idx} (Std MSE: {mse_std:.6f}"

        if has_rtg:
            ax.plot(
                range(num_frames),
                predicted_actions_rtg[:, dim_idx],
                label="RTG-Simple",
                color="#32CD32",
                linewidth=2.0,
                alpha=0.85,
                linestyle=":",
            )

            mse_rtg = np.mean((predicted_actions_rtg[:, dim_idx] - ground_truth_actions[:, dim_idx]) ** 2)
            mse_rtg_list.append(mse_rtg)
            dim_title += f", RTG MSE: {mse_rtg:.6f}"

        if has_rtg_qp:
            ax.plot(
                range(num_frames),
                predicted_actions_rtg_qp[:, dim_idx],
                label="RTG-QP",
                color="#8A2BE2",
                linewidth=2.0,
                alpha=0.85,
                linestyle="-.",
            )

            mse_rtg_qp = np.mean((predicted_actions_rtg_qp[:, dim_idx] - ground_truth_actions[:, dim_idx]) ** 2)
            mse_rtg_qp_list.append(mse_rtg_qp)
            dim_title += f", QP MSE: {mse_rtg_qp:.6f}"

        dim_title += ")"
        ax.set_ylabel(f"Action Dim {dim_idx}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Frame Index", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")  # noqa: FBT003
        ax.legend(loc="best", fontsize=8)
        ax.set_title(dim_title, fontsize=10)

    for idx in range(action_dim, nrows * ncols):
        if idx < len(axes):
            axes[idx].axis("off")

    # Overall metrics
    overall_mse_std = np.mean((predicted_actions - ground_truth_actions) ** 2)
    mae_std = np.mean(np.abs(predicted_actions - ground_truth_actions))
    smooth_gt = compute_smoothness(ground_truth_actions)
    smooth_std = compute_smoothness(predicted_actions)

    stats_text = "--- Original ---\n"
    stats_text += f"MSE:  {overall_mse_std:.6f}\n"
    stats_text += f"MAE:  {mae_std:.6f}\n"
    stats_text += f"Jitter: {smooth_std:.6f}\n"

    if has_rtg:
        overall_mse_rtg = np.mean((predicted_actions_rtg - ground_truth_actions) ** 2)
        mae_rtg = np.mean(np.abs(predicted_actions_rtg - ground_truth_actions))
        smooth_rtg = compute_smoothness(predicted_actions_rtg)

        stats_text += "\n--- RTG ---\n"
        stats_text += f"MSE:  {overall_mse_rtg:.6f}\n"
        stats_text += f"MAE:  {mae_rtg:.6f}\n"
        stats_text += f"Jitter: {smooth_rtg:.6f}\n"
    if has_rtg_qp:
        overall_mse_rtg_qp = np.mean((predicted_actions_rtg_qp - ground_truth_actions) ** 2)
        mae_rtg_qp = np.mean(np.abs(predicted_actions_rtg_qp - ground_truth_actions))
        smooth_rtg_qp = compute_smoothness(predicted_actions_rtg_qp)

        stats_text += "\n--- RTG-QP ---\n"
        stats_text += f"MSE:  {overall_mse_rtg_qp:.6f}\n"
        stats_text += f"MAE:  {mae_rtg_qp:.6f}\n"
        stats_text += f"Jitter: {smooth_rtg_qp:.6f}\n"
    if has_rtg or has_rtg_qp:
        stats_text += f"\n--- GT Jitter: {smooth_gt:.6f} ---\n"
        stats_text += f"Frames: {num_frames}  Dims: {action_dim}"
    else:
        stats_text += f"\nGT Jitter: {smooth_gt:.6f}\n"
        stats_text += f"Frames: {num_frames}  Dims: {action_dim}"

    fig.text(
        0.99,
        0.01,
        stats_text,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.7},
        fontsize=9,
        family="monospace",
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.99])

    if has_rtg and has_rtg_qp:
        suffix = "_rtg_rtgqp_compare"
    elif has_rtg:
        suffix = "_rtg_compare"
    elif has_rtg_qp:
        suffix = "_rtgqp_compare"
    else:
        suffix = ""
    output_path = output_dir / f"episode_{episode_idx}{suffix}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logging.info(f"Saved comparison plot to: {output_path}")
    logging.info(
        f"Episode {episode_idx} - Original MSE: {overall_mse_std:.6f}, MAE: {mae_std:.6f}, Jitter: {smooth_std:.6f}"
    )
    if has_rtg:
        logging.info(
            f"Episode {episode_idx} - RTG-S    MSE: {overall_mse_rtg:.6f}, MAE: {mae_rtg:.6f}, Jitter: {smooth_rtg:.6f}"
        )
    if has_rtg_qp:
        logging.info(
            f"Episode {episode_idx} - RTG-QP   MSE: {overall_mse_rtg_qp:.6f}, MAE: {mae_rtg_qp:.6f}, Jitter: {smooth_rtg_qp:.6f}"
        )

    return {
        "mse_std": overall_mse_std,
        "mae_std": mae_std,
        "jitter_std": smooth_std,
        "mse_rtg": overall_mse_rtg if has_rtg else None,
        "mae_rtg": mae_rtg if has_rtg else None,
        "jitter_rtg": smooth_rtg if has_rtg else None,
        "mse_rtg_qp": overall_mse_rtg_qp if has_rtg_qp else None,
        "mae_rtg_qp": mae_rtg_qp if has_rtg_qp else None,
        "jitter_rtg_qp": smooth_rtg_qp if has_rtg_qp else None,
        "jitter_gt": smooth_gt,
    }


def main():
    parser = argparse.ArgumentParser(description="Open-loop testing for trained Pi0/Pi0.5 policies")
    parser.add_argument(
        "--checkpoint_dir", type=str, required=True, help="Path to the checkpoint directory (local path or gs:// URL)"
    )
    parser.add_argument(
        "--config_name",
        type=str,
        required=True,
        help="Name of the training configuration (e.g., pi05_wuji, pi0_fast_droid)",
    )
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to test")
    parser.add_argument("--output_dir", type=str, default="open_loop_results", help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda, cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for episode selection")
    parser.add_argument(
        "--episode_indices",
        type=str,
        default=None,
        help="Comma-separated list of specific episode indices to evaluate (e.g., '0,5,10')",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Direct path to dataset directory (optional, overrides config repo_id)",
    )
    parser.add_argument(
        "--prompt", type=str, default=None, help="Default prompt/instruction to use for all episodes (optional)"
    )
    parser.add_argument(
        "--compare_rtg", action="store_true", help="Enable RTG vs standard comparison (runs both and plots together)"
    )
    parser.add_argument("--compare_rtg_qp", action="store_true", help="Enable QP-RTG vs original comparison")
    parser.add_argument("--rtg_guidance_steps", type=int, default=3, help="RTG smoothing prefix length")
    parser.add_argument("--rtg_qp_smoothness_weight", type=float, default=1.0, help="QP-RTG smoothness weight")
    parser.add_argument("--rtg_qp_anchor_weight", type=float, default=10.0, help="QP-RTG anchor weight")
    parser.add_argument("--rtg_qp_old_weight", type=float, default=1.0, help="QP-RTG old trajectory weight")
    parser.add_argument("--rtg_qp_new_weight", type=float, default=1.0, help="QP-RTG new chunk weight")
    parser.add_argument("--rtg_qp_velocity_scale", type=float, default=1.25, help="QP-RTG adaptive velocity scale")
    parser.add_argument("--rtg_qp_min_step", type=float, default=1e-3, help="QP-RTG minimum per-step velocity bound")

    args = parser.parse_args()

    # Setup
    setup_logging()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect device if cuda not available
    if args.device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    logging.info("=" * 80)
    logging.info("Pi0/Pi0.5 Open-Loop Evaluation")
    logging.info("=" * 80)
    logging.info(f"Checkpoint: {args.checkpoint_dir}")
    logging.info(f"Config: {args.config_name}")
    logging.info(f"Number of Episodes: {args.num_episodes}")
    logging.info(f"Output Directory: {args.output_dir}")
    logging.info(f"Device: {args.device}")
    logging.info(f"Seed: {args.seed}")
    if args.dataset_path:
        logging.info(f"Dataset Path (override): {args.dataset_path}")
    if args.compare_rtg:
        logging.info("RTG Comparison: ENABLED")
        logging.info(f"  guidance_steps={args.rtg_guidance_steps}")
    if args.compare_rtg_qp:
        logging.info("QP-RTG Comparison: ENABLED")
        logging.info(
            "  guidance_steps=%d, smooth=%.2f, anchor=%.2f, old=%.2f, new=%.2f, vel_scale=%.2f",
            args.rtg_guidance_steps,
            args.rtg_qp_smoothness_weight,
            args.rtg_qp_anchor_weight,
            args.rtg_qp_old_weight,
            args.rtg_qp_new_weight,
            args.rtg_qp_velocity_scale,
        )
    logging.info("")
    logging.info("Open-Loop Mode: Model predicts actions from observations only")
    logging.info("   Ground truth actions are NOT fed back into the model")
    logging.info("=" * 80)

    # Load policy and config
    policy, train_config = load_policy_and_config(args.checkpoint_dir, args.config_name, args.device)

    # Load dataset
    dataset, data_config = load_dataset(train_config, args.dataset_path)

    # Select episodes
    if args.episode_indices is not None:
        # Use specific episode indices provided by user
        episode_indices = [int(idx.strip()) for idx in args.episode_indices.split(",")]
        logging.info(f"Using specified episode indices: {episode_indices}")
    else:
        # Randomly select episodes
        episode_indices = get_episode_indices(dataset, args.num_episodes, args.seed)

    # Process each episode
    logging.info("=" * 80)
    logging.info("Starting open-loop evaluation...")
    logging.info("=" * 80)

    all_metrics = []

    for episode_idx in episode_indices:
        logging.info(f"\n{'='*80}")
        logging.info(f"Processing Episode {episode_idx}")
        logging.info(f"{'='*80}")

        try:
            (
                predicted_actions,
                predicted_actions_rtg,
                predicted_actions_rtg_qp,
                ground_truth_actions,
            ) = predict_episode_actions(
                policy,
                dataset,
                data_config,
                train_config,
                episode_idx,
                args.prompt,
                compare_rtg=args.compare_rtg,
                compare_rtg_qp=args.compare_rtg_qp,
                rtg_guidance_steps=args.rtg_guidance_steps,
                rtg_qp_smoothness_weight=args.rtg_qp_smoothness_weight,
                rtg_qp_anchor_weight=args.rtg_qp_anchor_weight,
                rtg_qp_old_weight=args.rtg_qp_old_weight,
                rtg_qp_new_weight=args.rtg_qp_new_weight,
                rtg_qp_velocity_scale=args.rtg_qp_velocity_scale,
                rtg_qp_min_step=args.rtg_qp_min_step,
            )

            if predicted_actions is not None and ground_truth_actions is not None:
                metrics = plot_episode_comparison(
                    predicted_actions,
                    ground_truth_actions,
                    episode_idx,
                    output_dir,
                    args.config_name,
                    predicted_actions_rtg=predicted_actions_rtg,
                    predicted_actions_rtg_qp=predicted_actions_rtg_qp,
                )
                if metrics:
                    all_metrics.append(metrics)
        except Exception as e:
            logging.error(f"Error processing episode {episode_idx}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Print summary
    logging.info("\n" + "=" * 80)
    logging.info("Evaluation Complete - Summary")
    logging.info("=" * 80)
    logging.info(f"Successfully evaluated {len(all_metrics)} / {len(episode_indices)} episodes")
    logging.info(f"Results saved to: {output_dir}")

    if all_metrics:
        mses_std = [m["mse_std"] for m in all_metrics]
        maes_std = [m["mae_std"] for m in all_metrics]
        jitters_std = [m["jitter_std"] for m in all_metrics]
        jitters_gt = [m["jitter_gt"] for m in all_metrics]

        logging.info("\n--- Standard ---")
        logging.info(f"  MSE:    {np.mean(mses_std):.6f} (std: {np.std(mses_std):.6f})")
        logging.info(f"  MAE:    {np.mean(maes_std):.6f} (std: {np.std(maes_std):.6f})")
        logging.info(f"  Jitter: {np.mean(jitters_std):.6f} (std: {np.std(jitters_std):.6f})")

        if args.compare_rtg:
            mses_rtg = [m["mse_rtg"] for m in all_metrics]
            maes_rtg = [m["mae_rtg"] for m in all_metrics]
            jitters_rtg = [m["jitter_rtg"] for m in all_metrics]

            logging.info("\n--- RTG ---")
            logging.info(f"  MSE:    {np.mean(mses_rtg):.6f} (std: {np.std(mses_rtg):.6f})")
            logging.info(f"  MAE:    {np.mean(maes_rtg):.6f} (std: {np.std(maes_rtg):.6f})")
            logging.info(f"  Jitter: {np.mean(jitters_rtg):.6f} (std: {np.std(jitters_rtg):.6f})")

            logging.info("\n--- Comparison ---")
            logging.info(f"  GT Jitter:  {np.mean(jitters_gt):.6f}")
            logging.info(f"  MSE  delta: {np.mean(mses_rtg) - np.mean(mses_std):+.6f} (RTG - Standard)")
            logging.info(f"  MAE  delta: {np.mean(maes_rtg) - np.mean(maes_std):+.6f} (RTG - Standard)")
            logging.info(f"  Jitter delta: {np.mean(jitters_rtg) - np.mean(jitters_std):+.6f} (RTG - Standard)")
        if args.compare_rtg_qp:
            mses_rtg_qp = [m["mse_rtg_qp"] for m in all_metrics]
            maes_rtg_qp = [m["mae_rtg_qp"] for m in all_metrics]
            jitters_rtg_qp = [m["jitter_rtg_qp"] for m in all_metrics]

            logging.info("\n--- RTG-QP ---")
            logging.info(f"  MSE:    {np.mean(mses_rtg_qp):.6f} (std: {np.std(mses_rtg_qp):.6f})")
            logging.info(f"  MAE:    {np.mean(maes_rtg_qp):.6f} (std: {np.std(maes_rtg_qp):.6f})")
            logging.info(f"  Jitter: {np.mean(jitters_rtg_qp):.6f} (std: {np.std(jitters_rtg_qp):.6f})")

            logging.info("\n--- Comparison ---")
            logging.info(f"  GT Jitter:  {np.mean(jitters_gt):.6f}")
            logging.info(f"  MSE  delta: {np.mean(mses_rtg_qp) - np.mean(mses_std):+.6f} (RTG-QP - Original)")
            logging.info(f"  MAE  delta: {np.mean(maes_rtg_qp) - np.mean(maes_std):+.6f} (RTG-QP - Original)")
            logging.info(f"  Jitter delta: {np.mean(jitters_rtg_qp) - np.mean(jitters_std):+.6f} (RTG-QP - Original)")

    logging.info("=" * 80)


if __name__ == "__main__":
    main()
