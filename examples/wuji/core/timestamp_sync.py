"""
Timestamp synchronizer - aligns data from multiple topics
TimestampSynchronizer - Aligns data from multiple topics based on timestamps

Supports two modes:
1. Strict mode: only returns timestamp-aligned data
2. Relaxed mode (default): returns the latest data, supports interpolation
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SyncedData:
    """Synchronized data structure"""

    timestamp: float  # synchronization timestamp (seconds)
    joint_states: dict[str, list[float]] = field(default_factory=dict)
    images: dict[str, Any] = field(default_factory=dict)

    def is_complete(self, required_joints: list[str], required_images: list[str]) -> bool:
        """Check whether the data is complete"""
        for key in required_joints:
            if key not in self.joint_states:
                return False
        return all(key in self.images for key in required_images)


class TimestampSynchronizer:
    """
    Timestamp synchronizer

    Aligns data from multiple topics by matching timestamps.
    Supports synchronization of joint states and image data.

    Attributes:
        max_time_diff: maximum allowed time difference (seconds)
        buffer_size: buffer size per topic
        required_joints: list of required joint-state topics
        required_images: list of required image topics
    """

    def __init__(
        self,
        max_time_diff: float = 0.05,  # 50ms
        buffer_size: int = 100,
        required_joints: list[str] | None = None,
        required_images: list[str] | None = None,
    ):
        """
        Initialize the synchronizer

        Args:
            max_time_diff: maximum allowed time difference (seconds), default 50ms
            buffer_size: buffer size per topic
            required_joints: list of required joint-state topic keys
            required_images: list of required image topic keys
        """
        self.max_time_diff = max_time_diff
        self.buffer_size = buffer_size
        self.required_joints = required_joints or []
        self.required_images = required_images or []

        # Data buffers: {topic_key: deque of (timestamp, data)}
        self._joint_buffers: dict[str, deque] = {}
        self._image_buffers: dict[str, deque] = {}

        # Initialize buffers
        for key in self.required_joints:
            self._joint_buffers[key] = deque(maxlen=buffer_size)
        for key in self.required_images:
            self._image_buffers[key] = deque(maxlen=buffer_size)

        # Thread lock
        self._lock = threading.Lock()

        # Callback function
        self._sync_callback: Callable[[SyncedData], None] | None = None

        # Last synchronization time
        self._last_sync_time: float = -1.0
        self._last_debug_time: float = 0
        self._first_sync_logged: bool = False

    def set_sync_callback(self, callback: Callable[[SyncedData], None]) -> None:
        """Set the callback invoked when synchronization completes"""
        self._sync_callback = callback

    def add_joint_data(self, key: str, timestamp: float, data: list[float]) -> None:
        """
        Add joint-state data

        Args:
            key: topic key (e.g. 'left_arm', 'right_hand')
            timestamp: timestamp (seconds)
            data: list of joint angles
        """
        with self._lock:
            if key not in self._joint_buffers:
                self._joint_buffers[key] = deque(maxlen=self.buffer_size)
            self._joint_buffers[key].append((timestamp, data))

        self._try_sync()

    def add_image_data(self, key: str, timestamp: float, data: Any) -> None:
        """
        Add image data

        Args:
            key: topic key (e.g. 'cam_high', 'cam_left_wrist')
            timestamp: timestamp (seconds)
            data: image data (numpy array)
        """
        with self._lock:
            if key not in self._image_buffers:
                self._image_buffers[key] = deque(maxlen=self.buffer_size)
            self._image_buffers[key].append((timestamp, data))

        self._try_sync()

    def _try_sync(self) -> SyncedData | None:
        """
        Attempt to synchronize data - simplified version using the latest data
        """
        with self._lock:
            # Check whether all required buffers have data
            missing_joints = []
            missing_images = []

            missing_joints = [
                key
                for key in self.required_joints
                if key not in self._joint_buffers or len(self._joint_buffers[key]) == 0
            ]
            missing_images = [
                key
                for key in self.required_images
                if key not in self._image_buffers or len(self._image_buffers[key]) == 0
            ]

            if missing_joints or missing_images:
                # Log missing information every 5 seconds
                current_time = time.time()
                if current_time - self._last_debug_time > 5:
                    self._last_debug_time = current_time
                    if missing_joints:
                        logger.warning(f"Missing joint data: {missing_joints}")
                    if missing_images:
                        logger.warning(f"Missing image data: {missing_images}")
                    # Log the state of the available buffers
                    for key, buf in self._joint_buffers.items():
                        if len(buf) > 0:
                            logger.info(f"  Joint {key}: {len(buf)} entries, latest timestamp: {buf[-1][0]:.3f}")
                    for key, buf in self._image_buffers.items():
                        if len(buf) > 0:
                            logger.info(f"  Image {key}: {len(buf)} entries, latest timestamp: {buf[-1][0]:.3f}")
                return None

            # Use the current time as the reference
            ref_timestamp = time.time()

            # Build the synced data - use the latest data directly
            synced = SyncedData(timestamp=ref_timestamp)

            # Get joint data (using the latest entry)
            for key in self.required_joints:
                if self._joint_buffers[key]:
                    synced.joint_states[key] = self._joint_buffers[key][-1][1]

            # Get image data (using the latest entry)
            for key in self.required_images:
                if self._image_buffers[key]:
                    synced.images[key] = self._image_buffers[key][-1][1]

            # Check whether the data is complete
            if synced.is_complete(self.required_joints, self.required_images):
                # Log timestamp info (first time only)
                if not self._first_sync_logged:
                    self._first_sync_logged = True
                    logger.info("=== First synchronization succeeded, timestamp info ===")
                    for key, buf in self._joint_buffers.items():
                        if len(buf) > 0:
                            logger.info(f"  Joint {key}: timestamp {buf[-1][0]:.3f}")
                    for key, buf in self._image_buffers.items():
                        if len(buf) > 0:
                            logger.info(f"  Image {key}: timestamp {buf[-1][0]:.3f}")

                return synced

        return None

    def _find_closest(self, buffer: deque, ref_timestamp: float) -> Any | None:
        """
        Find the buffer entry whose timestamp is closest to the reference timestamp

        Args:
            buffer: data buffer
            ref_timestamp: reference timestamp

        Returns:
            The closest entry; if the time difference exceeds the threshold, returns
            the latest entry (relaxed mode).
        """
        if not buffer:
            return None

        best_data = None
        best_diff = float("inf")

        for timestamp, data in buffer:
            diff = abs(timestamp - ref_timestamp)
            if diff < best_diff:
                best_diff = diff
                best_data = data

        if best_diff <= self.max_time_diff:
            return best_data

        # Relaxed mode: if the time difference exceeds the threshold, return the latest entry
        if buffer:
            return buffer[-1][1]
        return None

    def _cleanup_old_data(self, current_timestamp: float) -> None:
        """Remove stale data"""
        cutoff = current_timestamp - self.max_time_diff * 2

        for buffer in self._joint_buffers.values():
            while buffer and buffer[0][0] < cutoff:
                buffer.popleft()

        for buffer in self._image_buffers.values():
            while buffer and buffer[0][0] < cutoff:
                buffer.popleft()

    def get_latest_synced(self) -> SyncedData | None:
        """Get the latest synchronized data (does not trigger the callback)"""
        return self._try_sync()

    def clear(self) -> None:
        """Clear all buffers"""
        with self._lock:
            for buffer in self._joint_buffers.values():
                buffer.clear()
            for buffer in self._image_buffers.values():
                buffer.clear()
            self._last_sync_time = -1.0
            self._first_sync_logged = False
