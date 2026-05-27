#!/usr/bin/env python3
"""
OpenPI ROS2 environment implementation

Implements the openpi_client.runtime.environment.Environment interface
for use with the OpenPI Runtime framework.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np
from openpi_client.runtime import environment as _environment
import rclpy
from rclpy.node import Node
from typing_extensions import override

from wuji.core import DeployConfig
from wuji.core import ROS2Interface
from wuji.core import SyncedData
from wuji.core.utils import build_observation_state

logger = logging.getLogger(__name__)


class OpenPIRosEnvironment(_environment.Environment):
    """
    Wuji robot ROS2 environment (OpenPI)

    Implements the Environment interface and interacts with the robot through ROS2 topics:
    - Subscribes to joint states and camera images
    - Uses timestamp synchronization to align data from multiple topics
    - Publishes action commands

    Supports single-arm (27D) or dual-arm (54D) control
    """

    def __init__(self, node: Node, config: DeployConfig):
        """
        Initialize the OpenPI ROS2 environment

        Args:
            node: ROS2 node
            config: deployment configuration
        """
        self._node = node
        self._config = config

        # Create the ROS2 interface
        self._ros_interface = ROS2Interface(node, config)

        # Thread-safe data storage
        self._data_lock = threading.Lock()
        self._latest_synced: SyncedData | None = None

        logger.info(f"OpenPI environment initialization complete, arm_mode={config.arm_mode}")
        logger.info(f"Task instruction: {config.prompt}")

        # Wait for data to be ready
        logger.info("Waiting for ROS2 data to be ready...")
        if not self._ros_interface.wait_for_data(timeout=config.data_timeout_s):
            raise RuntimeError(
                f"Timed out waiting for ROS2 data ({config.data_timeout_s}s).\n"
                f"Required joints: {config.get_required_joints()}\n"
                f"Required cameras: {config.get_required_cameras()}\n"
                "Please verify that the topics are publishing data."
            )

    @override
    def reset(self) -> None:
        """Reset the environment"""
        logger.info("Resetting environment...")

        # Clear the buffers
        self._ros_interface.clear()
        with self._data_lock:
            self._latest_synced = None

        # Wait for new data
        timeout = 10.0
        start_time = time.time()

        while time.time() - start_time < timeout:
            rclpy.spin_once(self._node, timeout_sec=0.1)
            synced = self._ros_interface.get_synced_data(timeout=0.1)
            if synced is not None:
                with self._data_lock:
                    self._latest_synced = synced
                logger.info("Environment reset complete, data ready")
                return

        logger.warning("Environment reset timed out, continuing anyway")

    @override
    def is_episode_complete(self) -> bool:
        """Check whether the episode is complete (controlled by Runtime's max_episode_steps)"""
        return False

    @override
    def get_observation(self) -> dict:
        """
        Get the current observation

        Returns:
            Observation dictionary in the format expected by the OpenPI policy:
            - observation/state: state vector (numpy array)
            - prompt: task language instruction
            - observation/image: head camera image
            - observation/wrist_image: wrist camera image
        """
        # Process ROS2 callbacks to fetch the latest data
        rclpy.spin_once(self._node, timeout_sec=0.001)

        # Get synced data
        synced = self._ros_interface.get_synced_data(timeout=0.1)
        if synced is not None:
            with self._data_lock:
                self._latest_synced = synced

        with self._data_lock:
            if self._latest_synced is None:
                raise RuntimeError("Synchronized data is not ready")
            synced_data = self._latest_synced

        # Build the state vector
        # Order: left_arm(7), left_hand(20), right_arm(7), right_hand(20)
        state = build_observation_state(
            synced_data,
            arm_mode=self._config.arm_mode,
            use_ee_pose=self._config.use_ee_pose,
        )

        if len(state) == 0:
            raise RuntimeError("Failed to build the state vector")

        # Build the observation dictionary
        obs = {
            "observation/state": state,
            "prompt": self._config.prompt,
        }

        # Add images
        # stereo -> observation/image (base/head camera)
        if "stereo" in synced_data.images:
            img = synced_data.images["stereo"]
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            obs["observation/image"] = img.copy()

        # Add wrist cameras (supports multiple cameras in dual-arm mode)
        if self._config.arm_mode in ("single_left", "dual", "both") and "cam_left_wrist" in synced_data.images:
            img = synced_data.images["cam_left_wrist"]
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            obs["observation/left_wrist_image"] = img.copy()

        if self._config.arm_mode in ("single_right", "dual", "both") and "cam_right_wrist" in synced_data.images:
            img = synced_data.images["cam_right_wrist"]
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            obs["observation/right_wrist_image"] = img.copy()

        return obs

    @override
    def apply_action(self, action: dict) -> None:
        """
        Execute an action

        Args:
            action: action dictionary from ActionChunkBroker
                - actions or action: action array
        """
        # Extract the action array from the action dictionary
        action_array = action.get("actions")
        if action_array is None:
            action_array = action.get("action")

        if action_array is None:
            logger.warning("Could not find 'actions' or 'action' key in the action dictionary")
            return

        # Ensure it is a numpy array
        if not isinstance(action_array, np.ndarray):
            action_array = np.array(action_array, dtype=np.float32)

        # If the array is multi-dimensional, take the first row
        if len(action_array.shape) > 1:
            action_array = action_array[0]

        # Publish the action
        self._ros_interface.publish_action(action_array)
