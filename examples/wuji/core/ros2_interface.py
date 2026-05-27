#!/usr/bin/env python3
"""
ROS2 interface layer

Provides unified ROS2 topic subscription and publishing for use with OpenPI.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
import threading
import time
from typing import Any

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState

from .config import DeployConfig
from .timestamp_sync import SyncedData
from .timestamp_sync import TimestampSynchronizer
from .utils import convert_joints_to_hardware
from .utils import convert_joints_to_model

logger = logging.getLogger(__name__)


class ROS2Interface:
    """
    Unified ROS2 interface

    Provides:
    - Joint state subscription (JointState)
    - Camera image subscription (Image)
    - End-effector pose subscription (Pose, optional)
    - Action command publishing (JointState / Float32MultiArray)
    - Timestamp synchronization

    Usage:
        config = DeployConfig.from_yaml("config/deploy.yaml")
        interface = ROS2Interface(node, config)

        # Get synchronized data
        synced = interface.get_synced_data()

        # Publish an action
        interface.publish_action(action_array)
    """

    def __init__(
        self,
        node: Node,
        config: DeployConfig,
        on_synced_data: Callable[[SyncedData], None] | None = None,
    ):
        """
        Initialize the ROS2 interface

        Args:
            node: ROS2 node
            config: deployment configuration
            on_synced_data: synced data callback (optional)
        """
        self._node = node
        self._config = config
        self._on_synced_data = on_synced_data

        # CV Bridge
        self._bridge = CvBridge()

        # QoS configuration
        self._qos = self._create_qos_profile()

        # Storage
        self._joint_subscribers: dict[str, Any] = {}
        self._camera_subscribers: dict[str, Any] = {}
        self._ee_subscribers: dict[str, Any] = {}
        self._action_publishers: dict[str, Any] = {}

        # Timestamp synchronizer
        self._synchronizer = TimestampSynchronizer(
            max_time_diff=config.max_time_diff,
            buffer_size=config.sync_buffer_size,
            required_joints=config.get_required_joints(),
            required_images=config.get_required_cameras(),
        )

        # Thread-safe data storage
        self._data_lock = threading.Lock()
        self._latest_synced: SyncedData | None = None
        self._data_ready_event = threading.Event()

        # Create subscribers and publishers
        self._create_joint_subscribers()
        self._create_camera_subscribers()
        if config.use_ee_pose:
            self._create_ee_subscribers()
        self._create_action_publishers()

        logger.info(f"ROS2Interface initialization complete, arm_type={config.arm_type}, arm_mode={config.arm_mode}")
        logger.info(
            f"  Arm hardware unit: {config.get_arm_hw_unit()}, dexterous-hand hardware unit: {config.get_hand_hw_unit()}"
        )

    def _create_qos_profile(self) -> QoSProfile:
        """Create the QoS profile"""
        reliability = (
            QoSReliabilityPolicy.BEST_EFFORT
            if self._config.qos_reliability == "best_effort"
            else QoSReliabilityPolicy.RELIABLE
        )
        return QoSProfile(
            reliability=reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=self._config.qos_history_depth,
        )

    def _create_joint_subscribers(self) -> None:
        """Create joint state subscribers"""
        topics = self._config.get_joint_state_topics()
        for key, cfg in topics.items():
            if not cfg.get("enabled", True):
                continue
            topic = cfg["topic"]
            callback = self._make_joint_callback(key)
            sub = self._node.create_subscription(JointState, topic, callback, self._qos)
            self._joint_subscribers[key] = sub
            logger.info(f"Subscribed to joint topic: {topic} (key: {key})")

    def _create_camera_subscribers(self) -> None:
        """Create camera subscribers"""
        topics = self._config.get_camera_topics()
        for key, cfg in topics.items():
            if not cfg.get("enabled", True):
                continue
            topic = cfg["topic"]

            # Detect whether this is a compressed image topic
            if "compressed" in topic.lower():
                callback = self._make_compressed_camera_callback(key)
                sub = self._node.create_subscription(CompressedImage, topic, callback, self._qos)
                logger.info(f"Subscribed to compressed camera topic: {topic} (key: {key})")
            else:
                callback = self._make_camera_callback(key)
                sub = self._node.create_subscription(Image, topic, callback, self._qos)
                logger.info(f"Subscribed to camera topic: {topic} (key: {key})")

            self._camera_subscribers[key] = sub

    def _create_ee_subscribers(self) -> None:
        """Create end-effector pose subscribers"""
        if self._config.use_left_arm():
            topic = self._config.left_ee_pose_topic
            callback = self._make_ee_callback("left_ee")
            sub = self._node.create_subscription(Pose, topic, callback, self._qos)
            self._ee_subscribers["left_ee"] = sub
            logger.info(f"Subscribed to end-effector pose topic: {topic} (key: left_ee)")

        if self._config.use_right_arm():
            topic = self._config.right_ee_pose_topic
            callback = self._make_ee_callback("right_ee")
            sub = self._node.create_subscription(Pose, topic, callback, self._qos)
            self._ee_subscribers["right_ee"] = sub
            logger.info(f"Subscribed to end-effector pose topic: {topic} (key: right_ee)")

    def _create_action_publishers(self) -> None:
        """Create action command publishers"""
        topics = self._config.get_action_cmd_topics()
        for key, cfg in topics.items():
            if not cfg.get("enabled", True):
                continue
            topic = cfg["topic"]
            pub = self._node.create_publisher(JointState, topic, self._qos)
            self._action_publishers[key] = pub
            logger.info(f"Created action publisher: {topic} (key: {key})")

    def _make_joint_callback(self, key: str) -> Callable:
        """Create joint state callback (includes unit conversion: hardware -> model)"""
        # Determine the hardware unit
        hw_unit = self._config.get_arm_hw_unit() if "arm" in key else self._config.get_hand_hw_unit()

        def callback(msg: JointState) -> None:
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if timestamp == 0.0:
                timestamp = time.time()

            # Unit conversion: hardware unit -> radians (model unit)
            joint_data = list(msg.position)
            joint_data = convert_joints_to_model(joint_data, hw_unit)

            self._synchronizer.add_joint_data(key, timestamp, joint_data)
            self._try_update_synced()

        return callback

    def _make_camera_callback(self, key: str) -> Callable:
        """Create camera callback"""

        def callback(msg: Image) -> None:
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if timestamp == 0.0:
                timestamp = time.time()
            try:
                cv_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
                self._synchronizer.add_image_data(key, timestamp, cv_image)
                self._try_update_synced()
            except Exception as e:
                logger.debug(f"Image conversion failed for {key}: {e}")

        return callback

    def _make_compressed_camera_callback(self, key: str) -> Callable:
        """Create compressed camera callback"""

        def callback(msg: CompressedImage) -> None:
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if timestamp == 0.0:
                timestamp = time.time()
            try:
                # Decode the compressed image
                np_arr = np.frombuffer(msg.data, np.uint8)
                bgr_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                # Convert BGR to RGB
                rgb_image = bgr_image[:, :, ::-1].copy()
                self._synchronizer.add_image_data(key, timestamp, rgb_image)
                self._try_update_synced()
            except Exception as e:
                logger.debug(f"Compressed image conversion failed for {key}: {e}")

        return callback

    def _make_ee_callback(self, key: str) -> Callable:
        """Create end-effector pose callback"""

        def callback(msg: Pose) -> None:
            timestamp = time.time()  # Pose has no header
            ee_data = self._pose_to_6dof(msg)
            self._synchronizer.add_joint_data(key, timestamp, ee_data)
            self._try_update_synced()

        return callback

    def _pose_to_6dof(self, msg: Pose) -> list[float]:
        """Convert a Pose message to a 6DOF list [px, py, pz, ax, ay, az]"""
        from .utils import quaternion_to_euler

        roll, pitch, yaw = quaternion_to_euler(
            msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        )
        return [
            float(msg.position.x),
            float(msg.position.y),
            float(msg.position.z),
            float(roll),
            float(pitch),
            float(yaw),
        ]

    def _try_update_synced(self) -> None:
        """Try to update the synchronized data"""
        synced = self._synchronizer.get_latest_synced()
        if synced is not None:
            with self._data_lock:
                self._latest_synced = synced
            self._data_ready_event.set()
            if self._on_synced_data:
                self._on_synced_data(synced)

    def get_synced_data(self, timeout: float = 1.0) -> SyncedData | None:
        """
        Get the latest synchronized data

        Args:
            timeout: wait timeout (seconds)

        Returns:
            Synchronized data, or None if the wait timed out
        """
        if self._data_ready_event.wait(timeout=timeout):
            with self._data_lock:
                return self._latest_synced
        return None

    def wait_for_data(self, timeout: float = 30.0) -> bool:
        """
        Wait for data to be ready

        Args:
            timeout: timeout (seconds)

        Returns:
            True if data is ready, False if it timed out
        """
        start_time = time.time()
        last_log_time = 0

        while True:
            elapsed = time.time() - start_time

            # Process ROS2 callbacks
            rclpy.spin_once(self._node, timeout_sec=0.1)

            # Check data
            synced = self._synchronizer.get_latest_synced()
            if synced is not None:
                with self._data_lock:
                    self._latest_synced = synced
                self._data_ready_event.set()
                logger.info(f"Data ready ({elapsed:.1f}s)")
                return True

            # Check for timeout
            if elapsed > timeout:
                logger.error(f"Timed out waiting for data ({timeout}s)")
                return False

            # Periodic logging
            current_second = int(elapsed)
            if current_second >= last_log_time + 5:
                last_log_time = current_second
                logger.info(f"Waiting for data... ({elapsed:.0f}s)")

    def publish_action(self, action: np.ndarray) -> None:
        """
        Publish an action command

        Args:
            action: action array
                - 27D: arm(7) + hand(20) single arm
                - 54D: left_arm(7) + left_hand(20) + right_arm(7) + right_hand(20) dual arm
        """
        action_dim = len(action)
        stamp = self._node.get_clock().now().to_msg()

        if action_dim == 27:
            # Single-arm mode
            arm_cmd = action[:7]
            hand_cmd = action[7:27]

            if self._config.use_left_arm() and not self._config.is_dual_arm():
                self._publish_joint_cmd("left_arm", arm_cmd, stamp)
                self._publish_joint_cmd("left_hand", hand_cmd, stamp)
            else:
                self._publish_joint_cmd("right_arm", arm_cmd, stamp)
                self._publish_joint_cmd("right_hand", hand_cmd, stamp)

        elif action_dim == 54:
            # Dual-arm mode
            left_arm_cmd = action[:7]
            left_hand_cmd = action[7:27]
            right_arm_cmd = action[27:34]
            right_hand_cmd = action[34:54]

            if self._config.use_left_arm():
                self._publish_joint_cmd("left_arm", left_arm_cmd, stamp)
                self._publish_joint_cmd("left_hand", left_hand_cmd, stamp)

            if self._config.use_right_arm():
                self._publish_joint_cmd("right_arm", right_arm_cmd, stamp)
                self._publish_joint_cmd("right_hand", right_hand_cmd, stamp)

        else:
            logger.warning(f"Unknown action dimension: {action_dim}")

    def publish_action_dict(self, action_dict: dict[str, np.ndarray]) -> None:
        """
        Publish action commands (dictionary format)

        Args:
            action_dict: action dictionary with keys "left_arm", "left_hand", "right_arm", "right_hand"
        """
        stamp = self._node.get_clock().now().to_msg()

        for key, cmd in action_dict.items():
            if key in self._action_publishers:
                self._publish_joint_cmd(key, cmd, stamp)

    def _publish_joint_cmd(self, key: str, cmd: np.ndarray, stamp) -> None:
        """Publish a joint command (includes unit conversion: model -> hardware)"""
        if key not in self._action_publishers:
            return

        # Determine the hardware unit and convert
        hw_unit = self._config.get_arm_hw_unit() if "arm" in key else self._config.get_hand_hw_unit()

        # Unit conversion: radians (model unit) -> hardware unit
        cmd_list = cmd.tolist() if isinstance(cmd, np.ndarray) else list(cmd)
        cmd_list = convert_joints_to_hardware(cmd_list, hw_unit)

        msg = JointState()
        msg.header.stamp = stamp
        msg.header.frame_id = ""

        # Only set position, matching wuji_client behavior
        msg.position = [float(x) for x in cmd_list]

        self._action_publishers[key].publish(msg)

    def clear(self) -> None:
        """Clear the buffers"""
        self._synchronizer.clear()
        with self._data_lock:
            self._latest_synced = None
        self._data_ready_event.clear()

    @property
    def action_publishers(self) -> dict[str, Any]:
        """Get the action publishers (legacy-interface compatible)"""
        return self._action_publishers

    @property
    def synchronizer(self) -> TimestampSynchronizer:
        """Get the timestamp synchronizer"""
        return self._synchronizer
