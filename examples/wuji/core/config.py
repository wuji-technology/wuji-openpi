#!/usr/bin/env python3
"""
Unified configuration module

Provides the configuration structure for the OpenPI control pipeline.
Supports loading from a YAML file or configuring via Python dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class TopicConfig:
    """Configuration for a single ROS2 topic"""

    topic: str
    enabled: bool = True
    msg_type: str = "JointState"  # JointState, Float32MultiArray, Image, Pose


@dataclass
class DeployConfig:
    """
    Unified deployment configuration

    Supports the OpenPI control pipeline and provides a unified ROS2 topic configuration.

    Arm Type:
        - "wuji": Wuji arm, low-level control uses degrees
        - "agilex": AgileX arm, low-level control uses degrees
        - "custom": custom; units must be configured manually

    Arm Mode:
        - "single_right": right arm only (default)
        - "single_left": left arm only
        - "dual" / "both": dual arms

    EE Pose:
        - use_ee_pose: whether to include the end-effector pose
        - ee_pose_dof: 6 DOF [px, py, pz, ax, ay, az]

    Unit Convention:
        - Model inference always uses radians
        - Hardware units are configured automatically based on arm_type, or specified
          manually via arm_hw_unit/hand_hw_unit
        - When reading observations: hardware units -> radians
        - When publishing actions: radians -> hardware units
    """

    # =========================================================================
    # Basic configuration
    # =========================================================================
    id: str = "wuji_robot"
    arm_mode: Literal["single_right", "single_left", "dual", "both"] = "single_right"
    use_ee_pose: bool = False

    # =========================================================================
    # Arm type configuration
    # =========================================================================
    # Arm type: "wuji", "agilex", "custom"
    arm_type: Literal["wuji", "agilex", "custom"] = "wuji"

    # Hardware unit configuration (only needed when arm_type="custom")
    # "degree": hardware uses degrees
    # "radian": hardware uses radians
    arm_hw_unit: Literal["degree", "radian"] = "degree"
    hand_hw_unit: Literal["degree", "radian"] = "degree"

    # Robot degrees of freedom
    arm_dof: int = 7
    hand_dof: int = 20
    ee_pose_dof: int = 6

    # =========================================================================
    # Control configuration
    # =========================================================================
    control_hz: float = 30.0
    max_time_diff: float = 0.05  # Timestamp synchronization tolerance (seconds)

    # =========================================================================
    # OpenPI-specific configuration
    # =========================================================================
    server_host: str = "localhost"
    server_port: int = 12346
    action_horizon: int = 50
    num_episodes: int = 1
    max_episode_steps: int = 1000000
    prompt: str = "pick up the cube"

    # =========================================================================
    # Action broker configuration
    # =========================================================================
    broker_mode: Literal["serial", "rtg", "rtg_qp"] | None = None

    # =========================================================================
    # RTG (Real-Time Trajectory Generation) configuration
    # =========================================================================
    rtg_enabled: bool = False
    rtg_trigger_fraction: float = 0.5  # trigger next inference once this fraction of the chunk is consumed
    rtg_guidance_steps: int = (
        3  # per the reference paper, use a short guidance prefix and apply cubic interpolation smoothing
    )
    rtg_qp_enabled: bool = False
    rtg_qp_smoothness_weight: float = 1.0
    rtg_qp_anchor_weight: float = 10.0
    rtg_qp_old_weight: float = 1.0
    rtg_qp_new_weight: float = 1.0
    rtg_qp_velocity_scale: float = 1.25
    rtg_qp_min_step: float = 1e-3

    # =========================================================================
    # ROS2 topic configuration - joint state subscriptions
    # =========================================================================
    left_arm_joint_topic: str = "/tianji_arm/left/joint_state"
    right_arm_joint_topic: str = "/tianji_arm/right/joint_state"
    left_hand_angle_topic: str = "/wuji_hand/left/joint_state"
    right_hand_angle_topic: str = "/wuji_hand/right/joint_state"

    # =========================================================================
    # ROS2 topic configuration - end-effector pose subscriptions (optional)
    # =========================================================================
    left_ee_pose_topic: str = "/left_ee_pose"
    right_ee_pose_topic: str = "/right_ee_pose"

    # =========================================================================
    # ROS2 topic configuration - camera subscriptions
    # =========================================================================
    camera_head_topic: str = "/stereo/right/compressed"
    camera_left_wrist_topic: str = "/cam_left_wrist/color/image_rect_raw/compressed"
    camera_right_wrist_topic: str = "/cam_right_wrist/color/image_rect_raw/compressed"

    # =========================================================================
    # ROS2 topic configuration - action command publishers
    # =========================================================================
    left_arm_cmd_topic: str = "/tianji_arm/left/joint_command"
    right_arm_cmd_topic: str = "/tianji_arm/right/joint_command"
    left_hand_cmd_topic: str = "/wuji_hand/left/joint_command"
    right_hand_cmd_topic: str = "/wuji_hand/right/joint_command"

    # =========================================================================
    # Camera shape configuration
    # =========================================================================
    camera_head_shape: tuple = (480, 640, 3)
    camera_left_wrist_shape: tuple = (480, 640, 3)
    camera_right_wrist_shape: tuple = (480, 640, 3)

    # =========================================================================
    # Time synchronization configuration
    # =========================================================================
    sync_slop: float = 0.1
    sync_queue_size: int = 10
    sync_buffer_size: int = 100

    # =========================================================================
    # Timeout configuration
    # =========================================================================
    data_timeout_s: float = 30.0
    observation_timeout_s: float = 1.0
    connect_timeout_s: float = 10.0

    # =========================================================================
    # QoS configuration
    # =========================================================================
    qos_reliability: str = "best_effort"
    qos_history_depth: int = 1

    # =========================================================================
    # Initial position configuration
    # =========================================================================
    initial_left_arm_positions: list[float] | None = None
    initial_right_arm_positions: list[float] | None = None

    # =========================================================================
    # Helper methods
    # =========================================================================

    @property
    def side(self) -> str:
        """Side parameter for OpenPI compatibility"""
        if self.arm_mode == "single_right":
            return "right"
        if self.arm_mode == "single_left":
            return "left"
        return "both"

    def use_left_arm(self) -> bool:
        """Whether the left arm is used"""
        return self.arm_mode in ("single_left", "dual", "both")

    def use_right_arm(self) -> bool:
        """Whether the right arm is used"""
        return self.arm_mode in ("single_right", "dual", "both")

    def is_dual_arm(self) -> bool:
        """Whether the mode is dual-arm"""
        return self.arm_mode in ("dual", "both")

    def get_state_dim(self) -> int:
        """Get the state vector dimension"""
        arm_count = 2 if self.is_dual_arm() else 1
        single_arm_dim = self.arm_dof + self.hand_dof
        if self.use_ee_pose:
            single_arm_dim += self.ee_pose_dof
        return single_arm_dim * arm_count

    def get_arm_hw_unit(self) -> str:
        """
        Get the arm hardware unit

        Returns the hardware unit based on arm_type:
        - wuji: radian (marked as radians to disable unit conversion; the hardware
          actually outputs degrees, but no conversion was applied during training)
        - agilex: degree
        - custom: uses the arm_hw_unit configuration

        Returns:
            "degree" or "radian"
        """
        if self.arm_type == "wuji":
            return "radian"  # disable conversion to preserve training-time mixed units (arm in degrees + dexterous hand in radians)
        if self.arm_type == "agilex":
            return "degree"
        # custom
        return self.arm_hw_unit

    def get_hand_hw_unit(self) -> str:
        """
        Get the dexterous-hand hardware unit

        Returns the hardware unit based on arm_type:
        - wuji: radian - the Wuji dexterous-hand hardware uses radians
        - agilex: degree
        - custom: uses the hand_hw_unit configuration

        Returns:
            "degree" or "radian"
        """
        if self.arm_type == "wuji":
            return "radian"  # the Wuji dexterous hand uses radians, no conversion needed
        if self.arm_type == "agilex":
            return "degree"
        # custom
        return self.hand_hw_unit

    def need_arm_unit_conversion(self) -> bool:
        """Whether arm unit conversion is needed (required when hardware uses degrees)"""
        return self.get_arm_hw_unit() == "degree"

    def need_hand_unit_conversion(self) -> bool:
        """Whether dexterous-hand unit conversion is needed (required when hardware uses degrees)"""
        return self.get_hand_hw_unit() == "degree"

    def get_broker_mode(self) -> Literal["serial", "rtg", "rtg_qp"]:
        """Resolve the action broker mode to use for the current deployment."""
        if self.broker_mode is not None:
            return self.broker_mode

        enabled_modes = [
            mode
            for mode, enabled in (
                ("rtg", self.rtg_enabled),
                ("rtg_qp", self.rtg_qp_enabled),
            )
            if enabled
        ]
        if len(enabled_modes) > 1:
            raise ValueError(f"Multiple broker flags enabled simultaneously: {enabled_modes}")

        if enabled_modes:
            return enabled_modes[0]
        return "serial"

    def get_required_joints(self) -> list[str]:
        """Get the joint topic keys that need to be subscribed to"""
        joints = []
        if self.use_left_arm():
            joints.extend(["left_arm", "left_hand"])
        if self.use_right_arm():
            joints.extend(["right_arm", "right_hand"])
        return joints

    def get_required_cameras(self) -> list[str]:
        """Get the camera topic keys that need to be subscribed to"""
        cameras = ["stereo"]
        if self.use_left_arm():
            cameras.append("cam_left_wrist")
        if self.use_right_arm():
            cameras.append("cam_right_wrist")
        return cameras

    def get_joint_state_topics(self) -> dict[str, dict[str, Any]]:
        """Build the joint-state topic configuration dictionary (OpenPI-compatible format)"""
        topics = {}
        if self.use_left_arm():
            topics["left_arm"] = {"topic": self.left_arm_joint_topic, "enabled": True}
            topics["left_hand"] = {"topic": self.left_hand_angle_topic, "enabled": True}
        if self.use_right_arm():
            topics["right_arm"] = {"topic": self.right_arm_joint_topic, "enabled": True}
            topics["right_hand"] = {"topic": self.right_hand_angle_topic, "enabled": True}
        return topics

    def get_camera_topics(self) -> dict[str, dict[str, Any]]:
        """Build the camera topic configuration dictionary (OpenPI-compatible format)"""
        topics = {"stereo": {"topic": self.camera_head_topic, "enabled": True}}
        if self.use_left_arm():
            topics["cam_left_wrist"] = {"topic": self.camera_left_wrist_topic, "enabled": True}
        if self.use_right_arm():
            topics["cam_right_wrist"] = {"topic": self.camera_right_wrist_topic, "enabled": True}
        return topics

    def get_action_cmd_topics(self) -> dict[str, dict[str, Any]]:
        """Build the action command topic configuration dictionary (OpenPI-compatible format)"""
        topics = {}
        if self.use_left_arm():
            topics["left_arm"] = {"topic": self.left_arm_cmd_topic, "enabled": True}
            topics["left_hand"] = {"topic": self.left_hand_cmd_topic, "enabled": True}
        if self.use_right_arm():
            topics["right_arm"] = {"topic": self.right_arm_cmd_topic, "enabled": True}
            topics["right_hand"] = {"topic": self.right_hand_cmd_topic, "enabled": True}
        return topics

    @classmethod
    def from_yaml(cls, yaml_path: str) -> DeployConfig:
        """Load configuration from a YAML file"""
        path = Path(yaml_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Field mapping (compatibility with legacy configuration format)
        field_mapping = {
            "host": "server_host",
            "port": "server_port",
            "inference_rate": "control_hz",
            "side": "arm_mode",  # requires special handling
        }

        # Handle side -> arm_mode conversion
        if "side" in data:
            side_value = data.pop("side")
            if side_value == "left":
                data["arm_mode"] = "single_left"
            elif side_value == "right":
                data["arm_mode"] = "single_right"
            elif side_value == "both":
                data["arm_mode"] = "dual"

        # Apply field mapping
        for old_key, new_key in field_mapping.items():
            if old_key in data and new_key not in data:
                data[new_key] = data.pop(old_key)

        if "broker_mode" not in data:
            if data.get("rtg_qp_enabled"):
                data["broker_mode"] = "rtg_qp"
            elif data.get("rtg_enabled"):
                data["broker_mode"] = "rtg"

        # Handle nested topic configurations
        if "joint_state_topics" in data:
            jst = data.pop("joint_state_topics")
            if "left_arm" in jst:
                data["left_arm_joint_topic"] = jst["left_arm"].get("topic", data.get("left_arm_joint_topic"))
            if "right_arm" in jst:
                data["right_arm_joint_topic"] = jst["right_arm"].get("topic", data.get("right_arm_joint_topic"))
            if "left_hand" in jst:
                data["left_hand_angle_topic"] = jst["left_hand"].get("topic", data.get("left_hand_angle_topic"))
            if "right_hand" in jst:
                data["right_hand_angle_topic"] = jst["right_hand"].get("topic", data.get("right_hand_angle_topic"))

        if "camera_topics" in data:
            ct = data.pop("camera_topics")
            if "cam_high" in ct:
                data["camera_head_topic"] = ct["cam_high"].get("topic", data.get("camera_head_topic"))
            if "cam_left_wrist" in ct:
                data["camera_left_wrist_topic"] = ct["cam_left_wrist"].get("topic", data.get("camera_left_wrist_topic"))
            if "cam_right_wrist" in ct:
                data["camera_right_wrist_topic"] = ct["cam_right_wrist"].get(
                    "topic", data.get("camera_right_wrist_topic")
                )

        if "action_cmd_topics" in data:
            act = data.pop("action_cmd_topics")
            if "left_arm" in act:
                data["left_arm_cmd_topic"] = act["left_arm"].get("topic", data.get("left_arm_cmd_topic"))
            if "right_arm" in act:
                data["right_arm_cmd_topic"] = act["right_arm"].get("topic", data.get("right_arm_cmd_topic"))
            if "left_hand" in act:
                data["left_hand_cmd_topic"] = act["left_hand"].get("topic", data.get("left_hand_cmd_topic"))
            if "right_hand" in act:
                data["right_hand_cmd_topic"] = act["right_hand"].get("topic", data.get("right_hand_cmd_topic"))

        # Handle initial positions
        if "initial_arm_positions" in data:
            iap = data.pop("initial_arm_positions")
            if "left_arm" in iap:
                data["initial_left_arm_positions"] = iap["left_arm"]
            if "right_arm" in iap:
                data["initial_right_arm_positions"] = iap["right_arm"]

        # Filter out invalid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        return cls(**filtered_data)

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to a YAML file"""
        path = Path(yaml_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dictionary
        data = {
            "id": self.id,
            "arm_mode": self.arm_mode,
            "use_ee_pose": self.use_ee_pose,
            "arm_dof": self.arm_dof,
            "hand_dof": self.hand_dof,
            "control_hz": self.control_hz,
            "max_time_diff": self.max_time_diff,
            # OpenPI
            "server_host": self.server_host,
            "server_port": self.server_port,
            "action_horizon": self.action_horizon,
            "prompt": self.prompt,
            "broker_mode": self.get_broker_mode(),
            "rtg_trigger_fraction": self.rtg_trigger_fraction,
            "rtg_guidance_steps": self.rtg_guidance_steps,
            "rtg_qp_smoothness_weight": self.rtg_qp_smoothness_weight,
            "rtg_qp_anchor_weight": self.rtg_qp_anchor_weight,
            "rtg_qp_old_weight": self.rtg_qp_old_weight,
            "rtg_qp_new_weight": self.rtg_qp_new_weight,
            "rtg_qp_velocity_scale": self.rtg_qp_velocity_scale,
            "rtg_qp_min_step": self.rtg_qp_min_step,
            # Topics
            "joint_state_topics": self.get_joint_state_topics(),
            "camera_topics": self.get_camera_topics(),
            "action_cmd_topics": self.get_action_cmd_topics(),
            # Timeouts
            "data_timeout_s": self.data_timeout_s,
            "sync_slop": self.sync_slop,
            "qos_reliability": self.qos_reliability,
        }

        # Add initial positions
        if self.initial_left_arm_positions or self.initial_right_arm_positions:
            data["initial_arm_positions"] = {}
            if self.initial_left_arm_positions:
                data["initial_arm_positions"]["left_arm"] = self.initial_left_arm_positions
            if self.initial_right_arm_positions:
                data["initial_arm_positions"]["right_arm"] = self.initial_right_arm_positions

        with path.open("w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
