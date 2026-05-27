#!/usr/bin/env python3
"""
Utility functions module

Provides pose conversion, math utilities, and other shared functionality.
"""

from __future__ import annotations

import math

import numpy as np


def quaternion_to_euler(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float]:
    """
    Convert a quaternion to Euler angles (roll, pitch, yaw)

    Args:
        qx, qy, qz, qw: quaternion components

    Returns:
        (roll, pitch, yaw) Euler angles (radians)
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """
    Convert Euler angles to a quaternion

    Args:
        roll: rotation about the X axis (radians)
        pitch: rotation about the Y axis (radians)
        yaw: rotation about the Z axis (radians)

    Returns:
        (qx, qy, qz, qw) quaternion components
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return qx, qy, qz, qw


def normalize_angle(angle: float) -> float:
    """
    Normalize an angle to the range [-pi, pi]

    Args:
        angle: input angle (radians)

    Returns:
        Normalized angle
    """
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


# =============================================================================
# Degree/radian conversion functions
# =============================================================================

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


def deg2rad(degrees: float) -> float:
    """Degrees to radians (single value)"""
    return degrees * DEG2RAD


def rad2deg(radians: float) -> float:
    """Radians to degrees (single value)"""
    return radians * RAD2DEG


def deg2rad_array(degrees: list[float]) -> list[float]:
    """Degrees to radians (array)"""
    return [d * DEG2RAD for d in degrees]


def rad2deg_array(radians: list[float]) -> list[float]:
    """Radians to degrees (array)"""
    return [r * RAD2DEG for r in radians]


def convert_joints_to_model(
    joints: list[float],
    hw_unit: str,
) -> list[float]:
    """
    Convert joint data from hardware units to model units (radians)

    Used for unit conversion when reading observations.

    Args:
        joints: list of joint data
        hw_unit: hardware unit ("degree" or "radian")

    Returns:
        Converted joint data (radians)
    """
    if hw_unit == "degree":
        return deg2rad_array(joints)
    return joints  # already in radians


def convert_joints_to_hardware(
    joints: list[float],
    hw_unit: str,
) -> list[float]:
    """
    Convert joint data from model units (radians) to hardware units

    Used for unit conversion when publishing actions.

    Args:
        joints: list of joint data (radians)
        hw_unit: hardware unit ("degree" or "radian")

    Returns:
        Converted joint data (in hardware units)
    """
    if hw_unit == "degree":
        return rad2deg_array(joints)
    return joints  # keep radians


def interpolate_joint_positions(start: list[float], end: list[float], steps: int) -> list[list[float]]:
    """
    Linearly interpolate between two joint positions

    Args:
        start: starting joint positions
        end: target joint positions
        steps: number of interpolation steps

    Returns:
        List of interpolated joint positions
    """
    if len(start) != len(end):
        raise ValueError("Starting and target joint position dimensions do not match")

    start_arr = np.array(start)
    end_arr = np.array(end)

    positions = []
    for i in range(steps + 1):
        t = i / steps
        pos = start_arr + t * (end_arr - start_arr)
        positions.append(pos.tolist())

    return positions


def pack_action_array(
    left_arm: list[float] | None = None,
    left_hand: list[float] | None = None,
    right_arm: list[float] | None = None,
    right_hand: list[float] | None = None,
    arm_mode: str = "single_right",
) -> np.ndarray:
    """
    Pack separated arm/hand actions into a unified action array

    Args:
        left_arm: left arm joints (7 DOF)
        left_hand: left hand joints (20 DOF)
        right_arm: right arm joints (7 DOF)
        right_hand: right hand joints (20 DOF)
        arm_mode: arm mode ("single_left", "single_right", "dual", "both")

    Returns:
        Packed action array
            - Single arm: 27D (arm + hand)
            - Dual arm: 54D (left_arm + left_hand + right_arm + right_hand)
    """
    action = []

    if arm_mode in ("single_left", "dual", "both"):
        if left_arm is not None:
            action.extend(left_arm)
        if left_hand is not None:
            action.extend(left_hand)

    if arm_mode in ("single_right", "dual", "both"):
        if right_arm is not None:
            action.extend(right_arm)
        if right_hand is not None:
            action.extend(right_hand)

    return np.array(action, dtype=np.float32)


def unpack_action_array(
    action: np.ndarray,
    arm_mode: str = "single_right",
    arm_dof: int = 7,
    hand_dof: int = 20,
) -> dict:
    """
    Unpack a unified action array into separated arm/hand actions

    Args:
        action: action array
        arm_mode: arm mode
        arm_dof: arm degrees of freedom
        hand_dof: hand degrees of freedom

    Returns:
        Unpacked action dictionary {"left_arm", "left_hand", "right_arm", "right_hand"}
    """
    result = {}
    idx = 0

    arm_dof + hand_dof

    if arm_mode == "single_left":
        result["left_arm"] = action[idx : idx + arm_dof]
        idx += arm_dof
        result["left_hand"] = action[idx : idx + hand_dof]
    elif arm_mode == "single_right":
        result["right_arm"] = action[idx : idx + arm_dof]
        idx += arm_dof
        result["right_hand"] = action[idx : idx + hand_dof]
    elif arm_mode in ("dual", "both"):
        # Dual arm: left_arm + left_hand + right_arm + right_hand
        result["left_arm"] = action[idx : idx + arm_dof]
        idx += arm_dof
        result["left_hand"] = action[idx : idx + hand_dof]
        idx += hand_dof
        result["right_arm"] = action[idx : idx + arm_dof]
        idx += arm_dof
        result["right_hand"] = action[idx : idx + hand_dof]

    return result


def build_observation_state(
    synced_data,
    arm_mode: str = "single_right",
    use_ee_pose: bool = False,  # noqa: FBT001, FBT002
) -> np.ndarray:
    """
    Build a state vector from synchronized data

    Args:
        synced_data: SyncedData object
        arm_mode: arm mode
        use_ee_pose: whether to include the end-effector pose

    Returns:
        State vector (numpy array)

    State order:
        - Single arm: arm(7) + [ee(6)] + hand(20)
        - Dual arm: left_arm(7) + [left_ee(6)] + left_hand(20) + right_arm(7) + [right_ee(6)] + right_hand(20)
    """
    state_parts = []

    if arm_mode in ("single_left", "dual", "both"):
        if "left_arm" in synced_data.joint_states:
            state_parts.extend(synced_data.joint_states["left_arm"])
        if use_ee_pose and "left_ee" in synced_data.joint_states:
            state_parts.extend(synced_data.joint_states["left_ee"])
        if "left_hand" in synced_data.joint_states:
            state_parts.extend(synced_data.joint_states["left_hand"])

    if arm_mode in ("single_right", "dual", "both"):
        if "right_arm" in synced_data.joint_states:
            state_parts.extend(synced_data.joint_states["right_arm"])
        if use_ee_pose and "right_ee" in synced_data.joint_states:
            state_parts.extend(synced_data.joint_states["right_ee"])
        if "right_hand" in synced_data.joint_states:
            state_parts.extend(synced_data.joint_states["right_hand"])

    return np.array(state_parts, dtype=np.float32)
