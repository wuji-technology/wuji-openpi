#!/usr/bin/env python3
"""
OpenPI deployment entry point

Uses the openpi_client framework to run the Wuji robot control system.
Connects to the remote policy server via WebSocket for inference.

Architecture:
    Runtime (coordinator)
      |-> Environment (Wuji ROS2 environment)
      |-> Agent (policy agent)
      |     |-> ActionChunkBroker (action chunk management)
      |           |-> WebsocketClientPolicy (policy server client)
      |-> Subscribers (observers)

Usage:
    # Use a config file
    python -m wuji.deploy.main -c config/deploy.yaml

    # Use command-line arguments
    python -m wuji.deploy.main --host 192.168.1.100 --port 12346 --side both
"""

from __future__ import annotations

import argparse
import logging
import sys

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import rclpy
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import JointState

from wuji.core import DeployConfig

from .ros_env import OpenPIRosEnvironment

logger = logging.getLogger(__name__)


class WujiDeployNode(Node):
    """Wuji deployment ROS2 node"""

    def __init__(self, name: str = "wuji_openpi_node"):
        super().__init__(name)
        self.get_logger().info(f"Node {name} created")


def parse_args(argv: list[str] | None = None) -> DeployConfig:
    """Parse command-line arguments and return the configuration"""
    parser = argparse.ArgumentParser(description="Wuji robot deployment using OpenPI framework")
    parser.add_argument("-c", "--config", default=None, help="config file path")
    parser.add_argument("--host", default=None, help="policy server address")
    parser.add_argument("--port", type=int, default=None, help="policy server port")
    parser.add_argument("--prompt", default=None, help="task language instruction")
    parser.add_argument("--side", choices=["left", "right", "both"], default=None, help="which side to control")
    parser.add_argument("--arm-mode", choices=["single_left", "single_right", "dual"], default=None)
    parser.add_argument("--control-hz", type=float, default=None, help="control frequency (Hz)")
    parser.add_argument("--action-horizon", type=int, default=None, help="action chunk length")
    parser.add_argument("--num-episodes", type=int, default=None, help="number of episodes")
    parser.add_argument("--max-episode-steps", type=int, default=None, help="maximum steps per episode")
    parser.add_argument(
        "--broker-mode",
        choices=["serial", "rtg", "rtg_qp"],
        default=None,
        help="action broker mode",
    )

    parsed = parser.parse_args(argv)

    # Load from config file
    config = DeployConfig.from_yaml(parsed.config) if parsed.config else DeployConfig()

    # Command-line argument overrides
    if parsed.host:
        config.server_host = parsed.host
    if parsed.port:
        config.server_port = parsed.port
    if parsed.prompt:
        config.prompt = parsed.prompt
    if parsed.side:
        # Convert side -> arm_mode
        if parsed.side == "left":
            config.arm_mode = "single_left"
        elif parsed.side == "right":
            config.arm_mode = "single_right"
        else:
            config.arm_mode = "dual"
    if parsed.arm_mode:
        config.arm_mode = parsed.arm_mode
    if parsed.control_hz:
        config.control_hz = parsed.control_hz
    if parsed.action_horizon:
        config.action_horizon = parsed.action_horizon
    if parsed.num_episodes:
        config.num_episodes = parsed.num_episodes
    if parsed.max_episode_steps:
        config.max_episode_steps = parsed.max_episode_steps
    if parsed.broker_mode:
        config.broker_mode = parsed.broker_mode

    return config


def move_to_initial_positions(
    node: Node,
    environment: OpenPIRosEnvironment,
    config: DeployConfig,
) -> None:
    """Move the arms to their initial positions"""
    import time

    logger.info("Moving to initial positions...")

    arms_to_move = []
    if config.use_right_arm() and config.initial_right_arm_positions:
        arms_to_move.append(("right_arm", config.initial_right_arm_positions))
    if config.use_left_arm() and config.initial_left_arm_positions:
        arms_to_move.append(("left_arm", config.initial_left_arm_positions))

    if not arms_to_move:
        logger.info("No initial positions configured, skipping")
        return

    stamp = node.get_clock().now().to_msg()
    interface = environment._ros_interface  # noqa: SLF001

    for arm_key, positions in arms_to_move:
        if arm_key not in interface.action_publishers:
            logger.warning(f"Publisher for {arm_key} not found, skipping")
            continue

        msg = JointState()
        msg.header.stamp = stamp
        msg.header.frame_id = f"{arm_key}_init"

        if arm_key == "right_arm":
            msg.name = [f"right_joint_{i+1}" for i in range(7)]
        else:
            msg.name = [f"left_joint_{i+1}" for i in range(7)]

        msg.position = list(positions)
        interface.action_publishers[arm_key].publish(msg)
        logger.info(f"  {arm_key}: {[f'{p:.4f}' for p in positions]}")

    logger.info("Waiting for the arms to reach their initial positions (3s)...")
    time.sleep(3.0)
    logger.info("Initial position setup complete")


def run_openpi_deploy(config: DeployConfig) -> None:
    """
    Run the Wuji robot control system using the OpenPI framework

    Args:
        config: deployment configuration
    """
    # 1. Create the ROS2 node
    node = WujiDeployNode("wuji_openpi_node")

    # 2. Create the WebSocket client and connect to the policy server
    logger.info(f"Connecting to policy server: {config.server_host}:{config.server_port}")
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=config.server_host,
        port=config.server_port,
    )
    metadata = ws_client_policy.get_server_metadata()
    logger.info(f"Server metadata: {metadata}")

    # 3. Create the environment
    environment = OpenPIRosEnvironment(node=node, config=config)

    # 4. Build the hierarchical control system
    broker_mode = config.get_broker_mode()

    if broker_mode == "rtg":
        from openpi_client.rtg_action_broker import RTGActionBroker

        broker = RTGActionBroker(
            policy=ws_client_policy,
            action_horizon=config.action_horizon,
            control_hz=config.control_hz,
            trigger_fraction=config.rtg_trigger_fraction,
            guidance_steps=config.rtg_guidance_steps,
        )
        logger.info("Using RTGActionBroker (Real-Time Trajectory Generation smoothing)")
    elif broker_mode == "rtg_qp":
        from openpi_client.rtg_action_broker import QPRTGActionBroker

        broker = QPRTGActionBroker(
            policy=ws_client_policy,
            action_horizon=config.action_horizon,
            control_hz=config.control_hz,
            trigger_fraction=config.rtg_trigger_fraction,
            guidance_steps=config.rtg_guidance_steps,
            smoothness_weight=config.rtg_qp_smoothness_weight,
            anchor_weight=config.rtg_qp_anchor_weight,
            old_weight=config.rtg_qp_old_weight,
            new_weight=config.rtg_qp_new_weight,
            velocity_scale=config.rtg_qp_velocity_scale,
            min_step=config.rtg_qp_min_step,
        )
        logger.info("Using QPRTGActionBroker (QP-based Real-Time Trajectory Generation)")
    else:
        broker = action_chunk_broker.ActionChunkBroker(
            policy=ws_client_policy,
            action_horizon=config.action_horizon,
        )
        logger.info("Using ActionChunkBroker (serial chunk execution)")

    runtime = _runtime.Runtime(
        environment=environment,
        agent=_policy_agent.PolicyAgent(policy=broker),
        subscribers=[],
        max_hz=config.control_hz,
        num_episodes=config.num_episodes,
        max_episode_steps=config.max_episode_steps,
    )

    # 5. Log the configuration
    logger.info("=" * 60)
    logger.info("Wuji robot control system configuration (OpenPI):")
    logger.info(f"  Policy server: {config.server_host}:{config.server_port}")
    logger.info(f"  Control mode: {config.arm_mode}")
    logger.info(f"  Control frequency: {config.control_hz}Hz")
    logger.info(f"  Action chunk length: {config.action_horizon} steps")
    logger.info(f"  Inference frequency: {config.control_hz / config.action_horizon:.2f}Hz")
    logger.info(f"  Broker mode: {broker_mode}")
    if broker_mode == "rtg":
        logger.info(
            "  RTG params: trigger_fraction=%.2f, guidance_steps=%d",
            config.rtg_trigger_fraction,
            config.rtg_guidance_steps,
        )
    elif broker_mode == "rtg_qp":
        logger.info(
            "  QP-RTG params: trigger_fraction=%.2f, guidance_steps=%d, smooth=%.2f, anchor=%.2f, old=%.2f, new=%.2f, vel_scale=%.2f",
            config.rtg_trigger_fraction,
            config.rtg_guidance_steps,
            config.rtg_qp_smoothness_weight,
            config.rtg_qp_anchor_weight,
            config.rtg_qp_old_weight,
            config.rtg_qp_new_weight,
            config.rtg_qp_velocity_scale,
        )
    logger.info(f"  Number of episodes: {config.num_episodes}")
    logger.info(f"  Maximum steps per episode: {config.max_episode_steps}")
    logger.info(f"  Task instruction: {config.prompt}")
    logger.info("=" * 60)

    # 5.5 Move to initial positions (if configured)
    if config.initial_left_arm_positions or config.initial_right_arm_positions:
        move_to_initial_positions(node, environment, config)

    # 6. Run the control system
    logger.info("Starting run...")
    try:
        runtime.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        node.destroy_node()

    logger.info("Run complete!")


def main(argv: list[str] | None = None) -> None:
    """Main entry point"""
    program_name = sys.argv[0] if sys.argv else "wuji_openpi"
    raw_argv = sys.argv if argv is None else [program_name, *argv]
    cli_argv = remove_ros_args(raw_argv)[1:]
    config = parse_args(cli_argv)

    # Initialize ROS2
    rclpy.init(args=raw_argv)

    try:
        run_openpi_deploy(config)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
