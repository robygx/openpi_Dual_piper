"""Main entry point for running DualPiper robot with remote inference.

This script:
1. Connects to a remote policy server
2. Runs the robot control loop
3. Sends observations and receives actions
"""

import dataclasses
import json
import logging
from typing import Optional

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

from examples.dual_piper import env as _env
from examples.dual_piper import constants as _constants


@dataclasses.dataclass
class Args:
    """Command line arguments for DualPiper robot control."""

    # Server connection
    host: str = "192.168.1.100"
    """IP address of the remote policy server."""
    port: int = 8000
    """Port of the remote policy server."""

    # CAN interfaces for Piper arms
    can_left: str = "can0"
    """CAN interface name for the left arm."""
    can_right: str = "can1"
    """CAN interface name for the right arm."""

    # Policy settings
    action_horizon: int = 10
    """Number of action steps to predict at once (chunk size)."""

    # Episode settings
    num_episodes: int = 1
    """Number of episodes to run."""
    max_episode_steps: int = 1000
    """Maximum steps per episode."""

    # Control settings
    velocity: int = 50
    """Movement speed percentage (0-100)."""
    max_hz: int = 50
    """Maximum control frequency in Hz."""

    # Camera settings
    camera_serials: Optional[str] = None
    """JSON string mapping camera names to serial numbers, e.g., '{\"cam_high\": \"xxx\", \"cam_left_wrist\": \"yyy\", \"cam_right_wrist\": \"zzz\"}'."""

    # Task prompt
    prompt: str = "stack the blocks"
    """Task description prompt for the policy."""


def main(args: Args) -> None:
    """Run the DualPiper robot with remote inference.

    Args:
        args: Command line arguments
    """
    # 1. Connect to remote policy server
    logging.info(f"Connecting to policy server at {args.host}:{args.port}")
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )

    # Get and display server metadata
    metadata = ws_client_policy.get_server_metadata()
    logging.info(f"Connected! Server metadata: {metadata}")

    # 2. Create the robot environment
    logging.info("Initializing DualPiper environment...")
    environment = _env.DualPiperRealEnvironment(
        can_left=args.can_left,
        can_right=args.can_right,
        render_height=_constants.CONSTANTS.IMAGE_HEIGHT,
        render_width=_constants.CONSTANTS.IMAGE_WIDTH,
        velocity=args.velocity,
        camera_serials=json.loads(args.camera_serials) if args.camera_serials else None,
        prompt=args.prompt,
    )

    # 3. Create the policy agent with action chunking
    agent = _policy_agent.PolicyAgent(
        policy=action_chunk_broker.ActionChunkBroker(
            policy=ws_client_policy,
            action_horizon=args.action_horizon,
        )
    )

    # 4. Create and run the runtime
    runtime = _runtime.Runtime(
        environment=environment,
        agent=agent,
        subscribers=[],  # No subscribers for now
        max_hz=args.max_hz,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    logging.info("Starting control loop...")
    runtime.run()

    # Reset to home position after completion
    logging.info("Resetting robot to home position...")
    environment.reset()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    tyro.cli(main)
