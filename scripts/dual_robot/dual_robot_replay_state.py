#!/usr/bin/env python3
"""
Replay script for dual VR recorded data with Franka FER + XHand.

This script loads a recorded dataset and replays the actions on the robot,
allowing you to see the recorded manipulation behaviors.
"""

import os
import logging
import time
import sys
import argparse
from pathlib import Path
import torch
import numpy as np
import pandas as pd

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.franka_fer_xhand.franka_fer_xhand import FrankaFERXHand
from lerobot.robots.franka_fer_xhand.franka_fer_xhand_config import FrankaFERXHandConfig
from lerobot.robots.franka_fer.franka_fer_config import FrankaFERConfig
from lerobot.robots.xhand.xhand_config import XHandConfig
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import _init_rerun
from lerobot.utils.control_utils import init_keyboard_listener

# Default replay parameters
DEFAULT_ROBOT_IP = "172.16.0.1"
DEFAULT_EPISODE = 0
DEFAULT_REPLAY_SPEED = 1.0
DEFAULT_CONFIRM = True
DEFAULT_FPS = 30

# Initialize logging
init_logging()
logger = logging.getLogger(__name__)

def setup_robot(robot_ip: str) -> FrankaFERXHand:
    """Set up the composite robot for replay."""
    # Create arm configuration
    arm_config = FrankaFERConfig(
        server_ip=robot_ip,
        server_port=5000,
        home_position=[0, -0.785, 0, -2.356, 0, 1.571, -0.7],
        max_relative_target=None,
        cameras={}
    )
    
    # Create hand configuration
    hand_config = XHandConfig(
        protocol="RS485",
        serial_port="/dev/ttyUSB0",
        baud_rate=3000000,
        hand_id=0,
        control_frequency=200.0,
        max_torque=250.0,
        cameras={}
    )
    
    # Create composite robot configuration
    robot_config = FrankaFERXHandConfig(
        arm_config=arm_config,
        hand_config=hand_config,
        cameras={},
        synchronize_actions=True,
        action_timeout=0.1,
        check_arm_hand_collision=True,
        emergency_stop_both=True
    )
    
    # Create and connect robot
    robot = FrankaFERXHand(robot_config)
    return robot

def replay_episode(robot: FrankaFERXHand, actions: list, fps: float, speed_multiplier: float = 1.0):
    """Replay a sequence of actions on the robot."""
    dt = 1.0 / fps / speed_multiplier
    
    logger.info(f"Replaying {len(actions)} actions at {fps * speed_multiplier:.1f} FPS")
    log_say(f"Starting replay in 3 seconds...")
    time.sleep(3)
    
    log_say("Replay starting now!")
    
    start_time = time.perf_counter()
    
    for i, action in enumerate(actions):
        loop_start = time.perf_counter()
        
        action_dict = {}
        for i in range(7): 
            action_dict[f"arm_joint_{i}.pos"] = action[i]
        for i in range(12): 
            action_dict[f"hand_joint_{i}.pos"] = action[i+7]
        try:
            # Send action to robot
            performed_action = robot.send_action(action_dict)
            
            # Log progress periodically
            if i % 30 == 0:  # Every ~1 second at 30fps
                elapsed = time.perf_counter() - start_time
                progress = (i / len(actions)) * 100
                logger.info(f"Replay progress: {progress:.1f}% ({i}/{len(actions)}) - {elapsed:.1f}s elapsed")
        
        except Exception as e:
            logger.error(f"Error sending action {i}: {e}")
            logger.warning("Continuing with next action...")
            continue
        
        # Maintain timing
        elapsed = time.perf_counter() - loop_start
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        elif sleep_time < -0.01:  # More than 10ms behind
            logger.warning(f"Replay running behind schedule by {-sleep_time*1000:.1f}ms")
    
    total_time = time.perf_counter() - start_time
    log_say(f"Replay complete! Total time: {total_time:.1f}s")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Replay recorded dual robot dataset")
    parser.add_argument("dataset_path", help="Path to the dataset directory")
    parser.add_argument("--episode", type=int, default=DEFAULT_EPISODE, 
                       help=f"Episode to replay (default: {DEFAULT_EPISODE})")
    parser.add_argument("--speed", type=float, default=DEFAULT_REPLAY_SPEED,
                       help=f"Replay speed multiplier (default: {DEFAULT_REPLAY_SPEED})")
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP,
                       help=f"Robot IP address (default: {DEFAULT_ROBOT_IP})")
    parser.add_argument("--no-confirm", action="store_true",
                       help="Skip confirmation before starting replay")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS,
                       help=f"Fps (default: {DEFAULT_FPS})")
    return parser.parse_args()

def main():
    """Main replay function."""
    args = parse_args()
    
    logger.info("Setting up dual robot replay...")
    logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Episode: {args.episode}")
    logger.info(f"Speed: {args.speed}")
    logger.info(f"Robot IP: {args.robot_ip}")
    
    try:
        # Load dataset
        logger.info(f"Loading dataset from: {args.dataset_path}")
        data_dir = Path(args.dataset_path)

        episode_dir = data_dir / "data/chunk-000/"
        episode_file_names = sorted(os.listdir(episode_dir))

        data = pd.read_parquet(episode_dir / episode_file_names[args.episode])
        # actions = np.stack(data["action"].to_numpy())
        states = np.stack(data["observation.state"].to_numpy())
        actions = np.concatenate([states[:, :7], states[:,30:42]], axis=1)

        logger.info(f"Extracted {len(actions)} actions")
        
        # Set up robot
        logger.info("Setting up robot...")
        robot = setup_robot(args.robot_ip)
        
        # Connect robot
        logger.info("Connecting to robot...")
        robot.connect(calibrate=False)

        success = robot.reset_to_home()
        time.sleep(2.0)
        
        if not robot.is_connected:
            raise ValueError("Failed to connect to robot")
        
        logger.info("Robot connected successfully")
        
        # Initialize visualization
        _init_rerun(session_name="dual_robot_replay")
        
        # Set up keyboard listener for emergency stop
        listener, events = init_keyboard_listener()
        
        # Confirmation before starting
        if not args.no_confirm:
            log_say("Ready to replay. Starting in 5 seconds (Ctrl+C to cancel)...")
            time.sleep(5)
        
        # Replay the episode
        replay_episode(robot, actions, args.fps, args.speed)
        
    except KeyboardInterrupt:
        logger.info("Replay cancelled by user")
    except Exception as e:
        logger.error(f"Error during replay: {e}")
        raise
    finally:
        # Clean up
        logger.info("Cleaning up...")
        
        try:
            if 'robot' in locals() and robot.is_connected:
                robot.disconnect()
            logger.info("Robot disconnected")
        except Exception as e:
            logger.error(f"Error disconnecting robot: {e}")
        
        try:
            if 'listener' in locals():
                listener.stop()
        except Exception as e:
            logger.error(f"Error stopping keyboard listener: {e}")
        
        logger.info("Replay session complete!")

if __name__ == "__main__":
    main()