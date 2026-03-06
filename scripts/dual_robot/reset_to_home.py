#!/usr/bin/env python3
"""
Test VR control of Franka arm using real robot and VR device via teleoperator class
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.robots.franka_fer.franka_fer import FrankaFER
from lerobot.robots.franka_fer.franka_fer_config import FrankaFERConfig
from lerobot.teleoperators.franka_fer_vr.franka_fer_vr_teleoperator import FrankaFERVRTeleoperator
from lerobot.teleoperators.franka_fer_vr.config_franka_fer_vr import FrankaFERVRTeleoperatorConfig

def test_vr_arm_control():
    print("Testing VR control of Franka arm using FrankaFERVRTeleoperator...")
    
    # Initialize Franka robot with proper config
    print("Connecting to Franka robot...")
    robot_config = FrankaFERConfig(
        home_position = [0, -0.785, 0, -2.356, 0, 1.571, -0.7], 
        # home_position = [0, -0.785, 0, -2.356, 1.0, 1.4, -0.7], 
    )
    robot = FrankaFER(robot_config)
    robot.connect()
    print("Connected to Franka robot")
    
    # Home robot to neutral position
    print("Homing robot to neutral position...")
    home_action = {f"joint_{i}.pos": robot_config.home_position[i] for i in range(7)}
    # time.sleep(5.0)
    robot.reset_to_home()
    time.sleep(2.0)  # Wait for robot to reach home position
    print("Robot home position")
    

if __name__ == "__main__":
    success = test_vr_arm_control()
    sys.exit(0 if success else 1)