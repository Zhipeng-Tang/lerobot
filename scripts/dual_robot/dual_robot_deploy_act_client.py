#!/usr/bin/env python3
"""
ACT Policy Client - Robot control with remote inference.
Runs on the robot machine, calls server for policy inference.
"""

import argparse
import sys
from pathlib import Path
import time
import json
import requests
import numpy as np
import rerun as rr

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lerobot.robots.franka_fer_xhand.franka_fer_xhand import FrankaFERXHand
from lerobot.robots.franka_fer_xhand.franka_fer_xhand_config import FrankaFERXHandConfig
from lerobot.robots.franka_fer.franka_fer_config import FrankaFERConfig
from lerobot.robots.xhand.xhand_config import XHandConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.configs import ColorMode

DEFAULT_FPS = 30
DEFAULT_SERVER_PORT = 5001
DEFAULT_IMAGE_WIDTH = 320
DEFAULT_IMAGE_HEIGHT = 240


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy ACT Client")
    parser.add_argument("--server-ip", type=str, required=True,
                        help="Server IP address")
    parser.add_argument("--server-port", type=int, default=DEFAULT_SERVER_PORT,
                        help=f"Server port (default: {DEFAULT_SERVER_PORT})")
    parser.add_argument("--width", type=int, default=DEFAULT_IMAGE_WIDTH,
                        help=f"Image width (default: {DEFAULT_IMAGE_WIDTH})")
    parser.add_argument("--height", type=int, default=DEFAULT_IMAGE_HEIGHT,
                        help=f"Image height (default: {DEFAULT_IMAGE_HEIGHT})")
    return parser.parse_args()


def check_server_health(server_url: str) -> bool:
    """Check if the server is running."""
    try:
        resp = requests.get(f"{server_url}/health", timeout=5)
        return resp.status_code == 200 and resp.json().get("policy_loaded", False)
    except Exception as e:
        print(f"Server health check failed: {e}")
        return False


def call_inference(server_url: str, observation: dict) -> dict:
    """Call the inference endpoint."""
    resp = requests.post(
        f"{server_url}/inference",
        json={"observation": observation},
        timeout=10
    )
    resp.raise_for_status()
    return resp.json()


def prepare_observation(obs: dict, width: int, height: int) -> dict:
    """Prepare observation dict for server."""
    # Extract state (54 dims)
    env_state = []
    
    # Arm joint positions (7)
    for i in range(7):
        env_state.append(obs[f"arm_joint_{i}.pos"])
    
    # Arm joint velocities (7)
    for i in range(7):
        env_state.append(obs[f"arm_joint_{i}.vel"])
    
    # EE pose (16)
    for i in range(16):
        env_state.append(obs[f"arm_ee_pose.{i:02d}"])
    
    # Hand joint positions (12)
    for i in range(12):
        env_state.append(obs[f"hand_joint_{i}.pos"])
    
    # Hand joint torque (12)
    for i in range(12):
        env_state.append(obs[f"hand_joint_{i}.torque"])
    
    # Extract images
    images = {
        "front_left_view": obs.get("front_left_view", np.zeros((height, width, 3), dtype=np.uint8)).tolist(),
        "wrist_view": obs.get("wrist_view", np.zeros((height, width, 3), dtype=np.uint8)).tolist(),
        "head_view": obs.get("head_view", np.zeros((height, width, 3), dtype=np.uint8)).tolist(),
    }
    
    return {
        "state": [env_state],  # Server will handle history
        "images": images
    }


def main():
    print("=== Combo Robot ACT Policy Client ===")
    args = parse_args()
    
    # Tunable parameters
    ACTION_SCALE = 1.0
    SMOOTHING_ALPHA = 1.0
    QUERY_FREQUENCY = 48  # Query new action chunk every N steps
    
    print(f"Settings: ACTION_SCALE={ACTION_SCALE}, SMOOTHING_ALPHA={SMOOTHING_ALPHA}, QUERY_FREQUENCY={QUERY_FREQUENCY}")
    
    server_url = f"http://{args.server_ip}:{args.server_port}"
    print(f"Connecting to server: {server_url}")
    
    # Check server health
    if not check_server_health(server_url):
        print("ERROR: Server is not available!")
        return 1
    print("Server is ready!")
    
    # Create robot configuration
    arm_config = FrankaFERConfig(
        server_ip="172.16.0.1",
        server_port=5000,
        home_position=[0, -0.785, 0, -2.356, 0, 1.571, -0.7],
        cameras={}
    )
    
    hand_config = XHandConfig(
        protocol="RS485",
        serial_port="/dev/ttyUSB0",
        baud_rate=3000000,
        hand_id=0,
        control_frequency=30.0,
        max_torque=250.0,
        cameras={}
    )
    
    cameras = {
        "front_left_view": RealSenseCameraConfig(
            serial_number_or_name="243722073109",
            fps=60,
            width=args.width,
            height=args.height,
            color_mode=ColorMode.RGB,
        ),
        "head_view": RealSenseCameraConfig(
            serial_number_or_name="243722074964",
            fps=60,
            width=args.width,
            height=args.height,
            color_mode=ColorMode.RGB,
        ),
        "wrist_view": RealSenseCameraConfig(
            serial_number_or_name="243722075298",
            fps=60,
            width=args.width,
            height=args.height,
            color_mode=ColorMode.RGB,
        )
    }
    
    robot_config = FrankaFERXHandConfig(
        arm_config=arm_config,
        hand_config=hand_config,
        cameras=cameras,
        synchronize_actions=True,
        action_timeout=0.2,
        check_arm_hand_collision=True,
        emergency_stop_both=True
    )
    
    # Create robot
    robot = FrankaFERXHand(robot_config)
    
    # Connect robot
    print("Connecting to robot...")
    robot.connect(calibrate=False)
    
    if not robot.is_connected:
        print("Failed to connect to robot!")
        return 1
    
    print("Robot connected successfully")
    
    # Initialize Rerun
    rr.init("combo_robot_deployment", spawn=True)
    
    # Home robot
    print("Homing robot...")
    robot.reset_to_home()
    time.sleep(2)
    
    # Main control loop
    print("\n=== Starting control loop ===")
    print("Press Ctrl+C to stop")
    
    fps = 30
    dt = 1.0 / fps
    frame_idx = 0
    
    # Action smoothing
    action_smoothing_alpha = SMOOTHING_ALPHA
    prev_action = None
    
    # Action chunk from server
    action_chunk = None
    chunk_idx = 0
    n_action_steps = 8  # Default, will be updated from server response
    chunk_size = 8
    
    # State history for server
    obs_history_states = []
    obs_history_images = {
        "front_left_view": [],
        "wrist_view": [],
        "head_view": []
    }
    
    # For query frequency control
    query_count = 0
    
    try:
        while True:
            start_time = time.perf_counter()
            
            # Set rerun time
            rr.set_time_sequence("frame", frame_idx)
            rr.set_time_seconds("time", time.time())
            
            # Get observation
            obs = robot.get_observation()
            
            # Prepare observation for server
            env_state = []
            
            # Arm joint positions (7)
            for i in range(7):
                env_state.append(obs[f"arm_joint_{i}.pos"])
            
            # Arm joint velocities (7)
            for i in range(7):
                env_state.append(obs[f"arm_joint_{i}.vel"])
            
            # EE pose (16)
            for i in range(16):
                env_state.append(obs[f"arm_ee_pose.{i:02d}"])
            
            # Hand joint positions (12)
            for i in range(12):
                env_state.append(obs[f"hand_joint_{i}.pos"])
            
            # Hand joint torque (12)
            for i in range(12):
                env_state.append(obs[f"hand_joint_{i}.torque"])
            
            # Extract images
            front_left_view_image = obs.get("front_left_view", np.zeros((args.height, args.width, 3), dtype=np.uint8))
            wrist_view_image = obs.get("wrist_view", np.zeros((args.height, args.width, 3), dtype=np.uint8))
            head_view_image = obs.get("head_view", np.zeros((args.height, args.width, 3), dtype=np.uint8))
            
            # Update history
            obs_history_states.append(env_state)
            obs_history_images["front_left_view"].append(front_left_view_image.tolist())
            obs_history_images["wrist_view"].append(wrist_view_image.tolist())
            obs_history_images["head_view"].append(head_view_image.tolist())
            
            # Keep only last 1 observations (n_obs_steps)
            n_obs_steps = 1
            if len(obs_history_states) > n_obs_steps:
                obs_history_states = obs_history_states[-n_obs_steps:]
                for key in obs_history_images:
                    obs_history_images[key] = obs_history_images[key][-n_obs_steps:]
            
            # Generate new action chunk when needed
            if action_chunk is None or chunk_idx >= n_action_steps or query_count % QUERY_FREQUENCY == 0:
                # Check if we have enough history
                if len(obs_history_states) == n_obs_steps:
                    # Prepare observation for server
                    observation = {
                        "state": obs_history_states,
                        "images": {
                            "front_left_view": obs_history_images["front_left_view"],
                            "wrist_view": obs_history_images["wrist_view"],
                            "head_view": obs_history_images["head_view"]
                        }
                    }
                    
                    # Call server for inference
                    try:
                        start_time = time.perf_counter()
                        inference_result = call_inference(server_url, observation)
                        action_chunk = np.array(inference_result["actions"])
                        n_action_steps = inference_result.get("n_action_steps", 8)
                        chunk_size = inference_result.get("chunk_size", 8)
                        inference_time = inference_result.get("inference_time_ms", 0)
                        print(f"Got new action chunk: {action_chunk.shape}, inference time: {inference_time:.1f}ms, inference and network time: {(time.perf_counter()-start_time)*1000:.1f}ms", flush=True)
                    except Exception as e:
                        print(f"Inference failed: {e}", flush=True)
                        action_chunk = np.array(
                            [
                                [obs[f"arm_joint_{i}.pos"] for i in range(7)]
                                + [obs[f"hand_joint_{i}.pos"] for i in range(12)] 
                                for _ in range(n_action_steps)
                            ]
                        )
                    
                    chunk_idx = 0
                else:
                    # Not enough history yet
                    action_chunk = np.array(
                        [
                            [obs[f"arm_joint_{i}.pos"] for i in range(7)]
                            + [obs[f"hand_joint_{i}.pos"] for i in range(12)] 
                            for _ in range(n_action_steps)
                        ]
                    )
                    chunk_idx = 0
            
            query_count += 1
            
            # Use current action from chunk
            action = action_chunk[chunk_idx]
            
            # Debug
            if frame_idx % 10 == 0:
                print(f"Using action {chunk_idx+1}/{n_action_steps} from chunk")
                print(f"Action range: [{action.min():.3f}, {action.max():.3f}]")
            
            chunk_idx += 1
            
            # Store raw action for debugging
            raw_action = action.copy()
            
            # Apply exponential smoothing
            if prev_action is not None:
                action = action_smoothing_alpha * action + (1 - action_smoothing_alpha) * prev_action
            prev_action = action.copy()
            
            # Apply action scaling for safety
            action = action * ACTION_SCALE
            
            # Split into arm and hand actions
            action_dict = {}
            
            # Arm actions (first 7)
            for i in range(7):
                action_dict[f"arm_joint_{i}.pos"] = float(action[i])
            
            # Hand actions (next 12)
            for i in range(12):
                action_dict[f"hand_joint_{i}.pos"] = float(action[7 + i])
            
            # Debug hand actions
            if frame_idx % 30 == 0:
                raw_hand = [raw_action[7+i] for i in range(12)]
                smooth_hand = [action[7+i] for i in range(12)]
                print(f"RAW hand: mean={np.mean(raw_hand):.3f}, range=[{min(raw_hand):.3f}, {max(raw_hand):.3f}]")
                print(f"SMOOTH hand: mean={np.mean(smooth_hand):.3f}")
            
            # Send action to robot
            robot.send_action(action_dict)
            
            # Log to Rerun
            for i in range(7):
                rr.log(f"robot/arm/joint_{i}/position", rr.Scalar(obs[f"arm_joint_{i}.pos"]))
                rr.log(f"robot/arm/joint_{i}/velocity", rr.Scalar(obs[f"arm_joint_{i}.vel"]))
                rr.log(f"robot/arm/joint_{i}/action", rr.Scalar(action_dict[f"arm_joint_{i}.pos"]))
            
            for i in range(12):
                rr.log(f"robot/hand/joint_{i}/position", rr.Scalar(obs[f"hand_joint_{i}.pos"]))
                rr.log(f"robot/hand/joint_{i}/action", rr.Scalar(action_dict[f"hand_joint_{i}.pos"]))
            
            # Log camera images
            if "front_left_view" in obs:
                rr.log("cameras/front_left_view", rr.Image(obs["front_left_view"]))
            if "wrist_view" in obs:
                rr.log("cameras/wrist_view", rr.Image(obs["wrist_view"]))
            if "head_view" in obs:
                rr.log("cameras/head_view", rr.Image(obs["head_view"]))
            
            # Log end-effector pose
            ee_pose = np.array([obs[f"arm_ee_pose.{i:02d}"] for i in range(16)]).reshape(4, 4)
            rr.log("robot/arm/ee_pose", rr.Transform3D(mat3x3=ee_pose[:3, :3], translation=ee_pose[:3, 3]))
            
            # Maintain loop timing
            elapsed = time.perf_counter() - start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)
            
            if frame_idx % fps == 0:
                print(f"Control loop running... (frame {frame_idx}, loop time: {elapsed*1000:.1f}ms)")
            
            frame_idx += 1
    
    except KeyboardInterrupt:
        print("\n=== Stopping control loop ===")
    except Exception as e:
        print(f"Error in control loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Disconnecting robot...")
        if robot.is_connected:
            robot.stop()
            robot.disconnect()
        print("Done!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
