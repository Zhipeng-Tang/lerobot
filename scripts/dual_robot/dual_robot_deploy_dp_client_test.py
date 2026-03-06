#!/usr/bin/env python3
"""
Test script for diffusion policy inference server.
Sends fake observations and checks if the server returns valid actions.
"""

import argparse
import sys
from pathlib import Path
import time
import json
import requests
import numpy as np

DEFAULT_SERVER_PORT = 5000
DEFAULT_IMAGE_WIDTH = 320
DEFAULT_IMAGE_HEIGHT = 240
DEFAULT_N_OBS_STEPS = 2
DEFAULT_ACTION_DIM = 19
DEFAULT_N_ACTION_STEPS = 8


def parse_args():
    parser = argparse.ArgumentParser(description="Test DP Inference Server")
    parser.add_argument("--server-ip", type=str, default="localhost",
                        help="Server IP address")
    parser.add_argument("--server-port", type=int, default=DEFAULT_SERVER_PORT,
                        help=f"Server port (default: {DEFAULT_SERVER_PORT})")
    parser.add_argument("--width", type=int, default=DEFAULT_IMAGE_WIDTH,
                        help=f"Image width (default: {DEFAULT_IMAGE_WIDTH})")
    parser.add_argument("--height", type=int, default=DEFAULT_IMAGE_HEIGHT,
                        help=f"Image height (default: {DEFAULT_IMAGE_HEIGHT})")
    parser.add_argument("--n-obs-steps", type=int, default=DEFAULT_N_OBS_STEPS,
                        help=f"Number of observation steps (default: {DEFAULT_N_OBS_STEPS})")
    return parser.parse_args()


def create_fake_observation(width: int, height: int, n_obs_steps: int, action_dim: int = 19):
    """Create fake observation data."""
    # State: 54 dims (arm 7 pos + 7 vel + 16 ee_pose + 12 hand pos + 12 hand torque)
    state_dim = 54
    
    # Create state history
    state_history = []
    for step in range(n_obs_steps):
        # Use sinusoidal patterns to simulate robot state
        state = []
        # Arm joint positions (7) - typical range [-pi, pi]
        for i in range(7):
            state.append(np.sin(step * 0.5 + i) * 0.5)
        # Arm joint velocities (7) - typical range [-2, 2]
        for i in range(7):
            state.append(np.cos(step * 0.3 + i) * 0.3)
        # EE pose (16) - 4x4 matrix flattened (position + quaternion)
        for i in range(16):
            state.append((np.sin(step * 0.2 + i * 0.1) * 0.1) if i < 3 else np.cos(step * 0.1 + i * 0.05))
        # Hand joint positions (12) - typical range [0, 1]
        for i in range(12):
            state.append(np.sin(step * 0.4 + i * 0.3) * 0.5 + 0.5)
        # Hand joint torque (12) - typical range [-1, 1]
        for i in range(12):
            state.append(np.cos(step * 0.4 + i * 0.3) * 0.2)
        
        assert len(state) == state_dim, f"State dim mismatch: {len(state)} != {state_dim}"
        state_history.append(state)
    
    # Create fake images (RGB)
    def create_fake_image(h, w):
        # Create a simple gradient image with some pattern
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # Add some gradient
        for i in range(h):
            img[i, :, 0] = int((i / h) * 200)  # Red gradient
            img[i, :, 1] = int((i / h) * 150 + 50)  # Green gradient
        # Add a moving square pattern
        square_size = 30
        offset = int(time.time() * 10) % w
        img[10:10+square_size, offset:offset+square_size, 2] = 255  # Blue square
        return img.tolist()
    
    images = {
        "tpv": [create_fake_image(height, width) for _ in range(n_obs_steps)],
        "wrist": [create_fake_image(height, width) for _ in range(n_obs_steps)],
        "overhead": [create_fake_image(height, width) for _ in range(n_obs_steps)],
    }
    
    return {
        "state": state_history,
        "images": images
    }


def check_health(server_url: str) -> bool:
    """Check server health."""
    try:
        resp = requests.get(f"{server_url}/health", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            print(f"Server status: {data}")
            return data.get("policy_loaded", False)
        else:
            print(f"Health check failed: {resp.status_code}")
            return False
    except Exception as e:
        print(f"Failed to connect to server: {e}")
        return False


def test_inference(server_url: str, observation: dict, n_action_steps: int = 8):
    """Test inference endpoint."""
    print("\n--- Testing inference ---")
    print(f"Sending observation with {len(observation['state'])} state steps")
    for key, images in observation["images"].items():
        print(f"  {key}: {len(images)} images, size {len(images[0])}x{len(images[0][0])}")
    
    start_time = time.perf_counter()
    
    try:
        resp = requests.post(
            f"{server_url}/inference",
            json={"observation": observation},
            timeout=30
        )
        resp.raise_for_status()
        result = resp.json()
        
        elapsed = time.perf_counter() - start_time
        
        print(f"\nInference successful!")
        print(f"  Total time: {elapsed*1000:.1f}ms")
        print(f"  Server inference time: {result.get('inference_time_ms', 0):.1f}ms")
        print(f"  Action shape: {result.get('action_dim', 'N/A')}D")
        print(f"  Number of actions: {result.get('n_action_steps', 'N/A')}")
        
        actions = result.get("actions", [])
        if actions:
            actions = np.array(actions)
            print(f"\nAction statistics:")
            print(f"  Min: {actions.min():.4f}")
            print(f"  Max: {actions.max():.4f}")
            print(f"  Mean: {actions.mean():.4f}")
            print(f"  Std: {actions.std():.4f}")
            
            # Check if actions are in reasonable range
            if actions.max() > 100 or actions.min() < -100:
                print("\n WARNING: Actions seem too large! Possible normalization issue.")
            else:
                print("\n Actions appear to be in reasonable range.")
        
        return True
    
    except requests.exceptions.RequestException as e:
        print(f"Inference failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        return False


def test_reset(server_url: str):
    """Test reset endpoint."""
    print("\n--- Testing reset ---")
    try:
        resp = requests.post(f"{server_url}/reset", timeout=5)
        resp.raise_for_status()
        print(f"Reset successful: {resp.json()}")
        return True
    except Exception as e:
        print(f"Reset failed: {e}")
        return False


def main():
    args = parse_args()
    
    server_url = f"http://{args.server_ip}:{args.server_port}"
    print(f"=== Testing Diffusion Policy Server ===")
    print(f"Server: {server_url}")
    print(f"Image size: {args.width}x{args.height}")
    print(f"Observation steps: {args.n_obs_steps}")
    
    # Check health
    print("\n--- Checking health ---")
    if not check_health(server_url):
        print("ERROR: Server is not ready!")
        print("Make sure the server is running with a loaded policy.")
        return 1
    
    # Test reset
    test_reset(server_url)
    
    # Create fake observation
    observation = create_fake_observation(
        width=args.width,
        height=args.height,
        n_obs_steps=args.n_obs_steps
    )
    
    # Test inference
    if not test_inference(server_url, observation):
        return 1
    
    # Test multiple inferences to check consistency
    print("\n--- Testing multiple inferences ---")
    num_tests = 3
    for i in range(num_tests):
        print(f"\nTest {i+1}/{num_tests}")
        observation = create_fake_observation(
            width=args.width,
            height=args.height,
            n_obs_steps=args.n_obs_steps
        )
        test_inference(server_url, observation)
    
    print("\n=== All tests passed! ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
