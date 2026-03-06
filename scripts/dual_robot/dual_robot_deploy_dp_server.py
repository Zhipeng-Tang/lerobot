#!/usr/bin/env python3
"""
Diffusion Policy Server - Model deployment with API support.
Runs on a separate machine with GPU.
"""

import argparse
import sys
from pathlib import Path
import time
import json
import threading
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from flask import Flask, request, jsonify
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.configs.types import PolicyFeature, FeatureType, NormalizationMode

app = Flask(__name__)

DEFAULT_IMAGE_WIDTH = 320
DEFAULT_IMAGE_HEIGHT = 240

policy = None
config = None
stats = None
n_obs_steps = None
horizon = None
n_action_steps = None
action_dim = None


def load_policy(ckpt_dirs: str, width: int, height: int):
    """Load the diffusion policy from checkpoint."""
    global policy, config, stats, n_obs_steps, horizon, n_action_steps, action_dim
    
    policy_path = Path(ckpt_dirs)
    print(f"Loading policy from {policy_path}")
    
    # Find the latest checkpoint directory
    checkpoint_dirs = sorted([d for d in policy_path.iterdir() if d.is_dir() and d.name.isdigit()])
    if not checkpoint_dirs:
        raise ValueError("No checkpoint directories found!")
    
    latest_checkpoint_dir = checkpoint_dirs[-1]
    print(f"Using checkpoint: {latest_checkpoint_dir}")
    
    # Load config
    config_path = latest_checkpoint_dir / "pretrained_model" / "config.json"
    with open(config_path) as f:
        config_dict = json.load(f)
    
    # Convert input_features and output_features to PolicyFeature objects
    input_features = {}
    for key, value in config_dict.get('input_features', {}).items():
        input_features[key] = PolicyFeature(
            type=FeatureType[value['type']],
            shape=tuple(value['shape'])
        )
    
    output_features = {}
    for key, value in config_dict.get('output_features', {}).items():
        output_features[key] = PolicyFeature(
            type=FeatureType[value['type']],
            shape=tuple(value['shape'])
        )
    
    # Convert normalization_mapping strings to NormalizationMode enums
    normalization_mapping = {}
    for key, value in config_dict.get('normalization_mapping', {}).items():
        normalization_mapping[key] = NormalizationMode[value]
    
    # Create DiffusionConfig
    config = DiffusionConfig(
        n_obs_steps=config_dict.get('n_obs_steps', 2),
        normalization_mapping=normalization_mapping,
        input_features=input_features,
        output_features=output_features,
        device=config_dict.get('device', 'cuda'),
        horizon=config_dict.get('horizon', 16),
        n_action_steps=config_dict.get('n_action_steps', 8),
        vision_backbone=config_dict.get('vision_backbone', 'resnet18'),
        pretrained_backbone_weights=config_dict.get('pretrained_backbone_weights'),
        crop_shape=config_dict.get('crop_shape'),
        crop_is_random=config_dict.get('crop_is_random', True),
        use_group_norm=config_dict.get('use_group_norm', True),
        spatial_softmax_num_keypoints=config_dict.get('spatial_softmax_num_keypoints', 80),
        use_separate_rgb_encoder_per_camera=config_dict.get('use_separate_rgb_encoder_per_camera', False),
        down_dims=tuple(config_dict.get('down_dims', [256, 512])),
        kernel_size=config_dict.get('kernel_size', 3),
        n_groups=config_dict.get('n_groups', 8),
        diffusion_step_embed_dim=config_dict.get('diffusion_step_embed_dim', 128),
        use_film_scale_modulation=config_dict.get('use_film_scale_modulation', True),
        noise_scheduler_type=config_dict.get('noise_scheduler_type', 'DDPM').upper(),
        num_train_timesteps=config_dict.get('num_train_timesteps', 100),
        beta_schedule=config_dict.get('beta_schedule', 'squaredcos_cap_v2'),
        beta_start=config_dict.get('beta_start', 0.0001),
        beta_end=config_dict.get('beta_end', 0.02),
        prediction_type=config_dict.get('prediction_type', 'epsilon'),
        clip_sample=config_dict.get('clip_sample', True),
        clip_sample_range=config_dict.get('clip_sample_range', 1.0),
        num_inference_steps=config_dict.get('num_inference_steps'),
        do_mask_loss_for_padding=config_dict.get('do_mask_loss_for_padding', False)
    )
    
    # Try to load actual stats from dataset if available
    stats_path = policy_path / "stats.json"
    if not stats_path.exists():
        stats_path = policy_path.parent / "stats.json"
    
    if stats_path.exists():
        with open(stats_path) as f:
            raw_stats = json.load(f)
        stats = {}
        for key in raw_stats:
            if key in raw_stats:
                stats[key] = {}
                for stat_type in ["mean", "std", "min", "max"]:
                    if stat_type in raw_stats[key]:
                        stats[key][stat_type] = torch.tensor(raw_stats[key][stat_type])
        print("Loaded dataset statistics from stats.json")
    else:
        print("Warning: No stats.json found, using dummy normalization")
        stats = {
            "observation.state": {
                "min": torch.ones(54) * -1,
                "max": torch.ones(54),
                "mean": torch.zeros(54),
                "std": torch.ones(54)
            },
            "observation.images.tpv": {
                "mean": torch.zeros(3, height, width),
                "std": torch.ones(3, height, width)
            },
            "observation.images.wrist": {
                "mean": torch.zeros(3, height, width),
                "std": torch.ones(3, height, width)
            },
            "observation.images.overhead": {
                "mean": torch.zeros(3, height, width),
                "std": torch.ones(3, height, width)
            },
            "action": {
                "min": torch.ones(19) * -1,
                "max": torch.ones(19),
                "mean": torch.zeros(19),
                "std": torch.ones(19)
            }
        }
    
    # Override settings for faster inference
    OVERRIDE_INFERENCE_STEPS = 25
    if OVERRIDE_INFERENCE_STEPS:
        config.num_inference_steps = OVERRIDE_INFERENCE_STEPS
    
    policy = DiffusionPolicy(config, dataset_stats=stats)
    
    # Load model weights
    model_path = latest_checkpoint_dir / "pretrained_model" / "model.safetensors"
    from safetensors.torch import load_file
    state_dict = load_file(model_path)
    policy.load_state_dict(state_dict)
    policy.eval()
    policy.to("cuda")
    
    n_obs_steps = config.n_obs_steps
    horizon = config.horizon
    n_action_steps = config.n_action_steps
    action_dim = config.output_features["action"].shape[0]
    
    print(f"Policy loaded successfully!")
    print(f"  n_obs_steps: {n_obs_steps}")
    print(f"  horizon: {horizon}")
    print(f"  n_action_steps: {n_action_steps}")
    print(f"  action_dim: {action_dim}")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "policy_loaded": policy is not None})


@app.route("/inference", methods=["POST"])
def inference():
    """
    Run inference on the policy.
    
    Expected JSON payload:
    {
        "observation": {
            "state": [[...], ...],  # List of n_obs_steps state vectors (54 dims each)
            "images": {
                "tpv": [[...], ...],   # List of n_obs_steps images (H x W x 3)
                "wrist": [[...], ...],
                "overhead": [[...], ...]
            }
        }
    }
    
    Returns:
    {
        "actions": [[...], ...],  # List of action vectors (19 dims each)
        "inference_time_ms": float
    }
    """
    global policy, n_obs_steps, horizon, n_action_steps, action_dim
    
    if policy is None:
        return jsonify({"error": "Policy not loaded"}), 500
    
    data = request.json
    obs = data.get("observation", {})
    
    try:
        start_time = time.perf_counter()
        
        # Parse state observations
        state_list = obs.get("state", [])
        if len(state_list) < n_obs_steps:
            # Pad with zeros if not enough history
            while len(state_list) < n_obs_steps:
                state_list.append([0.0] * 54)
        state_list = state_list[-n_obs_steps:]
        
        # Parse image observations
        import torch.nn.functional as F
        images = obs.get("images", {})
        
        def process_image_list(img_list, target_size):
            """Process list of images to tensor."""
            if len(img_list) < n_obs_steps:
                # Pad with zeros
                while len(img_list) < n_obs_steps:
                    img_list.append(np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8))
            img_list = img_list[-n_obs_steps:]
            
            tensors = []
            for img in img_list:
                img_np = np.array(img).copy()  # Copy to avoid negative stride issues
                if img_np.shape[-1] == 3:
                    img_np = img_np[:, :, ::-1].copy()  # BGR to RGB
                tensor = torch.FloatTensor(img_np).permute(2, 0, 1) / 255.0
                tensor = F.interpolate(tensor.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
                tensors.append(tensor)
            return torch.stack(tensors)
        
        # Get image sizes from stats or use default
        img_height = DEFAULT_IMAGE_HEIGHT
        img_width = DEFAULT_IMAGE_WIDTH
        
        tpv_images = process_image_list(images.get("tpv", []), (img_height, img_width))
        wrist_images = process_image_list(images.get("wrist", []), (img_height, img_width))
        overhead_images = process_image_list(images.get("overhead", []), (img_height, img_width))
        
        # Build batch
        state_tensor = torch.FloatTensor(np.stack(state_list)).unsqueeze(0).cuda()
        tpv_tensor = tpv_images.unsqueeze(0).cuda()
        wrist_tensor = wrist_images.unsqueeze(0).cuda()
        overhead_tensor = overhead_images.unsqueeze(0).cuda()
        
        batch = {
            "observation.state": state_tensor,
            "observation.images.tpv": tpv_tensor,
            "observation.images.wrist": wrist_tensor,
            "observation.images.overhead": overhead_tensor
        }
        
        # Normalize inputs
        batch = policy.normalize_inputs(batch)
        
        # Stack images for the model
        batch["observation.images"] = torch.stack([
            batch["observation.images.tpv"],
            batch["observation.images.wrist"],
            batch["observation.images.overhead"]
        ], dim=-4)
        
        # Generate actions
        with torch.no_grad():
            actions_raw = policy.diffusion.generate_actions(batch)
            actions = policy.unnormalize_outputs({"action": actions_raw})["action"]
        
        # Extract action chunk
        action_chunk = actions[0, :n_action_steps].cpu().numpy()
        
        inference_time = (time.perf_counter() - start_time) * 1000
        
        return jsonify({
            "actions": action_chunk.tolist(),
            "inference_time_ms": inference_time,
            "action_dim": action_dim,
            "n_action_steps": n_action_steps
        })
    
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/reset", methods=["POST"])
def reset():
    """Reset the policy state."""
    global policy
    if policy is not None:
        policy.reset()
    return jsonify({"status": "ok"})


def parse_args():
    parser = argparse.ArgumentParser(description="Diffusion Policy Server")
    parser.add_argument("--ckpt-dirs", type=str, required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--port", type=int, default=5000,
                        help="Server port (default: 5000)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Server host (default: 0.0.0.0)")
    parser.add_argument("--width", type=int, default=DEFAULT_IMAGE_WIDTH,
                        help=f"Image width (default: {DEFAULT_IMAGE_WIDTH})")
    parser.add_argument("--height", type=int, default=DEFAULT_IMAGE_HEIGHT,
                        help=f"Image height (default: {DEFAULT_IMAGE_HEIGHT})")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=== Diffusion Policy Server ===")
    print(f"Loading policy from: {args.ckpt_dirs}")
    
    # Load policy
    load_policy(args.ckpt_dirs, args.width, args.height)
    
    print(f"\nStarting server on {args.host}:{args.port}")
    print("Endpoints:")
    print(f"  GET  /health    - Health check")
    print(f"  POST /inference - Run inference")
    print(f"  POST /reset     - Reset policy")
    
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
