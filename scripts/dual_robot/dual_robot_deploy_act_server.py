#!/usr/bin/env python3
"""
ACT Policy Server - Model deployment with API support.
Runs on a separate machine with GPU.
"""

import argparse
import sys
from pathlib import Path
import time
import json
import threading
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from flask import Flask, request, jsonify
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.configs.types import PolicyFeature, FeatureType, NormalizationMode

app = Flask(__name__)

DEFAULT_IMAGE_WIDTH = 320
DEFAULT_IMAGE_HEIGHT = 240

policy = None
config = None
stats = None
n_obs_steps = None
chunk_size = None
n_action_steps = None
action_dim = None


def load_policy(ckpt_dirs: str, width: int, height: int):
    """Load the ACT policy from checkpoint."""
    global policy, config, stats, n_obs_steps, chunk_size, n_action_steps, action_dim
    
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
    
    # Create ACTConfig
    config = ACTConfig(
        n_obs_steps=config_dict.get('n_obs_steps', 1),
        normalization_mapping=normalization_mapping,
        input_features=input_features,
        output_features=output_features,
        device=config_dict.get('device', 'cuda'),
        chunk_size=config_dict.get('chunk_size', 48),
        n_action_steps=config_dict.get('n_action_steps', 48),
        vision_backbone=config_dict.get('vision_backbone', 'resnet18'),
        pretrained_backbone_weights=config_dict.get('pretrained_backbone_weights'),
        replace_final_stride_with_dilation=config_dict.get('replace_final_stride_with_dilation', False),
        pre_norm=config_dict.get('pre_norm', False),
        dim_model=config_dict.get('dim_model', 512),
        n_heads=config_dict.get('n_heads', 8),
        dim_feedforward=config_dict.get('dim_feedforward', 3200),
        feedforward_activation=config_dict.get('feedforward_activation', 'relu'),
        n_encoder_layers=config_dict.get('n_encoder_layers', 4),
        n_decoder_layers=config_dict.get('n_decoder_layers', 1),
        use_vae=config_dict.get('use_vae', True),
        latent_dim=config_dict.get('latent_dim', 32),
        n_vae_encoder_layers=config_dict.get('n_vae_encoder_layers', 4),
        temporal_ensemble_coeff=config_dict.get('temporal_ensemble_coeff'),
        dropout=config_dict.get('dropout', 0.1),
        kl_weight=config_dict.get('kl_weight', 10.0)
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
            "observation.images.front_left_view": {
                "mean": torch.zeros(3, height, width),
                "std": torch.ones(3, height, width)
            },
            "observation.images.wrist_view": {
                "mean": torch.zeros(3, height, width),
                "std": torch.ones(3, height, width)
            },
            "observation.images.head_view": {
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
    
    policy = ACTPolicy(config, dataset_stats=stats)
    
    # Load model weights
    model_path = latest_checkpoint_dir / "pretrained_model" / "model.safetensors"
    from safetensors.torch import load_file
    state_dict = load_file(model_path)
    policy.load_state_dict(state_dict)
    policy.eval()
    policy.to("cuda")
    
    n_obs_steps = config.n_obs_steps
    chunk_size = config.chunk_size
    n_action_steps = config.n_action_steps
    action_dim = config.output_features["action"].shape[0]
    
    print(f"Policy loaded successfully!")
    print(f"  n_obs_steps: {n_obs_steps}")
    print(f"  chunk_size: {chunk_size}")
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
                "front_left_view": [[...], ...],   # List of n_obs_steps images (H x W x 3)
                "wrist_view": [[...], ...],
                "head_view": [[...], ...]
            }
        }
    }
    
    Returns:
    {
        "actions": [[...], ...],  # List of action vectors (19 dims each)
        "inference_time_ms": float
    }
    """
    global policy, n_obs_steps, chunk_size, n_action_steps, action_dim
    
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
                img_np = np.array(img).copy()
                if img_np.shape[-1] == 3:
                    img_np = img_np[:, :, ::-1].copy()  # BGR to RGB
                tensor = torch.FloatTensor(img_np).permute(2, 0, 1) / 255.0
                tensor = F.interpolate(tensor.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
                tensors.append(tensor)
            return torch.stack(tensors)
        
        # Get image sizes from stats or use default
        img_height = DEFAULT_IMAGE_HEIGHT
        img_width = DEFAULT_IMAGE_WIDTH
        
        # Map client image keys to model input keys
        front_left_view_images = process_image_list(images.get("front_left_view", []), (img_height, img_width))
        wrist_view_images = process_image_list(images.get("wrist_view_view", []), (img_height, img_width))
        head_view_images = process_image_list(images.get("head_view", []), (img_height, img_width))
        
        # Build batch - map to model expected keys
        state_tensor = torch.FloatTensor(np.stack(state_list)).cuda()
        front_left_view_tensor = front_left_view_images.cuda()
        wrist_view_tensor = wrist_view_images.cuda()
        head_view_tensor = head_view_images.cuda()
        
        batch = {
            "observation.state": state_tensor,
            "observation.images.front_left_view": front_left_view_tensor,
            "observation.images.wrist_view": wrist_view_tensor,
            "observation.images.head_view": head_view_tensor,
            "action_is_pad": torch.zeros(1, chunk_size, dtype=torch.bool).cuda()
        }
        
        # Generate actions
        with torch.no_grad():
            action_chunk_pred = policy.predict_action_chunk(batch)
            action_chunk = action_chunk_pred[0].cpu().numpy()
        
        # Extract action chunk
        action_chunk = action_chunk[:n_action_steps]
        
        inference_time = (time.perf_counter() - start_time) * 1000
        
        return jsonify({
            "actions": action_chunk.tolist(),
            "inference_time_ms": inference_time,
            "action_dim": action_dim,
            "n_action_steps": n_action_steps,
            "chunk_size": chunk_size
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
    parser = argparse.ArgumentParser(description="ACT Policy Server")
    parser.add_argument("--ckpt-dirs", type=str, required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--port", type=int, default=5001,
                        help="Server port (default: 5001)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Server host (default: 0.0.0.0)")
    parser.add_argument("--width", type=int, default=DEFAULT_IMAGE_WIDTH,
                        help=f"Image width (default: {DEFAULT_IMAGE_WIDTH})")
    parser.add_argument("--height", type=int, default=DEFAULT_IMAGE_HEIGHT,
                        help=f"Image height (default: {DEFAULT_IMAGE_HEIGHT})")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=== ACT Policy Server ===")
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
