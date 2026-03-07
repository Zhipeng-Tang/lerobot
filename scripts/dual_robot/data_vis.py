import os
import argparse
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Replay recorded dual robot dataset")
    parser.add_argument("--dataset-path", help="Path to the dataset path")
    parser.add_argument("--output-path", type=str, default="outputs/data_vis", help="Path to save the stats.json")
    return parser.parse_args()

if __name__ == "__main__": 
    args = parse_args()
    data_dir = args.dataset_path
    output_dir = args.output_path
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    episode_dir = data_dir / "data/chunk-000/"
    episode_file_names = sorted(os.listdir(episode_dir))

    data = pd.read_parquet(episode_dir / episode_file_names[20])
    actions = np.stack(data["action"].to_numpy())
    states = np.stack(data["observation.state"].to_numpy())

    x = np.linspace(0, actions.shape[0], actions.shape[0])
    # import pdb; pdb.set_trace()
    for i in range(7): 
        plt.figure(figsize=(10, 6))
        plt.plot(x, actions[:,i], label="action", linewidth=2)
        plt.plot(x, states[:,i], label="state", linewidth=2)
        plt.xlabel("Step", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.title(f"arm joint dim{i}", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / f"arm_joint_dim{i}", dpi=300, bbox_inches='tight')
    print("[INFO] Finish")
