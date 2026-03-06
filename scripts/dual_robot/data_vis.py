import os
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # data_dir = "/home/amax/workspace/dataset/lerobot/pick_up_the_beaker"
    data_dir = "/home/amax/workspace/dataset/lerobot/toaster"
    output_dir = "outputs/data_vis"
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
