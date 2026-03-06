import logging
import argparse
from tqdm import tqdm
from pathlib import Path
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging, log_say

# Initialize logging
init_logging()
logger = logging.getLogger(__name__)

def load_dataset(dataset_path: str) -> LeRobotDataset:
    """Load the recorded dataset."""
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found. Looking for: {dataset_path}")
    
    logger.info(f"Loading dataset from: {dataset_path}")
    dataset = LeRobotDataset(str(dataset_path))
    return dataset

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Replay recorded dual robot dataset")
    parser.add_argument("--source-dir", help="Path to the dataset directory")
    parser.add_argument("--output-dir", type=str, help="Directory to save the converted dataset")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Load dataset
    logger.info(f"Loading dataset from: {args.source_dir}")
    dataset = load_dataset(args.source_dir)

    features = dataset.meta.info["features"]
    default_features_key = ["timestamp", "frame_index", "episode_index", "index", "task_index"]
    for key in default_features_key: 
        features.pop(key, None)  # Remove default features if they exist
    
    output_dataset = LeRobotDataset.create(
        repo_id=str(args.output_dir),
        fps=dataset.meta.info["fps"],
        features=features,
        robot_type=dataset.meta.info["robot_type"],
        use_videos=True,  # Enable videos for proper storage
        image_writer_threads=4,
    )

    pbar = tqdm(total=len(dataset.episode_data_index["from"]), desc="Converting dataset")
    for episode_idx in range(len(dataset.episode_data_index["from"])): 
        start_idx = dataset.episode_data_index["from"][episode_idx]
        end_idx = dataset.episode_data_index["to"][episode_idx]
        for idx in range(start_idx, end_idx-1): 
            data = dataset[idx]
            next_data  = dataset[idx+1]

            output_dataset.add_frame(
                {
                    "observation.images.head_view": (data["observation.images.head_view"].permute(1, 2, 0) * 255).numpy().astype(np.uint8), 
                    "observation.images.front_left_view": (data["observation.images.front_left_view"].permute(1, 2, 0) * 255).numpy().astype(np.uint8),
                    "observation.images.wrist_view": (data["observation.images.wrist_view"].permute(1, 2, 0) * 255).numpy().astype(np.uint8),
                    "observation.state": data["observation.state"],
                    "action": next_data["observation.state"][[_ for _ in range(7)]+[i+30 for i in range(12)]]
                }, 
                task = data["task"]
            )
        output_dataset.save_episode()
        pbar.update(1)
    print("[INFO] Finish!")

