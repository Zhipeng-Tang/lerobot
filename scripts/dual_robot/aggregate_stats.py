import os
import json
import argparse
from pathlib import Path
from lerobot.datasets.utils import load_episodes_stats
from lerobot.datasets.compute_stats import aggregate_stats

def convert_dict_numpy2list(obj): 
    if isinstance(obj, dict): 
        return {
            key: convert_dict_numpy2list(value) for key, value in obj.items()
        }
    else: 
        return obj.tolist()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Replay recorded dual robot dataset")
    parser.add_argument("--dataset-path", help="Path to the dataset path")
    parser.add_argument("--output-path", type=str, help="Path to save the stats.json")
    return parser.parse_args()

if __name__=="__main__": 
    args = parse_args()
    dataset_root = args.dataset_path
    output_path = os.path.join(args.output_path, "stats.json")

    dataset_root = Path(dataset_root)
    episodes_stats = load_episodes_stats(dataset_root)
    stats = aggregate_stats(list(episodes_stats.values()))
    stats = convert_dict_numpy2list(stats)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print("Finish")
