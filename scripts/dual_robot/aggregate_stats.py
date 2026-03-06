import json
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

if __name__=="__main__": 
    dataset_root = "/data/workspace/tangzhipeng/dataset/franka_xhand_dataset/lerobot/pick_up_the_beaker"
    output_path = "outputs/policy/pick_up_the_beaker/stats.json"

    dataset_root = Path(dataset_root)
    episodes_stats = load_episodes_stats(dataset_root)
    stats = aggregate_stats(list(episodes_stats.values()))
    stats = convert_dict_numpy2list(stats)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print("Finish")
