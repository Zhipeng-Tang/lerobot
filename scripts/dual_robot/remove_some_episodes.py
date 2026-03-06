#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import shutil
import json
from pathlib import Path


def load_json(fpath):
    with open(fpath) as f:
        return json.load(f)


def write_json(data, fpath):
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, "w") as f:
        json.dump(data, f, indent=4)


def load_jsonlines(fpath):
    import jsonlines
    with jsonlines.open(fpath, "r") as reader:
        return list(reader)


def write_jsonlines(data, fpath):
    import jsonlines
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with jsonlines.open(fpath, "w") as writer:
        writer.write_all(data)


def copy_episodes(src_path, dst_path, episodes_to_keep):
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    
    print(f"Source: {src_path}")
    print(f"Destination: {dst_path}")
    print(f"Episodes to keep: {episodes_to_keep}")
    
    # Load metadata
    info = load_json(src_path / "meta" / "info.json")
    episodes = load_jsonlines(src_path / "meta" / "episodes.jsonl")
    episodes_stats = load_jsonlines(src_path / "meta" / "episodes_stats.jsonl")
    
    episodes_dict = {ep["episode_index"]: ep for ep in episodes}
    stats_dict = {stat["episode_index"]: stat for stat in episodes_stats}
    
    chunk_size = info.get("chunks_size", 1000)
    video_keys = [k for k, ft in info["features"].items() if ft.get("dtype") == "video"]
    
    # Create new metadata
    new_episodes = []
    new_stats = []
    total_frames = 0
    
    for new_idx, old_idx in enumerate(episodes_to_keep):
        # Update episode metadata
        ep = episodes_dict[old_idx].copy()
        ep["episode_index"] = new_idx
        new_episodes.append(ep)
        total_frames += ep["length"]
        
        # Update stats
        if old_idx in stats_dict:
            stat = stats_dict[old_idx].copy()
            stat["episode_index"] = new_idx
            new_stats.append(stat)
    
    # Update info
    new_info = info.copy()
    new_info["total_episodes"] = len(new_episodes)
    new_info["total_frames"] = total_frames
    new_info["total_chunks"] = (len(new_episodes) - 1) // chunk_size + 1 if new_episodes else 0
    new_info["total_videos"] = len(new_episodes) * len(video_keys) if video_keys else 0
    new_info["splits"] = {"train": f"0:{len(new_episodes)}"}
    
    # Create destination
    if dst_path.exists():
        shutil.rmtree(dst_path)
    dst_path.mkdir(parents=True)
    
    # Copy meta
    (dst_path / "meta").mkdir()
    write_json(new_info, dst_path / "meta" / "info.json")
    write_jsonlines(new_episodes, dst_path / "meta" / "episodes.jsonl")
    write_jsonlines(new_stats, dst_path / "meta" / "episodes_stats.jsonl")
    
    # Copy tasks.jsonl if exists
    tasks_path = src_path / "meta" / "tasks.jsonl"
    if tasks_path.exists():
        shutil.copy(tasks_path, dst_path / "meta" / "tasks.jsonl")
    
    # Copy parquet and video files
    for new_idx, old_idx in enumerate(episodes_to_keep):
        old_chunk = old_idx // chunk_size
        new_chunk = new_idx // chunk_size
        
        # Copy parquet
        src_parquet = src_path / f"data/chunk-{old_chunk:03d}/episode_{old_idx:06d}.parquet"
        dst_parquet = dst_path / f"data/chunk-{new_chunk:03d}/episode_{new_idx:06d}.parquet"
        if src_parquet.exists():
            dst_parquet.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_parquet, dst_parquet)
            print(f"  Copied episode_{old_idx:06d}.parquet -> episode_{new_idx:06d}.parquet")
        
        # Copy videos
        for vid_key in video_keys:
            src_video = src_path / f"videos/chunk-{old_chunk:03d}/{vid_key}/episode_{old_idx:06d}.mp4"
            dst_video = dst_path / f"videos/chunk-{new_chunk:03d}/{vid_key}/episode_{new_idx:06d}.mp4"
            if src_video.exists():
                dst_video.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src_video, dst_video)
    
    print(f"\nDone!")
    print(f"  Total episodes: {len(new_episodes)}")
    print(f"  Total frames: {total_frames}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src_path", type=str)
    parser.add_argument("dst_path", type=str)
    parser.add_argument("--episodes", nargs="+", type=int, required=True)
    args = parser.parse_args()
    
    copy_episodes(args.src_path, args.dst_path, sorted(set(args.episodes)))


if __name__ == "__main__":
    main()
