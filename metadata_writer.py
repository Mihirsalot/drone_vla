"""Functions for writing metadata files (info.json, stats.json, episodes, tasks)."""

import json
import pandas as pd
from pathlib import Path
from dataset_config import OBSERVATION_FEATURES, ACTION_FEATURES


def create_features_schema(image_shape: tuple[int, int] | None = None):
    """
    Create the features schema for info.json.

    Args:
        image_shape: Optional (height, width) of the raw video frame.
            If None, defaults to (224, 224).
    """
    # Default to 224x224 if no shape was provided
    if image_shape is None:
        height, width = 224, 224
    else:
        height, width = image_shape

    return {
        'episode_index': {'dtype': 'int64', 'shape': [1]},
        'timestamp': {'dtype': 'float32', 'shape': [1]},
        'index': {'dtype': 'int64', 'shape': [1]},
        'frame_index': {'dtype': 'int64', 'shape': [1]},  # Frame index within episode (for visualization)
        'task_index': {'dtype': 'int64', 'shape': [1]},
        'observation.state': {
            'dtype': 'float32',
            'shape': [len(OBSERVATION_FEATURES)],
            'names': OBSERVATION_FEATURES
        },
        'action': {
            'dtype': 'float32',
            'shape': [len(ACTION_FEATURES)],
            'names': ACTION_FEATURES
        },
        'observation.images.cam_high': {
            'dtype': 'video',
            # LeRobot convention: [channels, height, width]
            'shape': [3, height, width],
            'names': ['channels', 'height', 'width']
        },
        'next.done': {'dtype': 'bool', 'shape': [1]}
    }


def create_info_json(avg_fps, total_episodes, total_frames, image_shape: tuple[int, int] | None = None):
    """
    Create info.json content.

    Args:
        avg_fps: Average FPS across all videos.
        total_episodes: Number of episodes.
        total_frames: Total number of frames.
        image_shape: Optional (height, width) of raw video frames.
    """
    return {
        'codebase_version': '3.0',
        'fps': avg_fps,
        'features': create_features_schema(image_shape=image_shape),
        'total_episodes': total_episodes,
        'total_frames': total_frames,
        'robot_type': 'airsim_drone',
        'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
        'video_path': '../AirSim 5M Hover Data/T{file_index:03d}/recording.mp4'
    }


def save_metadata(meta_dir, info, stats, episodes_data, episode_index_records=None):
    """Save all metadata files."""
    # Save info.json
    with open(meta_dir / "info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    # Save stats.json
    with open(meta_dir / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Save episodes metadata
    episodes_df = pd.DataFrame(episodes_data)
    episodes_dir = meta_dir / "episodes"
    episodes_chunk_dir = episodes_dir / "chunk-000"
    episodes_chunk_dir.mkdir(exist_ok=True, parents=True)
    episodes_df.to_parquet(episodes_chunk_dir / "file-000.parquet", compression='snappy', index=False)
    
    # Save episode index map (includes missing/empty episodes)
    episode_index_map_path = None
    if episode_index_records:
        episode_index_df = pd.DataFrame(episode_index_records)
        episode_index_map_path = meta_dir / "episode_index_map.parquet"
        episode_index_df.to_parquet(episode_index_map_path, compression='snappy', index=False)
    
    # Save tasks
    tasks_df = pd.DataFrame({
        'task_index': [0]
    }, index=['takeoff and hover'])
    tasks_df.to_parquet(meta_dir / "tasks.parquet", compression='snappy')
    
    return {
        'info_json': meta_dir / "info.json",
        'stats_json': meta_dir / "stats.json",
        'episodes_dir': episodes_dir,
        'tasks_parquet': meta_dir / "tasks.parquet",
        'episode_index_map': episode_index_map_path
    }

