"""Functions for processing episodes from AirSim data."""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from video_sync_utils import (
    get_first_timestamp, get_relative_timestamps,
    calculate_absolute_timestamps, match_frame_to_telemetry
)
from dataset_config import (
    OBSERVATION_FEATURES, ACTION_FEATURES, TASK_NAME, TASK_INDEX,
    SYNC_THRESHOLD, OUTPUT_DIR
)



def process_episode_with_video(trial_dir, ep_idx, df_csv, video_path, rec_path, global_index):
    """Process an episode that has a video file."""
    # Get video start timestamp
    video_start_ts = get_first_timestamp(rec_path)
    print(f"  [{trial_dir.name}] Video start: {video_start_ts:.6f}s (from TimeStamp)")
    
    # Get video properties
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if fps <= 0:
        print(f"  [{trial_dir.name}] Invalid FPS")
        return None, None
    
    # Get relative timestamps from mp4 and calculate absolute timestamps
    rel_timestamps = get_relative_timestamps(video_path, frame_count)
    video_frame_timestamps = calculate_absolute_timestamps(video_start_ts, rel_timestamps)
    
    # Match CSV rows to video frames
    episode_rows = []
    csv_timestamps = df_csv['timestamp'].values
    cap = cv2.VideoCapture(str(video_path))
    
    for video_frame_idx in range(frame_count):
        video_ts = video_frame_timestamps[video_frame_idx]
        
        # Match frame to telemetry
        nearest_idx, diff = match_frame_to_telemetry(video_ts, csv_timestamps, SYNC_THRESHOLD)
        
        if nearest_idx is None:
            continue
        
        # Read video frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        row = df_csv.iloc[nearest_idx]
        
        # Create VideoFrame with relative timestamp
        frame_time_in_video = rel_timestamps[video_frame_idx]
        video_frame = {
            'path': str(video_path.relative_to(OUTPUT_DIR.parent)),
            'timestamp': np.float32(frame_time_in_video)
        }
        
        # Build action vector, filling missing features with 0.0 (e.g. dummy dims)
        action_values = []
        for f in ACTION_FEATURES:
            if f in df_csv.columns:
                action_values.append(np.float32(row[f]))
            else:
                action_values.append(np.float32(0.0))
        
        episode_rows.append({
            'episode_index': ep_idx,
            'trial_name': trial_dir.name,  # Store folder name for reference
            'frame_index': len(episode_rows),
            'timestamp': np.float32(frame_time_in_video),
            'observation.state': [np.float32(row[f]) for f in OBSERVATION_FEATURES],
            'action': action_values,
            'instruction': TASK_NAME,
            'task_index': TASK_INDEX,
            'next.done': False,
            'index': global_index + len(episode_rows),
            'observation.images.cam_high': video_frame
        })
    
    cap.release()
    
    if not episode_rows:
        return None, None
    
    episode_rows[-1]['next.done'] = True
    
    # Extract trial number from trial_name (e.g., "T001" -> 1, "T002" -> 2)
    trial_num = int(trial_dir.name[1:]) if trial_dir.name[1:].isdigit() else ep_idx + 1
    
    episode_metadata = {
        'episode_index': ep_idx,
        'trial_name': trial_dir.name,
        'length': len(episode_rows),
        'dataset_from_index': global_index,
        'dataset_to_index': global_index + len(episode_rows) - 1,
        'data/chunk_index': 0,
        'data/file_index': ep_idx,
        'has_video': True,
        'tasks': [TASK_NAME],
        'videos/observation.images.cam_high/chunk_index': 0,
        'videos/observation.images.cam_high/file_index': trial_num,
        'videos/observation.images.cam_high/from_timestamp': 0.0
    }
    
    print(f"  [{trial_dir.name}] Saved {len(episode_rows)}/{frame_count} synchronized frames")
    return episode_rows, episode_metadata

