"""Functions for writing data to parquet files and computing statistics."""

import numpy as np
import pandas as pd
import cv2
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
from dataset_config import OUTPUT_DIR, ACTION_FEATURES, OBSERVATION_FEATURES


def compute_delta_actions(action_arrays):
    return action_arrays
    """Compute delta actions: first frame = zeros, subsequent frames = current - previous."""
    return [np.zeros_like(action_arrays[0])] + [
        action_arrays[i] - action_arrays[i-1] 
        for i in range(1, len(action_arrays))
    ]


def compute_image_stats_simple(episode_rows, sample_rate=100):
    """Bare minimal: compute mean/std for each RGB channel."""
    pixels_r, pixels_g, pixels_b = [], [], []
    
    for i, row in enumerate(episode_rows):
        if i % sample_rate != 0 or 'observation.images.cam_high' not in row:
            continue
        
        vf = row['observation.images.cam_high']
        video_path = OUTPUT_DIR.parent / vf['path'] if not Path(vf['path']).is_absolute() else Path(vf['path'])
        if not video_path.exists():
            continue
        
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_MSEC, vf['timestamp'] * 1000.0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            continue
        
        if frame.shape[:2] != (224, 224):
            frame = cv2.resize(frame, (224, 224))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        pixels_r.extend(frame_rgb[:, :, 0].flatten())
        pixels_g.extend(frame_rgb[:, :, 1].flatten())
        pixels_b.extend(frame_rgb[:, :, 2].flatten())
    
    if not pixels_r:
        return {'mean': [0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0], 'min': [0.0, 0.0, 0.0], 'max': [1.0, 1.0, 1.0]}
    
    return {
        'mean': [np.mean(pixels_r), np.mean(pixels_g), np.mean(pixels_b)],
        'std': [np.std(pixels_r), np.std(pixels_g), np.std(pixels_b)],
        'min': [np.min(pixels_r), np.min(pixels_g), np.min(pixels_b)],
        'max': [np.max(pixels_r), np.max(pixels_g), np.max(pixels_b)]
    }


class IncrementalStats:
    """Incremental statistics using Welford's algorithm - memory efficient."""
    def __init__(self):
        # State: state_dim dims
        state_dim = len(OBSERVATION_FEATURES)
        action_dim = len(ACTION_FEATURES)

        self.state_n, self.state_mean, self.state_M2 = 0, np.zeros(state_dim, dtype=np.float64), np.zeros(state_dim, dtype=np.float64)
        self.state_min, self.state_max = np.full(state_dim, np.inf), np.full(state_dim, -np.inf)
        # Action: action_dim dims (delta actions)
        self.action_n, self.action_mean, self.action_M2 = 0, np.zeros(action_dim, dtype=np.float64), np.zeros(action_dim, dtype=np.float64)
        self.action_min, self.action_max = np.full(action_dim, np.inf), np.full(action_dim, -np.inf)
        # Store episode rows for image stats computation at the end
        self.all_episode_rows = []
    
    def update_batch(self, episode_rows):
        """Update stats from episode rows."""
        self.all_episode_rows.extend(episode_rows)
        action_arrays = [np.array(r['action'], dtype=np.float32) for r in episode_rows]
        delta_actions = compute_delta_actions(action_arrays)
        for row, delta in zip(episode_rows, delta_actions):
            s = np.array(row['observation.state'], dtype=np.float32)
            a = np.array(delta, dtype=np.float32)
            # State: Welford's algorithm
            self.state_n += 1
            delta_s = s - self.state_mean
            self.state_mean += delta_s / self.state_n
            self.state_M2 += delta_s * (s - self.state_mean)
            self.state_min, self.state_max = np.minimum(self.state_min, s), np.maximum(self.state_max, s)
            # Action: Welford's algorithm
            self.action_n += 1
            delta_a = a - self.action_mean
            self.action_mean += delta_a / self.action_n
            self.action_M2 += delta_a * (a - self.action_mean)
            self.action_min, self.action_max = np.minimum(self.action_min, a), np.maximum(self.action_max, a)
    
    def finalize(self):
        """Return stats in LeRobot format."""
        stats = {}
        if self.state_n > 0:
            state_std = np.sqrt(self.state_M2 / self.state_n) if self.state_n > 1 else np.zeros(self.state_dim)
            stats['observation.state'] = {
                'mean': self.state_mean.astype(np.float32).tolist(),
                'std': state_std.astype(np.float32).tolist(),
                'min': self.state_min.tolist(),
                'max': self.state_max.tolist()
            }
        else:
            stats['observation.state'] = {}
        if self.action_n > 0:
            action_std = np.sqrt(self.action_M2 / self.action_n) if self.action_n > 1 else np.zeros(self.action_dim)
            stats['action'] = {
                'mean': self.action_mean.astype(np.float32).tolist(),
                'std': action_std.astype(np.float32).tolist(),
                'min': self.action_min.tolist(),
                'max': self.action_max.tolist()
            }
        else:
            stats['action'] = {}
        
        # Image stats: Compute from sampled frames (bare minimal)
        stats['observation.images.cam_high'] = compute_image_stats_simple(self.all_episode_rows)
        
        return stats


def save_episode_to_parquet(episode_rows, ep_idx, chunk_dir):
    """Save a single episode to a parquet file."""
    df_ep = pd.DataFrame(episode_rows)
    
    # Select required columns in the correct order (matching info.json schema)
    # Order: episode_index, timestamp, index, frame_index, task_index, observation.state, action, observation.images.cam_high, next.done
    columns = ['episode_index', 'timestamp', 'index']
    if 'frame_index' in df_ep.columns:
        columns.append('frame_index')
    columns.extend(['task_index', 'observation.state', 'action'])
    if 'observation.images.cam_high' in df_ep.columns:
        columns.append('observation.images.cam_high')
    columns.append('next.done')
    
    df_data = df_ep[columns].copy()
    
    # IMPORTANT: Replace global indices with local indices (0-based per file)
    # LeRobot uses dataset_from_index from episode metadata to map global->local
    df_data['index'] = np.arange(len(df_data), dtype=np.int64)
    
    # Convert lists to numpy arrays
    observation_arrays = [np.array(x, dtype=np.float32) for x in df_data['observation.state']]
    action_arrays = [np.array(x, dtype=np.float32) for x in df_data['action']]
    
    # Compute delta actions
    # delta_action_arrays = compute_delta_actions(action_arrays)
    delta_action_arrays = action_arrays

    # Ensure action vectors all have length len(ACTION_FEATURES)
    target_action_dim = len(ACTION_FEATURES)
    fixed_delta_action_arrays = []
    for a in delta_action_arrays:
        a = np.asarray(a, dtype=np.float32)
        if a.shape[0] < target_action_dim:
            a = np.pad(a, (0, target_action_dim - a.shape[0]), mode="constant", constant_values=0.0)
        elif a.shape[0] > target_action_dim:
            a = a[:target_action_dim]
        fixed_delta_action_arrays.append(a)

    # Create PyArrow arrays
    obs_array = pa.array(
        [pa.array(x, type=pa.float32()) for x in observation_arrays],
        type=pa.list_(pa.float32()),
    )

    # Always store actions as list<float> with fixed length == len(ACTION_FEATURES)
    action_array = pa.array(
        [pa.array(x, type=pa.float32()) for x in fixed_delta_action_arrays],
        type=pa.list_(pa.float32()),
    )
    
    # Build table in the correct order (matching info.json schema)
    # Order: episode_index, timestamp, index, frame_index, task_index, observation.state, action, next.done
    table_columns = {
        'episode_index': pa.array(df_data['episode_index'].values, type=pa.int64()),
        'timestamp': pa.array(df_data['timestamp'].values, type=pa.float32()),
        'index': pa.array(df_data['index'].values, type=pa.int64()),  # Local indices (0, 1, 2, ...)
    }
    
    # Add frame_index if present (must be after index, before task_index)
    if 'frame_index' in df_data.columns:
        table_columns['frame_index'] = pa.array(df_data['frame_index'].values, type=pa.int64())
    
    # Add remaining columns
    table_columns.update({
        'task_index': pa.array(df_data['task_index'].values, type=pa.int64()),
        'observation.state': obs_array,
        'action': action_array,
        'next.done': pa.array(df_data['next.done'].values, type=pa.bool_()),
    })
    
    table = pa.table(table_columns)
    
    # Save to parquet
    file_path = chunk_dir / f"file-{ep_idx:03d}.parquet"
    pq.write_table(table, file_path, compression='snappy')
    
    return file_path


def compute_state_stats(df_all):
    """Compute statistics for observation.state."""
    state_data = np.array([np.array(x) for x in df_all['observation.state']])
    return {
        'max': state_data.max(axis=0).tolist(),
        'min': state_data.min(axis=0).tolist(),
        'mean': state_data.mean(axis=0).tolist(),
        'std': state_data.std(axis=0).tolist()
    }


def compute_action_stats(df_all):
    """Compute statistics for actions (using delta actions)."""
    # Compute delta actions for stats
    action_arrays = [np.array(x, dtype=np.float32) for x in df_all['action']]
    delta_action_arrays = compute_delta_actions(action_arrays)
    
    delta_action_data = np.array(delta_action_arrays)
    return {
        'max': delta_action_data.max(axis=0).tolist(),
        'min': delta_action_data.min(axis=0).tolist(),
        'mean': delta_action_data.mean(axis=0).tolist(),
        'std': delta_action_data.std(axis=0).tolist()
    }


def compute_image_stats(df_all, sample_rate=None):
    """Compute image statistics from sampled video frames."""
    image_pixels = []
    if sample_rate is None:
        sample_rate = max(1, len(df_all) // 1000)  # Sample up to 1000 frames
    
    for idx, row in enumerate(tqdm(df_all.iterrows(), total=len(df_all), desc="Sampling frames")):
        if idx % sample_rate != 0:
            continue
        
        _, row = row
        vf = row['observation.images.cam_high']
        if vf is None:
            continue
        
        # Load video frame
        video_path_str = vf['path']
        video_path = Path(video_path_str)
        if not video_path.is_absolute():
            video_path = OUTPUT_DIR.parent / video_path
        
        if not video_path.exists():
            continue
        
        cap = cv2.VideoCapture(str(video_path))
        frame_time = vf['timestamp']
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = int(frame_time * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            # Resize to expected size if needed (224x224)
            if frame.shape[:2] != (224, 224):
                frame = cv2.resize(frame, (224, 224))
            # Convert BGR to RGB and normalize to [0, 1]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
            image_pixels.append(frame_rgb.reshape(-1, 3))
    
    if image_pixels:
        all_pixels = np.vstack(image_pixels)
        return {
            'mean': all_pixels.mean(axis=0).tolist(),
            'std': all_pixels.std(axis=0).tolist(),
            'min': all_pixels.min(axis=0).tolist(),
            'max': all_pixels.max(axis=0).tolist()
        }
    return {}


def compute_all_stats(all_rows_for_stats):
    """Compute all statistics from episode rows."""
    if not all_rows_for_stats:
        return {
            'observation.state': {},
            'action': {},
            'observation.images.cam_high': {}
        }
    
    df_all = pd.DataFrame(all_rows_for_stats)
    
    stats = {
        'observation.state': compute_state_stats(df_all),
        'action': compute_action_stats(df_all),
    }
    
    # Compute image stats
    print("\nComputing image statistics...")
    image_stats = compute_image_stats(df_all)
    stats['observation.images.cam_high'] = image_stats
    
    if image_stats:
        print(f"✓ Computed image stats from sampled frames")
    else:
        print("⚠ No video frames found for image stats computation")
    
    return stats


def calculate_average_fps(trials):
    """Calculate average FPS from all video files."""
    fps_values = []
    for trial_dir in trials:
        video_path = trial_dir / "recording.mp4"
        if video_path.exists():
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps > 0:
                fps_values.append(fps)
    return int(np.mean(fps_values)) if fps_values else 30

