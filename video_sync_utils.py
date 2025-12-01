"""Utility functions for video-telemetry synchronization."""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm


def get_first_timestamp(rec_path):
    """Get first timestamp from airsim_rec.txt (convert ms to seconds if needed)."""
    df_rec = pd.read_csv(rec_path, sep='\t')
    timestamp = df_rec['TimeStamp'].iloc[0]
    return timestamp / 1000.0  # Convert ms to seconds


def get_relative_timestamps(video_path, frame_count):
    """Extract relative timestamps from mp4 file for each frame."""
    cap = cv2.VideoCapture(str(video_path))
    rel_timestamps = []
    for i in range(frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        cap.read()
        rel_timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)  # ms to seconds
    cap.release()
    return rel_timestamps


def calculate_absolute_timestamps(first_timestamp, rel_timestamps):
    """Calculate absolute timestamps: first_timestamp + relative_timestamp."""
    return [first_timestamp + ts for ts in rel_timestamps]


def match_frame_to_telemetry(frame_abs_ts, telemetry_ts, threshold=0.1):
    """Match frame timestamp to nearest telemetry entry. Returns (index, diff) or None if outside threshold."""
    diffs = np.abs(telemetry_ts - frame_abs_ts)
    nearest_idx = np.argmin(diffs)
    diff = diffs[nearest_idx]
    if diff <= threshold:
        return nearest_idx, diff
    return None, diff


def create_verification_video(episode_df, video_path, telemetry_path, output_path):
    """Create verification video showing synchronized frames with telemetry data."""
    # Load telemetry
    df_telemetry = pd.read_csv(telemetry_path)
    telemetry_ts = df_telemetry['timestamp'].values
    
    # Get video properties
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    for idx, row in enumerate(tqdm(episode_df.iterrows(), total=len(episode_df), desc="Writing video")):
        _, row = row
        
        # Get frame index from relative timestamp
        vf = row['observation.images.cam_high']
        frame_time_in_video = vf['timestamp']
        frame_idx = int(frame_time_in_video * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Find matched telemetry
        abs_timestamp = row['timestamp']
        diffs = np.abs(telemetry_ts - abs_timestamp)
        matched_idx = np.argmin(diffs)
        matched_telemetry_ts = telemetry_ts[matched_idx]
        time_diff_ms = (abs_timestamp - matched_telemetry_ts) * 1000
        
        # Get data
        action = row['action']
        state = row['observation.state']
        altitude = state[10]  # Altitude is at index 10 in observation.state
        
        # Add overlays
        y_pos = 30
        line_height = 30
        font_scale = 0.6
        thickness = 2
        color_green = (0, 255, 0)
        color_yellow = (0, 255, 255)
        color_cyan = (255, 255, 0)
        
        cv2.putText(frame, f"Frame: {idx} | Video: {frame_idx}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_green, thickness)
        y_pos += line_height
        
        cv2.putText(frame, f"Rel Time: {frame_time_in_video:.3f}s", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_green, thickness)
        y_pos += line_height
        
        cv2.putText(frame, f"Abs Time: {abs_timestamp:.6f}s", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_yellow, thickness)
        y_pos += line_height
        
        cv2.putText(frame, f"Telemetry: {matched_telemetry_ts:.6f}s", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_yellow, thickness)
        y_pos += line_height
        
        sync_color = color_green if abs(time_diff_ms) < 10 else color_yellow if abs(time_diff_ms) < 50 else (0, 0, 255)
        cv2.putText(frame, f"Sync Diff: {time_diff_ms:.2f}ms", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, sync_color, thickness)
        y_pos += line_height
        
        cv2.putText(frame, f"Altitude: {altitude:.2f}m", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_cyan, thickness)
        y_pos += line_height
        
        cv2.putText(frame, f"Action: T={action[0]:.3f} R={action[1]:.3f} P={action[2]:.3f} Y={action[3]:.3f}",
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_cyan, thickness)
        
        out.write(frame)
    
    cap.release()
    out.release()


def create_verification_video_from_parquet(parquet_path, video_path, telemetry_path, output_path, rec_path=None):
    """
    Create verification video from parquet file.
    Works with the LeRobot dataset parquet format.
    
    Args:
        parquet_path: Path to episode parquet file
        video_path: Path to original video file
        telemetry_path: Path to enhanced_telemetry.csv
        output_path: Path to save verification video
        rec_path: Optional path to airsim_rec.txt (for absolute timestamp calculation)
    """
    # Load parquet data
    df_parquet = pd.read_parquet(parquet_path)
    
    # Load telemetry
    df_telemetry = pd.read_csv(telemetry_path)
    telemetry_ts = df_telemetry['timestamp'].values
    
    # Get video properties
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get first timestamp if rec_path provided (for absolute time calculation)
    first_timestamp = None
    if rec_path and Path(rec_path).exists():
        first_timestamp = get_first_timestamp(rec_path)
    
    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    print(f"Creating verification video from {len(df_parquet)} frames...")
    
    for idx, row in enumerate(tqdm(df_parquet.itertuples(), total=len(df_parquet), desc="Writing video")):
        # Get relative timestamp from parquet (this is frame_time_in_video)
        rel_timestamp = row.timestamp
        frame_idx = int(rel_timestamp * fps)
        
        # Clamp frame index to valid range
        frame_idx = min(frame_idx, total_frames - 1)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate absolute timestamp if available
        abs_timestamp = first_timestamp + rel_timestamp if first_timestamp else rel_timestamp
        
        # Find matched telemetry
        diffs = np.abs(telemetry_ts - abs_timestamp)
        matched_idx = np.argmin(diffs)
        matched_telemetry_ts = telemetry_ts[matched_idx]
        time_diff_ms = (abs_timestamp - matched_telemetry_ts) * 1000
        
        # Get data from parquet (convert lists to arrays)
        action = np.array(row.action) if isinstance(row.action, list) else row.action
        state = np.array(row.observation.state) if isinstance(row.observation.state, list) else row.observation.state
        
        # Extract values (delta actions in parquet, but we show them as-is)
        altitude = float(state[10]) if len(state) > 10 else 0.0  # Altitude is at index 10
        
        # Add overlays
        y_pos = 30
        line_height = 30
        font_scale = 0.6
        thickness = 2
        color_green = (0, 255, 0)
        color_yellow = (0, 255, 255)
        color_cyan = (255, 255, 0)
        color_white = (255, 255, 255)
        
        # Episode info
        cv2.putText(frame, f"Episode: {row['episode_index']} | Frame: {idx}/{len(df_parquet)-1}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_white, thickness)
        y_pos += line_height
        
        cv2.putText(frame, f"Video Frame: {frame_idx}/{total_frames-1}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_green, thickness)
        y_pos += line_height
        
        cv2.putText(frame, f"Rel Time: {rel_timestamp:.3f}s", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_green, thickness)
        y_pos += line_height
        
        if first_timestamp:
            cv2.putText(frame, f"Abs Time: {abs_timestamp:.6f}s", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_yellow, thickness)
            y_pos += line_height
            
            cv2.putText(frame, f"Telemetry: {matched_telemetry_ts:.6f}s", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_yellow, thickness)
            y_pos += line_height
            
            sync_color = color_green if abs(time_diff_ms) < 10 else color_yellow if abs(time_diff_ms) < 50 else (0, 0, 255)
            cv2.putText(frame, f"Sync Diff: {time_diff_ms:.2f}ms", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, sync_color, thickness)
            y_pos += line_height
        
        # State data
        cv2.putText(frame, f"Altitude: {altitude:.2f}m", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_cyan, thickness)
        y_pos += line_height
        
        # Action data (delta actions from parquet)
        if len(action) >= 4:
            cv2.putText(frame, f"Delta Action: T={action[0]:.3f} R={action[1]:.3f} P={action[2]:.3f} Y={action[3]:.3f}",
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_cyan, thickness)
            y_pos += line_height
        
        # Global index
        cv2.putText(frame, f"Global Index: {row['index']}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_white, thickness)
        
        out.write(frame)
    
    cap.release()
    out.release()
    print(f"✓ Verification video saved: {output_path}")

