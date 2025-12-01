#!/usr/bin/env python3
"""
Main script to create LeRobot dataset from AirSim data.
Organized into modular functions and separate files.
"""

from pathlib import Path
from tqdm import tqdm
import cv2

from dataset_config import DATA_DIR, OUTPUT_DIR
from episode_processor import process_episode_with_video
from data_writer import save_episode_to_parquet, calculate_average_fps, IncrementalStats
from metadata_writer import create_info_json, save_metadata
from video_sync_utils import create_verification_video_from_parquet


def get_video_resolution(video_path: Path) -> tuple[int, int] | None:
    """
    Return (height, width) for a single mp4 file, or None if it can't be read.
    """
    if not video_path.exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    h, w = frame.shape[:2]
    return h, w


def main():
    """Main function to create LeRobot dataset."""
    print("Creating LeRobot Dataset")
    print("=" * 80)
    
    # Setup directories
    data_subdir = OUTPUT_DIR / "data"
    meta_dir = OUTPUT_DIR / "meta"
    data_subdir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all trials
    trials = sorted([d for d in DATA_DIR.iterdir() if d.is_dir() and d.name.startswith('T')])
    print(f"Found {len(trials)} trials\n")
    
    # Setup chunk directory
    chunk_dir = data_subdir / "chunk-000"
    chunk_dir.mkdir(exist_ok=True, parents=True)
    
    # Process episodes with streaming (write immediately, incremental stats)
    global_index = 0
    episodes_data = []
    ep_idx = 0
    stats_tracker = IncrementalStats()
    
    print("Processing and writing episodes (streaming)...")
    for trial_dir in tqdm(trials, desc="Processing"):
        csv_path = trial_dir / "enhanced_telemetry.csv"
        video_path = trial_dir / "recording.mp4"
        rec_path = trial_dir / "airsim_rec.txt"
        
        if not csv_path.exists():
            continue
        
        import pandas as pd
        df_csv = pd.read_csv(csv_path)
        
        has_video = video_path.exists()
        
        if not has_video or not rec_path.exists():
            print(f"  [{trial_dir.name}] No video or airsim_rec.txt, skipping")
            continue
        else:
            
        # Process episode
            episode_rows, episode_metadata = process_episode_with_video(
                trial_dir, ep_idx, df_csv, video_path, rec_path, global_index
            )
            if episode_rows and episode_metadata:
            # Write parquet immediately (streaming)
                file_path = save_episode_to_parquet(episode_rows, ep_idx, chunk_dir)
                print(f"  ✓ [{trial_dir.name}] Saved episode {ep_idx}: {len(episode_rows)} frames")
                # Update stats incrementally
                stats_tracker.update_batch(episode_rows)
                # Store metadata only
                episodes_data.append(episode_metadata)
                global_index += len(episode_rows)
                ep_idx += 1
    
    print(f"\nTotal frames: {global_index}")
    
    # Finalize statistics
    stats = stats_tracker.finalize()
    
    # Calculate average FPS
    print("\nCalculating FPS from videos...")
    avg_fps = calculate_average_fps(trials)
    print(f"Average FPS: {avg_fps}")

    # Infer raw image shape (height, width) from the first available video
    image_shape = None
    for trial_dir in trials:
        video_path = trial_dir / "recording.mp4"
        res = get_video_resolution(video_path)
        if res is not None:
            h, w = res
            image_shape = (h, w)
            print(f"\nDetected raw video resolution from {video_path.name}: {h}x{w}")
            break

    if image_shape is None:
        print("\n⚠ Could not detect video resolution, falling back to default 224x224 in info.json")

    # Create and save metadata (features schema now uses detected image_shape if available)
    info = create_info_json(avg_fps, len(episodes_data), global_index, image_shape=image_shape)
    saved_files = save_metadata(meta_dir, info, stats, episodes_data)
    
    # Print summary
    print(f"\n✓ Dataset created: {global_index} frames in {len(episodes_data)} episodes")
    print(f"✓ info.json saved to {saved_files['info_json']}")
    print(f"✓ stats.json saved to {saved_files['stats_json']}")
    print(f"✓ Episodes saved to {saved_files['episodes_dir']}")
    print(f"✓ Tasks saved to {saved_files['tasks_parquet']}")
    
    # Create verification video for T001
    print(f"\n{'=' * 80}")
    print("Creating verification video for T001...")
    print("=" * 80)
    
    t001_ep = next((ep for ep in episodes_data if ep['trial_name'] == 'T001' and ep['has_video']), None)
    if t001_ep:
        file_index = t001_ep['data/file_index']
        data_file = data_subdir / "chunk-000" / f"file-{file_index:03d}.parquet"
        
        trial_name = t001_ep['trial_name']
        video_path = DATA_DIR / trial_name / "recording.mp4"
        telemetry_path = DATA_DIR / trial_name / "enhanced_telemetry.csv"
        output_path = OUTPUT_DIR / f"verification_{trial_name.lower()}.mp4"
        
        if video_path.exists() and data_file.exists():
            rec_path = DATA_DIR / trial_name / "airsim_rec.txt"
            create_verification_video_from_parquet(data_file, video_path, telemetry_path, output_path, rec_path)
            print(f"✓ Verification video saved: {output_path}")
        else:
            print(f"Video file not found: {video_path}")
    else:
        print("T001 episode with video not found in dataset")


if __name__ == "__main__":
    main()
