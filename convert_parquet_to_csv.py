#!/usr/bin/env python3
"""
Convert LeRobot dataset parquet files to CSV format.
"""

import pandas as pd
from pathlib import Path
from dataset_config import OUTPUT_DIR

def convert_parquet_to_csv():
    """Convert all parquet files in the dataset to CSV."""
    data_dir = OUTPUT_DIR / "data" / "chunk-000"
    output_dir = OUTPUT_DIR / "data_csv"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Find all parquet files
    parquet_files = sorted(data_dir.glob("file-*.parquet"))
    
    if not parquet_files:
        print(f"No parquet files found in {data_dir}")
        return
    
    print(f"Found {len(parquet_files)} parquet files to convert\n")
    
    for parquet_file in parquet_files:
        print(f"Converting {parquet_file.name}...")
        
        # Read parquet file
        df = pd.read_parquet(parquet_file)
        
        # Convert list columns to strings for CSV compatibility
        # observation.state and action are stored as lists
        if 'observation.state' in df.columns:
            df['observation.state'] = df['observation.state'].apply(lambda x: str(x) if isinstance(x, list) else x)
        if 'action' in df.columns:
            df['action'] = df['action'].apply(lambda x: str(x) if isinstance(x, list) else x)
        
        # Save as CSV
        csv_file = output_dir / parquet_file.name.replace('.parquet', '.csv')
        df.to_csv(csv_file, index=False)
        
        print(f"  ✓ Saved {csv_file.name} ({len(df)} rows)")
    
    print(f"\n✓ Conversion complete! CSV files saved to: {output_dir}")
    
    # Also convert metadata parquet files
    meta_dir = OUTPUT_DIR / "meta"
    meta_output_dir = OUTPUT_DIR / "meta_csv"
    meta_output_dir.mkdir(exist_ok=True, parents=True)
    
    # Convert episodes metadata (from meta/episodes/chunk-000/file-000.parquet)
    episodes_parquet = meta_dir / "episodes" / "chunk-000" / "file-000.parquet"
    if episodes_parquet.exists():
        print(f"\nConverting episodes metadata from {episodes_parquet.relative_to(OUTPUT_DIR)}...")
        df_episodes = pd.read_parquet(episodes_parquet)
        csv_episodes = meta_output_dir / "episodes.csv"
        df_episodes.to_csv(csv_episodes, index=False)
        print(f"  ✓ Saved {csv_episodes.name} ({len(df_episodes)} episodes)")
    else:
        print(f"\n⚠ Episodes metadata file not found: {episodes_parquet}")
    
    # Convert tasks
    tasks_parquet = meta_dir / "tasks.parquet"
    if tasks_parquet.exists():
        print(f"Converting tasks metadata...")
        df_tasks = pd.read_parquet(tasks_parquet)
        csv_tasks = meta_output_dir / "tasks.csv"
        df_tasks.to_csv(csv_tasks, index=False)
        print(f"  ✓ Saved {csv_tasks.name}")
    
    # Convert episode index map if it exists
    episode_map_parquet = meta_dir / "episode_index_map.parquet"
    if episode_map_parquet.exists():
        print(f"Converting episode index map...")
        df_map = pd.read_parquet(episode_map_parquet)
        csv_map = meta_output_dir / "episode_index_map.csv"
        df_map.to_csv(csv_map, index=False)
        print(f"  ✓ Saved {csv_map.name}")


if __name__ == "__main__":
    convert_parquet_to_csv()

