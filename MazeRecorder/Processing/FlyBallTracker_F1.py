#!/usr/bin/env python3
"""
Batch track ball and fly in F1 videos using SLEAP.
Supports processing individual experiments or reading from YAML files.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import yaml


# Default paths
DATAFOLDER = Path("/mnt/upramdya_data/MD/F1_Tracks/Videos")
MODEL_BALL_CENTROID = Path("/mnt/upramdya_data/_Tracking_models/Sleap/mazerecorder/BallTracking/models/240926_141251.centroid.n=102")
MODEL_BALL_CENTERED = Path("/mnt/upramdya_data/_Tracking_models/Sleap/mazerecorder/BallTracking/models/240926_151129.centered_instance.n=102")
MODEL_FLY = Path("/mnt/upramdya_data/_Tracking_models/Sleap/mazerecorder/FlyTracking/Thorax/Labels/models/240924_164931.single_instance.n=192")


class TrackingStatus:
    """Track the status of video processing."""
    def __init__(self):
        self.videos_to_process = []  # (video_path, track_type, process_type)
        self.processed_videos = []
        self.directories = {}  # directory -> (processed, total)
    
    def add_to_process(self, video: Path, track_type: str, process_type: str):
        """Add a video to the processing queue."""
        self.videos_to_process.append((video, track_type, process_type))
    
    def add_processed(self, video: Path):
        """Mark a video as processed."""
        if video not in self.processed_videos:
            self.processed_videos.append(video)
    
    def update_directory_status(self, directory: Path, processed: int, total: int):
        """Update directory processing status."""
        self.directories[directory] = (processed, total)
    
    def print_summary(self, check_mode: bool = False):
        """Print processing summary."""
        print("\n" + "="*80)
        if check_mode:
            print("CHECK MODE RESULTS")
        else:
            print("PROCESSING SUMMARY")
        print("="*80)
        
        # Separate fully and partially processed directories
        fully_processed = []
        partially_processed = []
        
        for directory, (processed, total) in self.directories.items():
            if processed == total and total > 0:
                fully_processed.append((directory, processed, total))
            elif total > 0:
                partially_processed.append((directory, processed, total))
        
        if fully_processed:
            print("\n✅ FULLY PROCESSED DIRECTORIES:")
            for directory, processed, total in sorted(fully_processed):
                print(f"  - {directory.name} ({processed}/{total} videos)")
        
        if partially_processed:
            print("\n❌ DIRECTORIES WITH UNPROCESSED VIDEOS:")
            for directory, processed, total in sorted(partially_processed):
                print(f"  - {directory.name} ({processed}/{total} videos)")
        
        if self.videos_to_process:
            print(f"\n📋 ITEMS TO PROCESS ({len(self.videos_to_process)}):")
            for video, track_type, process_type in self.videos_to_process:
                video_name = video.stem
                directory_name = video.parent.name
                print(f"  - {directory_name}/{video_name}: {track_type} {process_type}")
        
        print(f"\n=== SUMMARY ===")
        print(f"Directories fully processed: {len(fully_processed)}")
        print(f"Directories with unprocessed videos: {len(partially_processed)}")
        print(f"Videos needing processing: {len(self.videos_to_process)}")
        print(f"Videos already processed: {len(self.processed_videos)}")
        print("="*80)


def check_tracking_files(video_path: Path) -> Dict[str, Dict[str, bool]]:
    """
    Check if tracking files exist for a video.
    
    Returns:
        Dict with 'ball' and 'fly' status, each containing 'slp' and 'h5' bools
    """
    output_folder = video_path.parent
    
    # Check for ball tracking files
    slp_files_ball = list(output_folder.glob("*tracked_ball*.slp"))
    h5_files_ball = list(output_folder.glob("*tracked_ball*.h5")) + \
                    list(output_folder.glob("*tracked_ball*.analysis.h5"))
    
    # Check for fly tracking files
    slp_files_fly = list(output_folder.glob("*tracked_fly*.slp"))
    h5_files_fly = list(output_folder.glob("*tracked_fly*.h5")) + \
                   list(output_folder.glob("*tracked_fly*.analysis.h5"))
    
    return {
        'ball': {
            'slp': len(slp_files_ball) > 0,
            'h5': len(h5_files_ball) > 0
        },
        'fly': {
            'slp': len(slp_files_fly) > 0,
            'h5': len(h5_files_fly) > 0
        }
    }


def scan_directory(directory: Path, status: TrackingStatus, verbose: bool = False):
    """Scan a directory for videos and check their tracking status."""
    # Only process _Checked directories
    if "_Checked" not in directory.name:
        return
    
    if verbose:
        print(f"Scanning directory: {directory.name}")
    
    # Find all videos
    videos = list(directory.glob("*.mp4"))
    
    if not videos:
        return
    
    dir_processed_count = 0
    dir_total_count = len(videos)
    
    for video in videos:
        tracking_status = check_tracking_files(video)
        video_fully_processed = True
        
        # Check ball tracking
        if not tracking_status['ball']['slp']:
            status.add_to_process(video, 'ball', 'slp')
            video_fully_processed = False
            if verbose:
                print(f"  {video.name}: needs ball tracking")
        elif not tracking_status['ball']['h5']:
            status.add_to_process(video, 'ball', 'h5')
            video_fully_processed = False
            if verbose:
                print(f"  {video.name}: needs ball h5 conversion")
        
        # Check fly tracking
        if not tracking_status['fly']['slp']:
            status.add_to_process(video, 'fly', 'slp')
            video_fully_processed = False
            if verbose:
                print(f"  {video.name}: needs fly tracking")
        elif not tracking_status['fly']['h5']:
            status.add_to_process(video, 'fly', 'h5')
            video_fully_processed = False
            if verbose:
                print(f"  {video.name}: needs fly h5 conversion")
        
        if video_fully_processed:
            status.add_processed(video)
            dir_processed_count += 1
    
    status.update_directory_status(directory, dir_processed_count, dir_total_count)


def track_video(video_path: Path, track_type: str, dry_run: bool = False) -> bool:
    """
    Run SLEAP tracking on a video.
    
    Args:
        video_path: Path to video file
        track_type: 'ball' or 'fly'
        dry_run: If True, only print command without executing
    
    Returns:
        True if successful, False otherwise
    """
    video_name = video_path.stem
    output_folder = video_path.parent
    
    if track_type == 'ball':
        output_file = output_folder / f"{video_name}_tracked_ball.slp"
        cmd = [
            "sleap-track",
            str(video_path),
            "--model", str(MODEL_BALL_CENTROID),
            "--model", str(MODEL_BALL_CENTERED),
            "--batch_size", "16",
            "--max_instances", "2",
            "--output", str(output_file),
            "--verbosity", "rich"
        ]
    elif track_type == 'fly':
        output_file = output_folder / f"{video_name}_tracked_fly.slp"
        cmd = [
            "sleap-track",
            str(video_path),
            "--model", str(MODEL_FLY),
            "--batch_size", "16",
            "--output", str(output_file),
            "--verbosity", "rich"
        ]
    else:
        print(f"Error: Unknown track type: {track_type}")
        return False
    
    print(f"\nTracking {track_type} for: {video_path.name}")
    print(f"Command: {' '.join(cmd)}")
    
    if dry_run:
        print("  [DRY RUN - not executing]")
        return True
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✓ {track_type.capitalize()} tracking complete for: {video_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {track_type.capitalize()} tracking failed for: {video_path.name}")
        print(f"Error: {e}")
        return False


def convert_to_h5(video_path: Path, track_type: str, dry_run: bool = False) -> bool:
    """
    Convert .slp file to .h5 format.
    
    Args:
        video_path: Path to video file
        track_type: 'ball' or 'fly'
        dry_run: If True, only print command without executing
    
    Returns:
        True if successful, False otherwise
    """
    video_name = video_path.stem
    output_folder = video_path.parent
    slp_file = output_folder / f"{video_name}_tracked_{track_type}.slp"
    
    if not slp_file.exists():
        print(f"Error: .slp file not found: {slp_file}")
        return False
    
    cmd = [
        "sleap-convert",
        str(slp_file),
        "--format", "analysis"
    ]
    
    print(f"\nConverting {track_type} tracking to h5 for: {video_path.name}")
    print(f"Command: {' '.join(cmd)}")
    
    if dry_run:
        print("  [DRY RUN - not executing]")
        return True
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✓ {track_type.capitalize()} h5 conversion complete for: {video_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {track_type.capitalize()} h5 conversion failed for: {video_path.name}")
        print(f"Error: {e}")
        return False


def process_videos(status: TrackingStatus, dry_run: bool = False):
    """Process all videos in the queue."""
    total = len(status.videos_to_process)
    
    if total == 0:
        print("\n🎉 ALL VIDEOS ARE ALREADY FULLY PROCESSED!")
        return True
    
    print(f"\n🚀 PROCESSING {total} ITEMS...")
    print("="*80)
    
    success_count = 0
    fail_count = 0
    
    for i, (video, track_type, process_type) in enumerate(status.videos_to_process, 1):
        print(f"\n[{i}/{total}] Processing: {video.parent.name}/{video.name}")
        print(f"  Type: {track_type} {process_type}")
        
        if process_type == 'slp':
            # Run tracking
            success = track_video(video, track_type, dry_run)
            if success and not dry_run:
                # Also convert to h5
                success = convert_to_h5(video, track_type, dry_run)
        elif process_type == 'h5':
            # Only convert to h5
            success = convert_to_h5(video, track_type, dry_run)
        else:
            print(f"Error: Unknown process type: {process_type}")
            success = False
        
        if success:
            success_count += 1
        else:
            fail_count += 1
            if not dry_run:
                print(f"⚠️  Failed to process {video.name}, continuing with next video...")
    
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"Successful: {success_count}/{total}")
    print(f"Failed: {fail_count}/{total}")
    
    return fail_count == 0


def get_directories_from_yaml(yaml_file: Path) -> List[Path]:
    """Read experiment directories from a YAML file."""
    if not yaml_file.exists():
        print(f"Error: YAML file not found: {yaml_file}")
        return []
    
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    
    if 'directories' not in data:
        print("Error: YAML file must contain 'directories' key")
        return []
    
    directories = [Path(d) for d in data['directories']]
    print(f"Loaded {len(directories)} experiments from YAML file")
    return directories


def main():
    parser = argparse.ArgumentParser(
        description="Batch track ball and fly in F1 videos using SLEAP"
    )
    parser.add_argument(
        "experiments",
        nargs="*",
        help="Experiment names or patterns to process (searches in datafolder)"
    )
    parser.add_argument(
        "--yaml", "-y",
        type=str,
        help="Path to YAML file containing list of experiment directories"
    )
    parser.add_argument(
        "--datafolder", "-d",
        type=str,
        default=str(DATAFOLDER),
        help=f"Data folder to search for experiments (default: {DATAFOLDER})"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be processed without actually processing"
    )
    parser.add_argument(
        "--check", "-c",
        action="store_true",
        help="Check processing status and exit"
    )
    parser.add_argument(
        "--check-and-process", "-cp",
        action="store_true",
        help="Check status then process unfinished items"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed scanning information"
    )
    
    args = parser.parse_args()
    
    # Determine which directories to process
    directories = []
    
    if args.yaml:
        # Load from YAML file
        directories = get_directories_from_yaml(Path(args.yaml))
    elif args.experiments:
        # Search for experiments in datafolder
        datafolder = Path(args.datafolder)
        all_subdirs = [d for d in datafolder.rglob("*") if d.is_dir()]
        
        # Filter by provided patterns
        for pattern in args.experiments:
            for subdir in all_subdirs:
                if pattern in str(subdir):
                    directories.append(subdir)
        
        directories = list(set(directories))  # Remove duplicates
        print(f"Found {len(directories)} matching directories")
    else:
        # Process all _Checked directories in datafolder
        datafolder = Path(args.datafolder)
        directories = [d for d in datafolder.iterdir() if d.is_dir() and "_Checked" in d.name]
        print(f"Found {len(directories)} _Checked directories in {datafolder}")
    
    if not directories:
        print("No directories to process")
        sys.exit(1)
    
    # Scan all directories
    status = TrackingStatus()
    
    print("\nScanning directories...")
    for directory in sorted(directories):
        scan_directory(directory, status, args.verbose)
    
    # Print summary
    status.print_summary(check_mode=args.check or args.check_and_process)
    
    # Process based on mode
    if args.check:
        # Just check, don't process
        sys.exit(0)
    elif args.check_and_process:
        # Check then process
        if len(status.videos_to_process) > 0:
            success = process_videos(status, args.dry_run)
            sys.exit(0 if success else 1)
        else:
            print("\n🎉 NOTHING TO PROCESS - ALL VIDEOS ARE ALREADY COMPLETE!")
            sys.exit(0)
    elif args.dry_run:
        # Dry run already printed in scan
        print("\n=== END DRY RUN ===")
        sys.exit(0)
    else:
        # Default: process everything
        success = process_videos(status, args.dry_run)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
