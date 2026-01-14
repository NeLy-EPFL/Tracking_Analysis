#!/usr/bin/env python3
"""
Script to verify already processed experiments by extracting frames from videos
and creating a grid visualization to check for crop issues.
"""

from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import argparse
import sys
import os
import subprocess


# Path definitions
datafolder = Path("/home/matthias/Videos/")


def extract_frame_from_video(video_path, frame_number=0):
    """
    Extract a specific frame from a video file.
    
    Args:
        video_path: Path to the video file
        frame_number: Which frame to extract (default: 0 for first frame)
    
    Returns:
        frame: The extracted frame as a numpy array, or None if extraction failed
    """
    if not video_path.exists():
        print(f"Warning: Video file not found: {video_path}")
        return None
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Warning: Could not open video: {video_path}")
        return None
    
    # Set the frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read the frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Warning: Could not read frame {frame_number} from {video_path}")
        return None
    
    # Convert to grayscale if needed
    if len(frame.shape) > 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    return frame


def verify_processed_folder(folder_path, output_folder=None, frame_number=0):
    """
    Verify a processed experiment folder by extracting frames from videos
    and creating a grid visualization.
    
    Args:
        folder_path: Path to the _Cropped folder
        output_folder: Where to save the verification image (defaults to same folder)
        frame_number: Which frame to extract from each video (default: 0)
    """
    folder = Path(folder_path)
    
    if not folder.exists() or not folder.is_dir():
        print(f"Error: Folder {folder} does not exist or is not a directory.")
        return False
    
    print(f"Verifying processed folder: {folder.name}")
    print(f"Extracting frame {frame_number} from each video...")
    
    # Set output folder
    if output_folder is None:
        output_folder = folder
    else:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
    
    # Create visualization (3 rows x 9 columns: original arena + left + right for each)
    fig, axs = plt.subplots(3, 9, figsize=(30, 10))
    
    # Track if we found any issues
    missing_videos = []
    
    for i in range(9):
        arena_num = i + 1
        arena_folder = folder / f"arena{arena_num}"
        
        # Initialize placeholders
        left_frame = None
        right_frame = None
        
        # Check if arena folder exists
        if not arena_folder.exists():
            print(f"Warning: Arena folder not found: {arena_folder}")
            missing_videos.append(f"arena{arena_num}/")
        else:
            # Look for Left video
            left_folder = arena_folder / "Left"
            if left_folder.exists():
                # Find video file (could be .mp4, .avi, etc.)
                video_files = list(left_folder.glob("*.mp4")) + list(left_folder.glob("*.avi"))
                if video_files:
                    left_video = video_files[0]
                    left_frame = extract_frame_from_video(left_video, frame_number)
                else:
                    print(f"Warning: No video found in {left_folder}")
                    missing_videos.append(f"arena{arena_num}/Left")
            else:
                print(f"Warning: Left folder not found: {left_folder}")
                missing_videos.append(f"arena{arena_num}/Left")
            
            # Look for Right video
            right_folder = arena_folder / "Right"
            if right_folder.exists():
                # Find video file
                video_files = list(right_folder.glob("*.mp4")) + list(right_folder.glob("*.avi"))
                if video_files:
                    right_video = video_files[0]
                    right_frame = extract_frame_from_video(right_video, frame_number)
                else:
                    print(f"Warning: No video found in {right_folder}")
                    missing_videos.append(f"arena{arena_num}/Right")
            else:
                print(f"Warning: Right folder not found: {right_folder}")
                missing_videos.append(f"arena{arena_num}/Right")
        
        # Create a combined view in the first row (if both frames exist)
        if left_frame is not None and right_frame is not None:
            # Create a side-by-side view
            # Note: right frame was rotated 180 in processing, so we rotate it back for display
            right_frame_display = cv2.rotate(right_frame, cv2.ROTATE_180)
            combined = cv2.hconcat([left_frame, right_frame_display])
            axs[0, i].imshow(combined, cmap="gray", vmin=0, vmax=255)
        elif left_frame is not None:
            axs[0, i].imshow(left_frame, cmap="gray", vmin=0, vmax=255)
        elif right_frame is not None:
            right_frame_display = cv2.rotate(right_frame, cv2.ROTATE_180)
            axs[0, i].imshow(right_frame_display, cmap="gray", vmin=0, vmax=255)
        else:
            # Show blank/error
            axs[0, i].text(0.5, 0.5, 'MISSING', ha='center', va='center', 
                          transform=axs[0, i].transAxes, fontsize=12, color='red')
        
        axs[0, i].set_title(f"Arena {arena_num}")
        axs[0, i].axis("off")
        
        # Show left half
        if left_frame is not None:
            axs[1, i].imshow(left_frame, cmap="gray", vmin=0, vmax=255)
        else:
            axs[1, i].text(0.5, 0.5, 'MISSING', ha='center', va='center',
                          transform=axs[1, i].transAxes, fontsize=10, color='red')
        axs[1, i].set_title(f"Left {arena_num}")
        axs[1, i].axis("off")
        
        # Show right half
        if right_frame is not None:
            axs[2, i].imshow(right_frame, cmap="gray", vmin=0, vmax=255)
        else:
            axs[2, i].text(0.5, 0.5, 'MISSING', ha='center', va='center',
                          transform=axs[2, i].transAxes, fontsize=10, color='red')
        axs[2, i].set_title(f"Right {arena_num}")
        axs[2, i].axis("off")
    
    plt.tight_layout()
    output_file = output_folder / "verification_check.png"
    plt.savefig(str(output_file), dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\nVerification image saved to: {output_file}")
    
    if missing_videos:
        print(f"\nWarning: {len(missing_videos)} video(s) missing or unreadable:")
        for vid in missing_videos:
            print(f"  - {vid}")
        return False
    else:
        print("\nAll videos found and verified successfully!")
        return True


def find_processed_folders(data_folder):
    """Find all folders that contain processed arenas (ending with _Cropped, _Videos_Checked, etc.)."""
    data_folder = Path(data_folder)
    processed_folders = []
    
    # Look for folders with arena subfolders structure
    for folder in data_folder.iterdir():
        if folder.is_dir():
            # Check if it has arena1, arena2, etc. subfolders
            has_arenas = any((folder / f"arena{i}").exists() for i in range(1, 10))
            if has_arenas:
                processed_folders.append(folder)
    
    return sorted(processed_folders)


def batch_verify(data_folder, output_base_folder=None, frame_number=0):
    """
    Verify all processed experiments in a data folder.
    
    Args:
        data_folder: Path to the folder containing processed experiments
        output_base_folder: Base folder for verification images (defaults to each experiment folder)
        frame_number: Which frame to extract from each video
    """
    processed_folders = find_processed_folders(data_folder)
    
    if not processed_folders:
        print(f"No processed folders (with arena1-9 structure) found in {data_folder}")
        return
    
    print(f"Found {len(processed_folders)} processed folders to verify:")
    for folder in processed_folders:
        print(f"  - {folder.name}")
    
    if output_base_folder:
        output_base = Path(output_base_folder)
        output_base.mkdir(parents=True, exist_ok=True)
    else:
        output_base = None
    
    print("\nStarting verification...")
    results = []
    
    for folder in processed_folders:
        print(f"\n{'='*80}")
        
        if output_base:
            output_folder = output_base / folder.name
        else:
            output_folder = None
        
        success = verify_processed_folder(folder, output_folder, frame_number)
        results.append((folder.name, success))
    
    # Summary
    print(f"\n{'='*80}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*80}")
    
    successful = sum(1 for _, s in results if s)
    failed = len(results) - successful
    
    print(f"Total processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed experiments:")
        for name, success in results:
            if not success:
                print(f"  - {name}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify already processed experiments by checking video crops"
    )
    parser.add_argument(
        "--folder", "-f", 
        type=str, 
        help="Specific _Cropped folder to verify"
    )
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Verify all processed folders in the data directory"
    )
    parser.add_argument(
        "--data-folder", "-d",
        type=str,
        default=str(datafolder),
        help=f"Data folder to search for processed experiments (default: {datafolder})"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output folder for verification images (default: same as input folder)"
    )
    parser.add_argument(
        "--frame", "-n",
        type=int,
        default=0,
        help="Frame number to extract from videos (default: 0)"
    )
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch mode: verify all processed folders
        batch_verify(args.data_folder, args.output, args.frame)
    elif args.folder:
        # Single folder mode
        verify_processed_folder(args.folder, args.output, args.frame)
    else:
        # Interactive mode: show available folders
        processed_folders = find_processed_folders(args.data_folder)
        
        if not processed_folders:
            print(f"No processed folders (with arena1-9 structure) found in {args.data_folder}")
            sys.exit(1)
        
        print("Available processed folders:")
        for i, folder in enumerate(processed_folders):
            print(f"{i+1}: {folder.name}")
        
        if os.isatty(sys.stdin.fileno()):
            choice = input(f"\nChoose a folder to verify (1-{len(processed_folders)}, or 'all' for batch): ")
            
            if choice.lower() == 'all':
                batch_verify(args.data_folder, args.output, args.frame)
            else:
                try:
                    folder_index = int(choice) - 1
                    if 0 <= folder_index < len(processed_folders):
                        verify_processed_folder(processed_folders[folder_index], args.output, args.frame)
                    else:
                        print("Invalid choice.")
                except ValueError:
                    print("Please enter a valid number or 'all'.")
        else:
            # Non-interactive mode
            print("Non-interactive mode: use --folder or --batch option")
            sys.exit(1)


if __name__ == "__main__":
    main()
