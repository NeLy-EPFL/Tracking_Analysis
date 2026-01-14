#!/usr/bin/env python3
"""
Test recombining all arena pairs in an experiment with a specific pixel offset.
Creates before/after grid visualizations.
"""

from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import argparse
import numpy as np
import sys


def extract_frame_from_video(video_path, frame_number=0):
    """Extract a specific frame from a video file."""
    if not video_path.exists():
        return None
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    if len(frame.shape) > 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    return frame


def recombine_frames(left_frame, right_frame, pixels_to_move):
    """
    Recombine left and right frames by moving pixels from right to left.
    
    Returns:
        new_left, new_right: The recombined frames
    """
    # Rotate right frame back to original orientation
    right_frame_unrotated = cv2.rotate(right_frame, cv2.ROTATE_180)
    
    # Extract the strip from the left side of the unrotated right frame
    strip_to_add = right_frame_unrotated[:, :pixels_to_move]
    
    # Remove this strip from the right frame
    right_frame_corrected_unrotated = right_frame_unrotated[:, pixels_to_move:]
    right_frame_corrected = cv2.rotate(right_frame_corrected_unrotated, cv2.ROTATE_180)
    
    # Add the strip to the right side of the left frame
    left_frame_corrected = np.hstack([left_frame, strip_to_add])
    
    return left_frame_corrected, right_frame_corrected


def test_all_arenas(experiment_folder, pixels_to_move, frame_number=0, output_path=None):
    """
    Test recombining all arenas in an experiment folder.
    
    Args:
        experiment_folder: Path to experiment folder
        pixels_to_move: Number of pixels to move from Right to Left
        frame_number: Which frame to extract
        output_path: Path to save the visualization
    """
    experiment_folder = Path(experiment_folder)
    
    print(f"Testing all arenas with {pixels_to_move} pixels moved from Right to Left")
    print(f"Extracting frame {frame_number} from each video\n")
    
    # Collect all frames
    original_lefts = []
    original_rights = []
    corrected_lefts = []
    corrected_rights = []
    arena_numbers = []
    
    for arena_num in range(1, 10):
        arena_folder = experiment_folder / f"arena{arena_num}"
        
        if not arena_folder.exists():
            print(f"Warning: arena{arena_num} folder not found, skipping")
            continue
        
        # Find Left video
        left_folder = arena_folder / "Left"
        left_video = None
        if left_folder.exists():
            video_files = list(left_folder.glob("*.mp4")) + list(left_folder.glob("*.avi"))
            if video_files:
                left_video = video_files[0]
        
        # Find Right video
        right_folder = arena_folder / "Right"
        right_video = None
        if right_folder.exists():
            video_files = list(right_folder.glob("*.mp4")) + list(right_folder.glob("*.avi"))
            if video_files:
                right_video = video_files[0]
        
        if left_video is None or right_video is None:
            print(f"Warning: Missing video for arena{arena_num}, skipping")
            continue
        
        # Extract frames
        left_frame = extract_frame_from_video(left_video, frame_number)
        right_frame = extract_frame_from_video(right_video, frame_number)
        
        if left_frame is None or right_frame is None:
            print(f"Warning: Could not extract frames for arena{arena_num}, skipping")
            continue
        
        print(f"Arena {arena_num}: {left_frame.shape[1]}x{left_frame.shape[0]} → ", end="")
        
        # Recombine
        left_corrected, right_corrected = recombine_frames(left_frame, right_frame, pixels_to_move)
        
        print(f"{left_corrected.shape[1]}x{left_corrected.shape[0]} (Left), "
              f"{right_corrected.shape[1]}x{right_corrected.shape[0]} (Right)")
        
        original_lefts.append(left_frame)
        original_rights.append(right_frame)
        corrected_lefts.append(left_corrected)
        corrected_rights.append(right_corrected)
        arena_numbers.append(arena_num)
    
    if not arena_numbers:
        print("Error: No valid arena pairs found")
        return False
    
    # Create visualization
    # We'll create a 4-row grid:
    # Row 1: Original Left frames
    # Row 2: Original Right frames  
    # Row 3: Corrected Left frames
    # Row 4: Corrected Right frames
    
    num_arenas = len(arena_numbers)
    fig, axes = plt.subplots(4, num_arenas, figsize=(3*num_arenas, 12))
    
    # Handle case where we only have one arena
    if num_arenas == 1:
        axes = axes.reshape(4, 1)
    
    for i, arena_num in enumerate(arena_numbers):
        # Row 0: Original Left
        axes[0, i].imshow(original_lefts[i], cmap='gray', vmin=0, vmax=255)
        axes[0, i].set_title(f'Arena {arena_num}\nOriginal Left')
        axes[0, i].axis('off')
        
        # Row 1: Original Right
        axes[1, i].imshow(original_rights[i], cmap='gray', vmin=0, vmax=255)
        axes[1, i].set_title('Original Right')
        axes[1, i].axis('off')
        
        # Row 2: Corrected Left
        axes[2, i].imshow(corrected_lefts[i], cmap='gray', vmin=0, vmax=255)
        axes[2, i].set_title(f'Corrected Left\n(+{pixels_to_move}px)')
        axes[2, i].axis('off')
        
        # Row 3: Corrected Right
        axes[3, i].imshow(corrected_rights[i], cmap='gray', vmin=0, vmax=255)
        axes[3, i].set_title(f'Corrected Right\n(-{pixels_to_move}px)')
        axes[3, i].axis('off')
    
    plt.suptitle(f'Recombination Test: {pixels_to_move} pixels moved from Right to Left', 
                 fontsize=16, y=0.995)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_path}")
    
    plt.show()
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test recombining all arena pairs in an experiment"
    )
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        required=True,
        help="Path to experiment folder"
    )
    parser.add_argument(
        "--pixels", "-p",
        type=int,
        required=True,
        help="Number of pixels to move from Right to Left"
    )
    parser.add_argument(
        "--frame", "-f",
        type=int,
        default=0,
        help="Frame number to test on (default: 0)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Path to save the visualization image"
    )
    
    args = parser.parse_args()
    
    success = test_all_arenas(args.experiment, args.pixels, args.frame, args.output)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
