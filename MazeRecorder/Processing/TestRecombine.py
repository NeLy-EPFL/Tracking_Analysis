#!/usr/bin/env python3
"""
Interactive script to test recombining Left and Right videos by transferring pixels.
This helps determine the correct number of pixels to move from Right video to Left video.
"""

from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import argparse
import sys
import numpy as np


def extract_frame_from_video(video_path, frame_number=0):
    """Extract a specific frame from a video file."""
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return None
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return None
    
    # Set the frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read the frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Error: Could not read frame {frame_number} from {video_path}")
        return None
    
    # Convert to grayscale if needed
    if len(frame.shape) > 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    return frame


def test_recombine(left_video, right_video, pixels_to_move, frame_number=0, save_path=None):
    """
    Test recombining Left and Right videos by moving pixels from Right to Left.
    
    Args:
        left_video: Path to Left video
        right_video: Path to Right video
        pixels_to_move: Number of pixel columns to move from Right to Left
        frame_number: Which frame to test on
        save_path: Optional path to save the result image
    
    Returns:
        True if successful, False otherwise
    """
    # Extract frames
    left_frame = extract_frame_from_video(left_video, frame_number)
    right_frame = extract_frame_from_video(right_video, frame_number)
    
    if left_frame is None or right_frame is None:
        return False
    
    print(f"\nOriginal dimensions:")
    print(f"  Left:  {left_frame.shape[1]}x{left_frame.shape[0]} (width x height)")
    print(f"  Right: {right_frame.shape[1]}x{right_frame.shape[0]} (width x height)")
    
    if pixels_to_move < 0:
        print("Error: pixels_to_move must be >= 0")
        return False
    
    if pixels_to_move >= right_frame.shape[1]:
        print(f"Error: pixels_to_move ({pixels_to_move}) must be less than right video width ({right_frame.shape[1]})")
        return False
    
    # Extract the strip from the left side of the Right video (before rotation)
    # Remember: the Right video was rotated 180 degrees during processing
    # So we need to think about this carefully
    
    # The Right frame as we see it in the video is already rotated 180
    # The missing pixels from Left are actually on the RIGHT side of the rotated Right video
    # So we need to:
    # 1. Take the rightmost pixels from the Right video
    # 2. Rotate them 180 degrees
    # 3. Append them to the right side of the Left video
    
    # OR equivalently:
    # 1. Rotate the Right video back 180 degrees to original orientation
    # 2. Take the leftmost pixels (which were the original right edge of the arena)
    # 3. Append them to the right side of the Left video
    
    # Let's use the second approach as it's clearer
    right_frame_unrotated = cv2.rotate(right_frame, cv2.ROTATE_180)
    
    # Extract the strip from the left side of the unrotated right frame
    strip_to_add = right_frame_unrotated[:, :pixels_to_move]
    
    # Remove this strip from the right frame and rotate back
    right_frame_corrected_unrotated = right_frame_unrotated[:, pixels_to_move:]
    right_frame_corrected = cv2.rotate(right_frame_corrected_unrotated, cv2.ROTATE_180)
    
    # Add the strip to the right side of the left frame
    left_frame_corrected = np.hstack([left_frame, strip_to_add])
    
    print(f"\nAfter moving {pixels_to_move} pixels from Right to Left:")
    print(f"  New Left:  {left_frame_corrected.shape[1]}x{left_frame_corrected.shape[0]}")
    print(f"  New Right: {right_frame_corrected.shape[1]}x{right_frame_corrected.shape[0]}")
    
    # Create visualization
    fig, axes = plt.subplots(3, 2, figsize=(15, 20))
    
    # Row 1: Original frames
    axes[0, 0].imshow(left_frame, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('Original Left')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(right_frame, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title('Original Right (rotated 180°)')
    axes[0, 1].axis('off')
    
    # Row 2: What we're doing
    axes[1, 0].imshow(right_frame_unrotated, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].axvline(x=pixels_to_move, color='red', linewidth=2, linestyle='--')
    axes[1, 0].set_title(f'Right (unrotated) - Red line shows {pixels_to_move}px to move')
    axes[1, 0].axis('off')
    
    # Show the strip
    if pixels_to_move > 0:
        axes[1, 1].imshow(strip_to_add, cmap='gray', vmin=0, vmax=255)
        axes[1, 1].set_title(f'Strip to add ({pixels_to_move}px wide)')
        axes[1, 1].axis('off')
    else:
        axes[1, 1].text(0.5, 0.5, 'No strip\n(0 pixels)', ha='center', va='center',
                       transform=axes[1, 1].transAxes, fontsize=16)
        axes[1, 1].axis('off')
    
    # Row 3: Corrected frames
    axes[2, 0].imshow(left_frame_corrected, cmap='gray', vmin=0, vmax=255)
    axes[2, 0].set_title(f'Corrected Left (+{pixels_to_move}px)')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(right_frame_corrected, cmap='gray', vmin=0, vmax=255)
    axes[2, 1].set_title(f'Corrected Right (-{pixels_to_move}px)')
    axes[2, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
    
    plt.show()
    
    return True


def find_video_pair(experiment_folder, arena_num=1):
    """Find Left and Right video files for a specific arena."""
    experiment_folder = Path(experiment_folder)
    arena_folder = experiment_folder / f"arena{arena_num}"
    
    if not arena_folder.exists():
        print(f"Error: Arena folder not found: {arena_folder}")
        return None, None
    
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
        print(f"Error: Could not find both Left and Right videos in arena{arena_num}")
        return None, None
    
    return left_video, right_video


def main():
    parser = argparse.ArgumentParser(
        description="Test recombining Left and Right videos by moving pixels"
    )
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        required=True,
        help="Path to experiment folder"
    )
    parser.add_argument(
        "--arena", "-a",
        type=int,
        default=1,
        help="Arena number to test (1-9, default: 1)"
    )
    parser.add_argument(
        "--pixels", "-p",
        type=int,
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
    
    # Find video pair
    left_video, right_video = find_video_pair(args.experiment, args.arena)
    
    if left_video is None or right_video is None:
        sys.exit(1)
    
    print(f"Testing arena {args.arena}:")
    print(f"  Left:  {left_video}")
    print(f"  Right: {right_video}")
    
    if args.pixels is not None:
        # Single test mode
        test_recombine(left_video, right_video, args.pixels, args.frame, args.output)
    else:
        # Interactive mode - try different values
        print("\nInteractive mode: Try different pixel values")
        print("Enter number of pixels to move (or 'q' to quit)")
        
        while True:
            try:
                user_input = input("\nPixels to move: ").strip()
                if user_input.lower() == 'q':
                    break
                
                pixels = int(user_input)
                test_recombine(left_video, right_video, pixels, args.frame, args.output)
                
            except ValueError:
                print("Please enter a valid number or 'q' to quit")
            except KeyboardInterrupt:
                print("\nExiting...")
                break


if __name__ == "__main__":
    main()
