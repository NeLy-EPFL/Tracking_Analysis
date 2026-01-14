#!/usr/bin/env python3
"""
Recombine Left and Right videos by transferring pixels from Right to Left using ffmpeg.
This script efficiently processes all arena pairs in an experiment.
"""


from pathlib import Path
import subprocess
import argparse
import sys
import shutil
import json
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import re
import threading
import tempfile
import time



def get_video_info(video_path):
    """Get video information using ffprobe."""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-count_packets',
        '-show_entries', 'stream=width,height,nb_read_packets,r_frame_rate',
        '-of', 'json',
        str(video_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error getting video info: {result.stderr}")
        return None
    
    info = json.loads(result.stdout)
    if 'streams' not in info or len(info['streams']) == 0:
        return None
    
    stream = info['streams'][0]
    return {
        'width': int(stream['width']),
        'height': int(stream['height']),
        'frame_rate': stream.get('r_frame_rate', '30'),
        'nb_frames': int(stream.get('nb_read_packets', 0))
    }



def monitor_progress(progress_file, total_frames, video_name, pbar=None):
    """Monitor ffmpeg progress file and update progress bar."""
    last_frame = 0
    while True:
        try:
            if not Path(progress_file).exists():
                time.sleep(0.1)
                continue
                
            with open(progress_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith('frame='):
                        try:
                            frame = int(line.split('=')[1].strip())
                            if frame > last_frame:
                                if pbar:
                                    pbar.update(frame - last_frame)
                                last_frame = frame
                        except:
                            pass
                    elif line.startswith('progress='):
                        status = line.split('=')[1].strip()
                        if status == 'end':
                            if pbar:
                                pbar.update(total_frames - last_frame)
                            return True
        except:
            pass
        time.sleep(0.2)



def recombine_video_pair(left_video, right_video, output_left, output_right, pixels_to_move, duration=None, use_cuda=True, show_progress=True):
    """
    Recombine a Left/Right video pair using ffmpeg.
    
    Args:
        left_video: Path to original Left video
        right_video: Path to original Right video  
        output_left: Path for corrected Left video
        output_right: Path for corrected Right video
        pixels_to_move: Number of pixels to move from Right to Left
        duration: Optional duration in seconds to process (for testing)
        use_cuda: Whether to use CUDA hardware acceleration (default: True)
        show_progress: Whether to show progress bars (default: True)
    
    Returns:
        True if successful, False otherwise
    """
    # Get video info
    right_info = get_video_info(right_video)
    if right_info is None:
        print(f"Error: Could not get info for {right_video}")
        return False
    
    right_width = right_info['width']
    
    if pixels_to_move >= right_width:
        print(f"Error: pixels_to_move ({pixels_to_move}) >= right video width ({right_width})")
        return False
    
    # Create output directories
    output_left.parent.mkdir(parents=True, exist_ok=True)
    output_right.parent.mkdir(parents=True, exist_ok=True)
    
    # Get total frames for progress tracking
    total_frames = right_info.get('nb_frames', 0)
    if total_frames == 0 and duration is not None:
        # Estimate frames from duration and frame rate
        try:
            fps_str = right_info.get('frame_rate', '30')
            if '/' in fps_str:
                num, den = fps_str.split('/')
                fps_val = float(num) / float(den)
            else:
                fps_val = float(fps_str)
            total_frames = int(duration * fps_val)
        except:
            total_frames = 0
    
    # Create temporary progress files
    progress_left = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='_left.log')
    progress_right = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='_right.log')
    progress_left_path = progress_left.name
    progress_right_path = progress_right.name
    progress_left.close()
    progress_right.close()
    
    # Build base ffmpeg commands
    base_inputs = ['-y', '-hide_banner', '-loglevel', 'error', '-stats']
    
    # Add CUDA hardware acceleration if requested
    hwaccel_args = []
    if use_cuda:
        hwaccel_args = ['-hwaccel', 'cuda']
    
    # Add duration limit if specified
    duration_args = []
    if duration is not None:
        duration_args = ['-t', str(duration)]
    
    # OPTIMIZED: Use libx264 instead of libx265 for much faster encoding
    # CRF 18 with libx264 is visually lossless and 10-20x faster than libx265 CRF 15
    encoding_args = [
        '-c:v', 'libx264',
        '-preset', 'veryfast',  # Much faster, negligible quality difference
        '-crf', '18',  # Visually lossless for tracking
        '-pix_fmt', 'yuv420p'
    ]
    
    # For true lossless, use: ['-c:v', 'libx264', '-preset', 'medium', '-qp', '0']
    
    # Process Left video (add strip from Right)
    left_cmd = [
        'ffmpeg',
        *hwaccel_args,
        *duration_args,
        '-i', str(left_video),
        '-i', str(right_video),
        '-filter_complex',
        f'[1:v]crop={pixels_to_move}:ih:{right_width-pixels_to_move}:0,transpose=2,transpose=2[strip];'
        f'[0:v][strip]hstack=inputs=2[out]',
        '-map', '[out]',
        *base_inputs,
        '-progress', progress_left_path,
        *encoding_args,
        str(output_left)
    ]
    
    # Process Right video (remove strip from right side)
    right_cmd = [
        'ffmpeg',
        *hwaccel_args,
        *duration_args,
        '-i', str(right_video),
        '-filter_complex',
        f'[0:v]crop={right_width-pixels_to_move}:ih:0:0[out]',
        '-map', '[out]',
        *base_inputs,
        '-progress', progress_right_path,
        *encoding_args,
        str(output_right)
    ]
    
    # Run both ffmpeg commands in parallel
    try:
        accel_method = "CUDA" if use_cuda else "CPU"
        print(f"    Encoding both videos in parallel ({accel_method})...")
        
        # Start both processes
        process_left = subprocess.Popen(
            left_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        process_right = subprocess.Popen(
            right_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Monitor progress if enabled
        if show_progress and total_frames > 0:
            # Create progress bars
            pbar_left = tqdm(total=total_frames, desc="      Left", unit="frames", leave=False)
            pbar_right = tqdm(total=total_frames, desc="      Right", unit="frames", leave=False)
            
            # Start monitoring threads
            monitor_thread_left = threading.Thread(
                target=monitor_progress,
                args=(progress_left_path, total_frames, "Left", pbar_left)
            )
            monitor_thread_right = threading.Thread(
                target=monitor_progress,
                args=(progress_right_path, total_frames, "Right", pbar_right)
            )
            
            monitor_thread_left.start()
            monitor_thread_right.start()
            
            # Wait for processes to complete
            process_left.wait()
            process_right.wait()
            
            # Wait for monitoring to finish
            monitor_thread_left.join(timeout=2)
            monitor_thread_right.join(timeout=2)
            
            # Close progress bars
            pbar_left.close()
            pbar_right.close()
        else:
            # Just wait for completion
            process_left.wait()
            process_right.wait()
        
        # Clean up progress files
        try:
            Path(progress_left_path).unlink()
            Path(progress_right_path).unlink()
        except:
            pass
        
        # Check for errors
        if process_left.returncode != 0:
            # If CUDA failed, try CPU fallback
            if use_cuda:
                print(f"    CUDA encoding failed for Left, retrying with CPU...")
                # Clean up partial output
                if output_left.exists():
                    output_left.unlink()
                if output_right.exists():
                    output_right.unlink()
                return recombine_video_pair(left_video, right_video, output_left, output_right, 
                                           pixels_to_move, duration, use_cuda=False, show_progress=show_progress)
            else:
                stderr_output = process_left.stderr.read() if process_left.stderr else "Unknown error"
                print(f"Error processing Left video: return code {process_left.returncode}")
                print(f"Error output: {stderr_output[:500]}")
                return False
        
        if process_right.returncode != 0:
            # If CUDA failed, try CPU fallback
            if use_cuda:
                print(f"    CUDA encoding failed for Right, retrying with CPU...")
                # Clean up partial outputs
                if output_left.exists():
                    output_left.unlink()
                if output_right.exists():
                    output_right.unlink()
                return recombine_video_pair(left_video, right_video, output_left, output_right, 
                                           pixels_to_move, duration, use_cuda=False, show_progress=show_progress)
            else:
                stderr_output = process_right.stderr.read() if process_right.stderr else "Unknown error"
                print(f"Error processing Right video: return code {process_right.returncode}")
                print(f"Error output: {stderr_output[:500]}")
                return False
        
        print(f"    ✓ Both videos encoded successfully")
        return True
        
    except subprocess.TimeoutExpired as e:
        print(f"Error: ffmpeg timed out after 1 hour")
        return False
    except Exception as e:
        print(f"Error running ffmpeg: {e}")
        import traceback
        traceback.print_exc()
        return False



def process_arena(arena_num, experiment_folder, output_folder, pixels_to_move, duration=None, use_temp=False, use_cuda=True, show_progress=True):
    """Process a single arena pair."""
    print(f"\n{'='*60}")
    print(f"Processing Arena {arena_num}")
    print(f"{'='*60}")
    
    arena_folder = experiment_folder / f"arena{arena_num}"
    
    if not arena_folder.exists():
        return arena_num, False, "Arena folder not found"
    
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
        return arena_num, False, "Missing video files"
    
    print(f"  Input Left:  {left_video.name}")
    print(f"  Input Right: {right_video.name}")
    
    # Set output paths
    output_left = output_folder / f"arena{arena_num}" / "Left" / left_video.name
    output_right = output_folder / f"arena{arena_num}" / "Right" / right_video.name
    
    print(f"  Output Left:  {output_left.relative_to(output_folder)}")
    print(f"  Output Right: {output_right.relative_to(output_folder)}")
    
    # If using temp storage, copy videos locally first
    temp_dir = None
    original_left = left_video
    original_right = right_video
    
    if use_temp:
        print(f"  Copying videos to local temp storage...")
        temp_dir = Path(tempfile.mkdtemp(prefix=f"recombine_arena{arena_num}_"))
        
        # Copy videos to temp
        temp_left = temp_dir / "left_input.mp4"
        temp_right = temp_dir / "right_input.mp4"
        
        shutil.copy2(left_video, temp_left)
        shutil.copy2(right_video, temp_right)
        
        # Use temp paths for processing
        left_video = temp_left
        right_video = temp_right
        
        # Output to temp as well
        temp_output_left = temp_dir / "left_output.mp4"
        temp_output_right = temp_dir / "right_output.mp4"
        original_output_left = output_left
        original_output_right = output_right
        output_left = temp_output_left
        output_right = temp_output_right
    
    # Recombine videos
    success = recombine_video_pair(left_video, right_video, output_left, output_right, pixels_to_move, duration, use_cuda, show_progress)
    
    # If using temp, copy results back
    if use_temp and success:
        print(f"  Copying results back to network storage...")
        original_output_left.parent.mkdir(parents=True, exist_ok=True)
        original_output_right.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_left, original_output_left)
        shutil.copy2(output_right, original_output_right)
    
    # Clean up temp directory
    if temp_dir and temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    if success:
        print(f"✓ Arena {arena_num} completed successfully")
        return arena_num, True, "Success"
    else:
        print(f"✗ Arena {arena_num} failed")
        return arena_num, False, "ffmpeg processing failed"



def recombine_experiment(experiment_folder, pixels_to_move=15, output_folder=None, num_workers=None, test_mode=False, test_arena=1, test_duration=10, use_temp=False, use_cuda=True, show_progress=True):
    """
    Recombine all arena pairs in an experiment folder.
    
    Args:
        experiment_folder: Path to experiment folder
        pixels_to_move: Number of pixels to move from Right to Left (default: 15)
        output_folder: Custom output folder path (default: auto-generated)
        num_workers: Number of parallel workers (default: number of CPUs)
        test_mode: If True, only process one arena for a limited duration
        test_arena: Arena number to test (1-9, only used in test mode)
        test_duration: Duration in seconds to process in test mode (default: 10)
        use_temp: If True, copy videos to local temp storage before processing (faster for network storage)
    """
    experiment_folder = Path(experiment_folder)
    
    if not experiment_folder.exists():
        print(f"Error: Experiment folder not found: {experiment_folder}")
        return False
    
    # Determine output folder
    if output_folder is None:
        # Generate output folder name
        folder_name = experiment_folder.name
        if test_mode:
            # Add _Test prefix for test mode
            if "_Videos_Checked" in folder_name:
                new_name = folder_name.replace("_Videos_Checked", "_Test_Recombined_Videos_Checked")
            elif "_Cropped" in folder_name:
                new_name = folder_name.replace("_Cropped", "_Test_Recombined_Cropped")
            else:
                new_name = folder_name + "_Test_Recombined"
        else:
            if "_Videos_Checked" in folder_name:
                new_name = folder_name.replace("_Videos_Checked", "_Recombined_Videos_Checked")
            elif "_Cropped" in folder_name:
                new_name = folder_name.replace("_Cropped", "_Recombined_Cropped")
            else:
                new_name = folder_name + "_Recombined"
        
        output_folder = experiment_folder.parent / new_name
    else:
        output_folder = Path(output_folder)
    
    if test_mode:
        print("="*60)
        print("TEST MODE")
        print("="*60)
        print(f"Processing only Arena {test_arena} for {test_duration} seconds")
    
    if use_temp:
        print("Using local temp storage for faster processing")
    
    print(f"Input:  {experiment_folder}")
    print(f"Output: {output_folder}")
    print(f"Pixels to move: {pixels_to_move}\n")
    
    # Check if output folder exists
    if output_folder.exists():
        if test_mode:
            shutil.rmtree(output_folder)
        else:
            response = input(f"Output folder already exists. Overwrite? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                return False
            shutil.rmtree(output_folder)
    
    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Copy metadata.json if it exists
    metadata_file = experiment_folder / "metadata.json"
    if metadata_file.exists():
        shutil.copy2(metadata_file, output_folder / "metadata.json")
        print("Copied metadata.json")
    
    # Copy duration.npy if it exists
    duration_file = experiment_folder / "duration.npy"
    if duration_file.exists():
        shutil.copy2(duration_file, output_folder / "duration.npy")
        print("Copied duration.npy")
    
    # Copy fps.npy if it exists
    fps_file = experiment_folder / "fps.npy"
    if fps_file.exists():
        shutil.copy2(fps_file, output_folder / "fps.npy")
        print("Copied fps.npy")
    
    print()
    
    if test_mode:
        # Test mode: process only one arena
        print(f"Processing arena {test_arena} (test mode - {test_duration}s)...\n")
        
        arena_num, success, message = process_arena(
            test_arena, experiment_folder, output_folder, pixels_to_move, duration=test_duration, use_temp=use_temp, use_cuda=use_cuda, show_progress=show_progress
        )
        
        if success:
            print(f"\n{'='*60}")
            print(f"Test successful!")
            print(f"Arena {arena_num} processed: {message}")
            print(f"{'='*60}\n")
            print(f"Test videos saved to: {output_folder}/arena{test_arena}/")
            print(f"  Left:  {output_folder}/arena{test_arena}/Left/")
            print(f"  Right: {output_folder}/arena{test_arena}/Right/")
            print("\nPlease review the test videos before processing the full experiment.")
            return True
        else:
            print(f"\n{'='*60}")
            print(f"Test failed!")
            print(f"Arena {arena_num}: {message}")
            print(f"{'='*60}\n")
            return False
    
    # Full processing mode
    # Determine number of workers
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 9)  # Max 9 arenas anyway
    
    print(f"Processing 9 arenas sequentially to show progress...\n")
    
    # Process arenas sequentially to show progress properly
    results = []
    for arena_num in range(1, 10):
        arena_num_result, success, message = process_arena(
            arena_num, experiment_folder, output_folder, pixels_to_move, None, use_temp=use_temp, use_cuda=use_cuda, show_progress=show_progress
        )
        results.append((arena_num_result, success, message))
        if not success:
            print(f"\nWarning: Arena {arena_num} - {message}")
    
    # Summary
    successful = sum(1 for _, s, _ in results if s)
    failed = 9 - successful
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Successful: {successful}/9")
    if failed > 0:
        print(f"Failed: {failed}/9")
        print("\nFailed arenas:")
        for arena_num, success, message in results:
            if not success:
                print(f"  Arena {arena_num}: {message}")
    print(f"{'='*60}\n")
    
    # Generate verification image
    if successful > 0:
        print("Generating verification image...")
        verify_script = Path(__file__).parent / "VerifyProcessedExperiments.py"
        if verify_script.exists():
            try:
                subprocess.run([
                    sys.executable,
                    str(verify_script),
                    '--folder', str(output_folder),
                    '--output', str(output_folder)
                ], check=True)
                print(f"Verification image saved to: {output_folder / 'verification_check.png'}")
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to generate verification image: {e}")
        else:
            print("Note: VerifyProcessedExperiments.py not found, skipping verification image")
    
    print(f"\nRecombined experiment saved to: {output_folder}")
    
    return successful == 9



def main():
    parser = argparse.ArgumentParser(
        description="Recombine Left and Right videos by transferring pixels using ffmpeg"
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
        default=15,
        help="Number of pixels to move from Right to Left (default: 15)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Custom output folder path (default: auto-generated)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        help="Number of parallel workers (default: number of CPUs)"
    )
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="Test mode: process only one arena for a short duration"
    )
    parser.add_argument(
        "--test-arena",
        type=int,
        default=1,
        help="Arena number to test (1-9, default: 1)"
    )
    parser.add_argument(
        "--test-duration",
        type=int,
        default=10,
        help="Duration in seconds for test mode (default: 10)"
    )
    parser.add_argument(
        "--use-temp",
        action="store_true",
        help="Copy videos to local temp storage before processing (much faster for network storage)"
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Disable CUDA hardware acceleration (use CPU only)"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars (more reliable, less output)"
    )
    
    args = parser.parse_args()
    
    success = recombine_experiment(
        args.experiment,
        args.pixels,
        args.output,
        args.workers,
        test_mode=args.test,
        test_arena=args.test_arena,
        test_duration=args.test_duration,
        use_temp=args.use_temp,
        use_cuda=not args.no_cuda,
        show_progress=not args.no_progress
    )
    
    if not success:
        sys.exit(1)



if __name__ == "__main__":
    main()
