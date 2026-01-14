"""
Enhanced Images2Vids Script with Data Protection and Robust Validation

Key improvements:
1. NEVER REMOVES ORIGINAL IMAGES - All processing is done on copies/outputs only
2. CUDA failsafe - Automatically falls back to CPU if CUDA fails
3. Duration validation - Validates video duration against duration.npy data
4. Comprehensive error handling and logging
5. Detailed progress reporting and failure logs
6. Command line options for CPU-only mode and skipping validation

Usage:
    python Images2Vids.py                                  # Normal operation (auto-fix invalid videos by default)
    python Images2Vids.py --dry-run                        # Preview what would be done
    python Images2Vids.py --cpu-only                       # Skip CUDA, use CPU only
    python Images2Vids.py --no-duration-check              # Skip duration validation
    python Images2Vids.py --no-auto-fix-invalid            # Disable auto-fix; prompt before removing/remaking
    python Images2Vids.py --duration-tolerance 2.0         # Set duration tolerance in seconds
"""

from pathlib import Path
from tqdm import tqdm
import subprocess
import os
import sys
import argparse
import numpy as np
from datetime import datetime
from utils_behavior import Utils

data_folder = Path("/home/matthias/Videos/")
# Known output roots to search for the experiment folder. Edit this list to include all
# locations where experiment folders may already be created. The script will pick the
# first path that contains a folder with the same name and a metadata.json file.
OUTPUT_PATHS = [
    Path("/mnt/upramdya_data/MD/Infection_Exps/InfectionCorridors/Experiments"),
    Path("/mnt/upramdya_data/MD/F1_Tracks/Videos"),
    Path("/home/matthias/Videos_output"),
    Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Videos"),
]

# Backwards-compatible single-variable for scripts that reference `output_path`.
# It will be set per-experiment below when a matching folder is found.
output_path = None

# fps = "29"


def check_video_integrity(video_path):
    """Check basic video integrity using ffprobe"""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_format", "-show_streams", video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error checking video integrity: {e.stderr.decode('utf-8')}")
        return False


def get_video_duration(video_path):
    """Get video duration in seconds using ffprobe"""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"Error getting video duration: {e}")
        return None


def validate_video_duration(video_path, expected_duration_sec, fps, tolerance_sec=1.0):
    """
    Validate that video duration matches expected duration within tolerance.

    Args:
        video_path: Path to video file
        expected_duration_sec: Expected duration in seconds
        fps: Frames per second
        tolerance_sec: Allowed difference in seconds (default 1.0)

    Returns:
        bool: True if duration is within tolerance, False otherwise
    """
    actual_duration = get_video_duration(video_path)
    if actual_duration is None:
        return False

    duration_diff = abs(actual_duration - expected_duration_sec)

    is_valid = duration_diff <= tolerance_sec
    if not is_valid:
        print(
            f"Duration mismatch for {video_path.name}: "
            f"expected {expected_duration_sec:.2f}s, got {actual_duration:.2f}s "
            f"(diff: {duration_diff:.2f}s, tolerance: {tolerance_sec:.2f}s)"
        )

    return is_valid


def check_ffmpeg_available():
    """Check if ffmpeg is available and working"""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
        return result.returncode == 0
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return False


def check_ffprobe_available():
    """Check if ffprobe is available and working"""
    try:
        result = subprocess.run(
            ["ffprobe", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
        return result.returncode == 0
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return False


def count_images_in_folder(images_folder):
    """Count the number of cropped images in a folder"""
    try:
        image_files = list(images_folder.glob("image*_cropped.jpg"))
        return len(image_files)
    except Exception as e:
        print(f"Error counting images in {images_folder}: {e}")
        return 0


def create_video_from_images(
    images_folder,
    output_folder,
    video_name,
    fps,
    expected_duration_sec=None,
    rotation=None,
    dry_run=False,
    cpu_only=False,
    duration_tolerance=1.0,
):
    """
    Create video from images with fallback options and validation.

    Args:
        images_folder: Path to folder containing images
        output_folder: Path to output folder
        video_name: Name of the video (without extension)
        fps: Frames per second
        expected_duration_sec: Expected duration in seconds for validation
        rotation: Rotation to apply ('rotater' for 90° clockwise)
        dry_run: If True, only print what would be done

    Returns:
        dict: Status information with 'success', 'method_used', 'message'
    """
    video_path = output_folder / f"{video_name}.mp4"
    temp_video_path = output_folder / f"{video_name}_temp.mp4"

    if video_path.exists():
        return {
            "success": False,
            "method_used": "none",
            "message": "Video already exists",
        }

    if dry_run:
        print(f"DRY RUN: would run ffmpeg to create {video_path}")
        if rotation:
            print(f"DRY RUN: would apply rotation {rotation} to {video_path}")
        return {
            "success": True,
            "method_used": "dry_run",
            "message": "Dry run completed",
        }

    # Count images to estimate expected frames
    num_images = count_images_in_folder(images_folder)
    if num_images == 0:
        return {
            "success": False,
            "method_used": "none",
            "message": "No images found in folder",
        }

    # Get current timestamp for log file
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    log_file_name = f"ffmpeg_log_{video_name}_{now_str}.txt"

    # Get the full path to the conda environment's ffmpeg
    import shutil

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        ffmpeg_path = "ffmpeg"  # Fallback to PATH lookup

    # Try CUDA first (unless cpu_only is specified), then fallback to CPU
    methods_to_try = []
    if not cpu_only:
        methods_to_try.append(
            {
                "name": "cuda",
                "command": (
                    f"{ffmpeg_path} -y -loglevel error -hwaccel cuda -r {fps} -i {images_folder.as_posix()}/image%d_cropped.jpg -pix_fmt yuv420p -c:v libx265 -crf 15 {temp_video_path.as_posix()}"
                ),
            }
        )

    methods_to_try.append(
        {
            "name": "cpu",
            "command": (
                f"{ffmpeg_path} -y -loglevel error -r {fps} -i {images_folder.as_posix()}/image%d_cropped.jpg -pix_fmt yuv420p -c:v libx265 -crf 15 {temp_video_path.as_posix()}"
            ),
        }
    )

    success = False
    method_used = "none"
    error_messages = []

    for method in methods_to_try:
        print(f"Attempting video creation with {method['name']} encoding...")

        try:
            # Remove temp file if it exists from previous attempt
            if temp_video_path.exists():
                temp_video_path.unlink()

            # Run ffmpeg command
            with open(log_file_name, "w") as f:
                result = subprocess.run(
                    method["command"],
                    shell=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=3600,  # 1 hour timeout
                )

            # Check if ffmpeg succeeded
            if result.returncode != 0:
                with open(log_file_name, "r") as f:
                    error_output = f.read()
                error_messages.append(
                    f"{method['name']}: FFmpeg failed with return code {result.returncode}. Error: {error_output}"
                )
                continue

            # Check if video file was created and has reasonable size
            if not temp_video_path.exists():
                error_messages.append(f"{method['name']}: Video file was not created")
                continue

            if temp_video_path.stat().st_size < 1024:  # Less than 1KB
                error_messages.append(
                    f"{method['name']}: Video file is too small ({temp_video_path.stat().st_size} bytes)"
                )
                continue

            # Check video integrity
            if not check_video_integrity(temp_video_path.as_posix()):
                error_messages.append(f"{method['name']}: Video integrity check failed")
                continue

            # Validate duration if expected duration is provided
            if expected_duration_sec is not None:
                if not validate_video_duration(
                    temp_video_path,
                    expected_duration_sec,
                    fps,
                    tolerance_sec=duration_tolerance,
                ):
                    error_messages.append(
                        f"{method['name']}: Video duration validation failed"
                    )
                    continue

            # All checks passed - this method worked
            success = True
            method_used = method["name"]
            break

        except subprocess.TimeoutExpired:
            error_messages.append(f"{method['name']}: FFmpeg timed out after 1 hour")
            continue
        except Exception as e:
            error_messages.append(f"{method['name']}: Unexpected error: {str(e)}")
            continue

    # Handle results
    if success:
        # Apply rotation if specified
        final_video_path = video_path
        if rotation:
            rotated_video_path = output_folder / f"{video_name}_rotated.mp4"
            if rotation == "rotater":
                print("Rotating video 90 degrees clockwise")
                try:
                    ffmpeg_command = f"{ffmpeg_path} -y -loglevel error -i {temp_video_path} -vf 'transpose=1' {rotated_video_path}"
                    result = subprocess.run(
                        ffmpeg_command, shell=True, timeout=1800
                    )  # 30 min timeout

                    if result.returncode == 0 and rotated_video_path.exists():
                        final_video_path = rotated_video_path
                        temp_video_path.unlink()  # Remove unrotated temp file
                    else:
                        print("Rotation failed, keeping unrotated video")
                        final_video_path = temp_video_path
                except Exception as e:
                    print(f"Rotation failed: {e}, keeping unrotated video")
                    final_video_path = temp_video_path
            else:
                final_video_path = temp_video_path
        else:
            final_video_path = temp_video_path

        # Move temp file to final location
        if final_video_path != video_path:
            final_video_path.rename(video_path)

        # Clean up log file on success
        if os.path.exists(log_file_name):
            os.remove(log_file_name)

        return {
            "success": True,
            "method_used": method_used,
            "message": f"Video created successfully using {method_used} encoding",
        }

    else:
        # All methods failed - keep log file and temp files for debugging
        error_message = f"All encoding methods failed for {video_name}:\n" + "\n".join(
            error_messages
        )
        print(f"ERROR: {error_message}")

        # Clean up temp file
        if temp_video_path.exists():
            temp_video_path.unlink()

        return {"success": False, "method_used": "none", "message": error_message}


def search_folder_for_images(
    folder_path,
    output_folder,
    fps,
    expected_durations=None,
    dry_run=False,
    cpu_only=False,
    no_duration_check=False,
    auto_fix_invalid=False,
    duration_tolerance=1.0,
):
    """
    Search for image folders and create videos with comprehensive validation.

    Args:
        folder_path: Path to search for image folders
        output_folder: Base output folder
        fps: Frames per second
        expected_durations: Dict mapping folder names to expected durations in seconds
        dry_run: If True, only show what would be done
    """
    subdirs = []
    # Only consider folders that contain cropped image frames to avoid needless traversal
    for subdir in folder_path.glob("**/*"):
        if subdir.is_dir() and any(subdir.glob("image*_cropped.jpg")):
            subdirs.append(subdir)

    # Track statistics
    stats = {
        "total": len(subdirs),
        "skipped_existing": 0,
        "created_successfully": 0,
        "failed": 0,
        "corrupted_removed": 0,
    }
    failed_videos = []

    with tqdm(total=len(subdirs), desc="Processing videos") as pbar:
        for subdir in subdirs:
            relative_subdir = subdir.relative_to(folder_path)
            video_output_folder = output_folder / relative_subdir
            video_name = relative_subdir.name
            video_path = video_output_folder / f"{video_name}.mp4"

            # Get expected duration for this video if available
            expected_duration = None
            if not no_duration_check and expected_durations:
                if video_name in expected_durations:
                    expected_duration = expected_durations[video_name]
                elif "__default_duration__" in expected_durations:
                    # Use the default duration for all videos
                    expected_duration = expected_durations["__default_duration__"]

            # In dry run, just report what would be done
            if dry_run:
                print(f"DRY RUN: found images in {subdir}")
                print(
                    f"DRY RUN: would ensure output folder {video_output_folder} exists"
                )
                print(f"DRY RUN: would create video named {video_path} with fps={fps}")
                if expected_duration:
                    print(
                        f"DRY RUN: would validate duration against {expected_duration:.2f}s (tolerance {duration_tolerance:.2f}s)"
                    )
                pbar.update(1)
                continue

            video_output_folder.mkdir(parents=True, exist_ok=True)

            # Check if video already exists and is valid
            video_needs_creation = True
            if video_path.exists():
                if check_video_integrity(video_path.as_posix()):
                    # Check duration if we have expected duration and duration check is enabled
                    if (
                        no_duration_check
                        or expected_duration is None
                        or validate_video_duration(
                            video_path,
                            expected_duration,
                            fps,
                            tolerance_sec=duration_tolerance,
                        )
                    ):
                        print(
                            f"Video {video_name} already exists and is valid, skipping"
                        )
                        stats["skipped_existing"] += 1
                        video_needs_creation = False
                    else:
                        print(
                            f"Video {video_name} exists but has wrong duration (tolerance {duration_tolerance:.2f}s)"
                        )
                else:
                    print(f"Video {video_name} exists but is corrupted")

                if video_needs_creation:
                    if auto_fix_invalid:
                        print(
                            f"Auto-fix enabled: removing existing invalid video: {video_path.as_posix()}"
                        )
                        try:
                            video_path.unlink()
                            stats["corrupted_removed"] += 1
                        except Exception as e:
                            print(f"Failed to remove {video_path}: {e}")
                            stats["failed"] += 1
                            video_needs_creation = False
                    else:
                        remove_video = input(
                            f"Do you want to remove the existing video {video_name} and recreate it? (y/n): "
                        )
                        if remove_video.lower() == "y":
                            print(f"Removing existing video: {video_path.as_posix()}")
                            video_path.unlink()
                            stats["corrupted_removed"] += 1
                        else:
                            print(f"Skipping {video_name}")
                            stats["skipped_existing"] += 1
                            video_needs_creation = False

            # Create video if needed
            if video_needs_creation:
                print(f"Creating video for {video_name}...")
                result = create_video_from_images(
                    subdir,
                    video_output_folder,
                    video_name,
                    fps,
                    expected_duration_sec=expected_duration,
                    cpu_only=cpu_only,
                    duration_tolerance=duration_tolerance,
                )

                if result["success"]:
                    print(f"✓ {video_name}: {result['message']}")
                    stats["created_successfully"] += 1
                else:
                    print(f"✗ {video_name}: {result['message']}")
                    stats["failed"] += 1
                    failed_videos.append(
                        {
                            "name": video_name,
                            "folder": str(subdir),
                            "error": result["message"],
                        }
                    )

            pbar.update(1)

    # Print summary
    print("\n" + "=" * 60)
    print("VIDEO CREATION SUMMARY")
    print("=" * 60)
    print(f"Total folders processed: {stats['total']}")
    print(f"Videos already existing (skipped): {stats['skipped_existing']}")
    print(f"Videos created successfully: {stats['created_successfully']}")
    print(f"Videos failed to create: {stats['failed']}")
    print(f"Corrupted videos removed: {stats['corrupted_removed']}")

    if failed_videos:
        print(f"\nFAILED VIDEOS ({len(failed_videos)}):")
        print("-" * 40)
        for failed in failed_videos:
            print(f"  • {failed['name']}")
            print(f"    Folder: {failed['folder']}")
            print(f"    Error: {failed['error']}")
            print()

        # Write failed videos to a log file for later reference
        log_file = output_folder / "failed_videos.log"
        with open(log_file, "w") as f:
            f.write("Failed Videos Log\n")
            f.write("=" * 50 + "\n\n")
            for failed in failed_videos:
                f.write(f"Video: {failed['name']}\n")
                f.write(f"Folder: {failed['folder']}\n")
                f.write(f"Error: {failed['error']}\n")
                f.write("-" * 30 + "\n")
        print(f"Failed videos logged to: {log_file}")

    print("=" * 60)

    # Return statistics for use by calling function
    return stats


def load_duration_data(experiment_folder):
    """
    Load duration data from duration.npy file in experiment folder.

    Args:
        experiment_folder: Path to experiment folder

    Returns:
        dict: Mapping of video names to expected durations in seconds, or None if not found
    """
    duration_file = experiment_folder / "duration.npy"
    if not duration_file.exists():
        print(f"Warning: duration.npy not found in {experiment_folder}")
        return None

    try:
        duration_data = np.load(duration_file, allow_pickle=True)

        # Handle different possible formats of duration data
        if isinstance(duration_data, np.ndarray):
            if duration_data.ndim == 0:
                # Single value - this might be total duration or duration per video
                total_duration = float(duration_data.item())
                print(f"Expected videos duration: {total_duration}")
                # Create a default mapping assuming all videos have this duration
                durations_dict = {}
                # We'll populate this as we encounter videos, using the single duration for all
                return {"__default_duration__": total_duration}
            elif len(duration_data.shape) == 1:
                # 1D array - assume these are durations for videos in order
                durations_dict = {}
                for i, duration in enumerate(duration_data):
                    durations_dict[f"video_{i:03d}"] = float(duration)
                return durations_dict
        elif isinstance(duration_data, dict):
            # Already a dictionary
            return {str(k): float(v) for k, v in duration_data.items()}
        else:
            print(f"Warning: Unrecognized duration data format in {duration_file}")
            return None

    except Exception as e:
        print(f"Error loading duration data from {duration_file}: {e}")
        return None


def process_all(
    dry_run=False,
    cpu_only=False,
    no_duration_check=False,
    auto_fix_invalid=False,
    duration_tolerance=1.0,
):
    # Gather experiments and matches first when in dry run to provide a clean summary
    recorded_folders = [
        f for f in data_folder.iterdir() if f.is_dir() and f.name.endswith("_Checked")
    ]

    # Track results for each experiment
    experiment_results = {}

    if dry_run:
        experiments = []
        matched_map = {}
        processing_map = {}
        missing = []
        for folder in recorded_folders:
            output_folder_name = folder.name.replace("_Cropped_Checked", "")
            experiments.append(output_folder_name)
            matched_output_folder = None
            matched_output_root = None
            processing_folder_found = None

            for root in OUTPUT_PATHS:
                # Check for normal folder
                candidate = root / output_folder_name
                if candidate.exists() and (candidate / "metadata.json").exists():
                    matched_output_root = root
                    matched_output_folder = candidate
                    break

                # Check for _Processing folder from interrupted run
                processing_candidate = root / f"{output_folder_name}_Processing"
                if (
                    processing_candidate.exists()
                    and (processing_candidate / "metadata.json").exists()
                ):
                    matched_output_root = root
                    processing_folder_found = processing_candidate
                    break

            if matched_output_folder:
                matched_map[output_folder_name] = str(matched_output_folder)
            elif processing_folder_found:
                processing_map[output_folder_name] = str(processing_folder_found)
            else:
                missing.append(output_folder_name)

        # Print summary
        print("DRY RUN SUMMARY")
        print("--------------")
        if experiments:
            print("Found experiments to process:")
            for e in experiments:
                print(f" - {e}")
        else:
            print("No experiments found to process in data folder.")

        print("")
        if matched_map:
            print("Found pre-made output folders (will use these):")
            for exp, path in matched_map.items():
                print(f" - {exp} -> {path}")
        else:
            print("No matching pre-made output folders found in OUTPUT_PATHS.")

        if processing_map:
            print("Found interrupted processing folders (will resume these):")
            for exp, path in processing_map.items():
                print(f" - {exp} -> {path} [RESUME]")

        print("")
        if missing:
            print(
                "Experiments missing prefilled output directories (no metadata.json found):"
            )
            for e in missing:
                print(f" - {e}")
            print("")
            print(f"Searched output roots: {[str(p) for p in OUTPUT_PATHS]}")

        # End dry run without performing any actions
        return experiment_results

    # Non-dry run processing
    for folder in recorded_folders:
        print(f"Processing folder: {folder.name}")
        output_folder_name = folder.name.replace("_Cropped_Checked", "")

        # Search known output roots for an existing experiment folder with metadata.json
        # Also look for _Processing folders from interrupted runs
        matched_output_root = None
        matched_output_folder = None
        processing_folder_found = None

        for root in OUTPUT_PATHS:
            # First try the normal folder name
            candidate = root / output_folder_name
            if candidate.exists() and (candidate / "metadata.json").exists():
                matched_output_root = root
                matched_output_folder = candidate
                break

            # Also check for _Processing folder from interrupted run
            processing_candidate = root / f"{output_folder_name}_Processing"
            if (
                processing_candidate.exists()
                and (processing_candidate / "metadata.json").exists()
            ):
                matched_output_root = root
                processing_folder_found = processing_candidate
                print(f"Found interrupted processing folder: {processing_candidate}")
                break

        if matched_output_folder is None and processing_folder_found is None:
            print(
                "Warning: this experiment found in the data folder doesn't have a prefilled output directory in any of the output paths known.\n"
                f"Searched paths: {[str(p) for p in OUTPUT_PATHS]}\n"
                f"Experiment folder name: {output_folder_name}\n"
                "Expected a folder with a metadata.json inside one of the output roots."
            )
            # Skip this experiment; user can create the experiment folder in one of the OUTPUT_PATHS
            experiment_results[output_folder_name] = {
                "status": "skipped",
                "reason": "no_output_directory",
            }
            continue

        # Use the matched output folder for further processing
        output_path_local = matched_output_root

        # Handle different scenarios: normal folder vs existing _Processing folder
        if matched_output_folder is not None:
            # Normal case: rename original folder to _Processing
            output_folder = matched_output_folder
            processing_output_folder = (
                output_path_local / f"{output_folder_name}_Processing"
            )
            if not processing_output_folder.exists():
                if dry_run:
                    print(
                        f"DRY RUN: would rename {output_folder} -> {processing_output_folder}"
                    )
                else:
                    output_folder.rename(processing_output_folder)
        elif processing_folder_found is not None:
            # Resume case: _Processing folder already exists from interrupted run
            processing_output_folder = processing_folder_found
            print(f"Resuming processing in existing folder: {processing_output_folder}")
        else:
            # This shouldn't happen due to the check above, but handle it anyway
            print(f"ERROR: No valid folder found for {output_folder_name}")
            continue

        # Load the fps value from the fps.npy file in the experiment directory
        fps_file = processing_output_folder / "fps.npy"
        if fps_file.exists():
            fps = np.load(fps_file)
            fps = str(fps)
        else:
            print(f"Error: fps.npy file not found in {processing_output_folder}")
            # Revert folder name to allow easy retry
            if not dry_run and processing_output_folder.exists():
                try:
                    original_folder = output_path_local / output_folder_name
                    processing_output_folder.rename(original_folder)
                    print(f"Reverted folder name to {original_folder.name}")
                except Exception as revert_error:
                    print(f"Warning: Could not revert folder name: {revert_error}")
            experiment_results[output_folder_name] = {
                "status": "failed",
                "reason": "no_fps_file",
            }
            continue

        # Load duration data for validation
        expected_durations = None
        if not no_duration_check:
            expected_durations = load_duration_data(processing_output_folder)
            if expected_durations:
                print(f"Loaded duration data for {len(expected_durations)} videos")
            else:
                print("No duration data available - skipping duration validation")
        else:
            print("Duration validation disabled by --no-duration-check flag")

        # Process the images with enhanced validation
        try:
            stats = search_folder_for_images(
                folder,
                processing_output_folder,
                fps,
                expected_durations=expected_durations,
                dry_run=dry_run,
                cpu_only=cpu_only,
                no_duration_check=no_duration_check,
                auto_fix_invalid=auto_fix_invalid,
                duration_tolerance=duration_tolerance,
            )
            print(f"Processing of {folder.name} complete.")
            experiment_results[output_folder_name] = stats or {"status": "unknown"}
        except Exception as e:
            print(f"Error processing {folder.name}: {e}")
            print("CRITICAL: Processing failed - original images are preserved")
            # Revert folder name to allow easy retry
            if not dry_run and processing_output_folder.exists():
                try:
                    original_folder = output_path_local / output_folder_name
                    processing_output_folder.rename(original_folder)
                    print(f"Reverted folder name to {original_folder.name}.")
                except Exception as revert_error:
                    print(f"Warning: Could not revert folder name: {revert_error}")
            experiment_results[output_folder_name] = {
                "status": "failed",
                "reason": str(e),
            }
            continue  # Skip to next experiment

        # Finalize: rename to _Videos if we created new videos OR if existing videos were all valid
        # This prevents CheckVideos.py from removing source images when no valid videos exist
        finalize_ok = False
        if stats:
            created_ok = stats.get("created_successfully", 0) > 0
            existing_all_valid = (
                stats.get("skipped_existing", 0) > 0 and stats.get("failed", 0) == 0
            )
            finalize_ok = created_ok or existing_all_valid

        if not dry_run and stats and finalize_ok:
            new_output_folder_name = f"{output_folder_name}_Videos"
            new_output_folder = output_path_local / new_output_folder_name
            try:
                processing_output_folder.rename(new_output_folder)
                if created_ok:
                    print(
                        f"Experiment {output_folder_name} completed successfully - {stats['created_successfully']} videos created"
                    )
                    experiment_results[output_folder_name][
                        "status"
                    ] = "completed_with_videos"
                else:
                    print(
                        f"Experiment {output_folder_name} finalized using existing valid videos (created this run: 0, failed: {stats.get('failed', 0)})"
                    )
                    experiment_results[output_folder_name][
                        "status"
                    ] = "completed_using_existing"
            except Exception as e:
                print(f"Warning: Could not rename output folder: {e}")
                print(f"Output remains in: {processing_output_folder}")
                experiment_results[output_folder_name][
                    "status"
                ] = "completed_but_rename_failed"
        elif not dry_run:
            # If no videos were created successfully, revert folder name for easy retry
            print(
                f"WARNING: Not finalizing {output_folder_name}: no new videos created and/or failures present"
            )
            try:
                original_folder = output_path_local / output_folder_name
                processing_output_folder.rename(original_folder)
                print(f"Reverted folder name to {original_folder.name} for easy retry")

            except Exception as revert_error:
                print(f"Warning: Could not revert folder name: {revert_error}")
                print(f"Folder remains as: {processing_output_folder.name}")
                print("You may need to manually rename it before retrying")
            experiment_results[output_folder_name]["status"] = "completed_no_videos"
        else:
            print(
                f"DRY RUN: would rename {processing_output_folder} -> _Videos only if videos were created"
            )
            print(f"DRY RUN: would revert to original name if no videos created")

    return experiment_results


def cleanup_processing_folders(dry_run=False):
    """
    Clean up any existing _Processing folders from failed runs.
    This allows easy retry without manual cleanup.
    """
    print("Searching for _Processing folders to clean up...")

    cleaned_count = 0
    for root in OUTPUT_PATHS:
        if not root.exists():
            continue

        processing_folders = list(root.glob("*_Processing"))
        for processing_folder in processing_folders:
            if not processing_folder.is_dir():
                continue

            # Extract original experiment name
            original_name = processing_folder.name.replace("_Processing", "")
            original_folder = root / original_name

            if dry_run:
                print(f"DRY RUN: would rename {processing_folder} -> {original_folder}")
            else:
                try:
                    processing_folder.rename(original_folder)
                    print(
                        f"✓ Cleaned up: {processing_folder.name} -> {original_folder.name}"
                    )
                    cleaned_count += 1
                except Exception as e:
                    print(f"✗ Failed to clean up {processing_folder.name}: {e}")

    if not dry_run:
        print(f"Cleanup completed. {cleaned_count} folders were cleaned up.")
    else:
        print("DRY RUN: cleanup completed.")


def main():
    parser = argparse.ArgumentParser(description="Create videos from cropped images")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do a dry run and show planned actions without making changes",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Skip CUDA acceleration and use CPU-only encoding",
    )
    parser.add_argument(
        "--no-duration-check",
        action="store_true",
        help="Skip duration validation against duration.npy",
    )
    parser.add_argument(
        "--skip-post-processing",
        action="store_true",
        help="Skip running CheckVideos.py after video creation",
    )
    parser.add_argument(
        "--cleanup-processing-folders",
        action="store_true",
        help="Clean up any existing _Processing folders from failed runs",
    )
    # Default to auto-fix enabled; allow opt-out
    parser.add_argument(
        "--no-auto-fix-invalid",
        action="store_false",
        dest="auto_fix_invalid",
        help="Disable automatic removal/remake and prompt instead",
    )
    parser.set_defaults(auto_fix_invalid=True)
    parser.add_argument(
        "--duration-tolerance",
        type=float,
        default=1.0,
        help="Duration tolerance in seconds for validation checks",
    )
    args = parser.parse_args()

    # Clean up processing folders if requested
    if args.cleanup_processing_folders:
        cleanup_processing_folders(dry_run=args.dry_run)
        if args.dry_run:
            return
        print("Cleanup completed. You can now run the script normally.")
        return

    # Check if required tools are available
    if not args.dry_run:
        print("Checking required tools...")

        if not check_ffmpeg_available():
            print("ERROR: ffmpeg is not available or not working properly")
            print("Please install ffmpeg or ensure it's in your PATH")
            print("The error you're seeing suggests a library compatibility issue.")
            print("Try running: conda install ffmpeg -c conda-forge")
            sys.exit(1)
        else:
            print("✓ ffmpeg is available")

        if not check_ffprobe_available():
            print("ERROR: ffprobe is not available or not working properly")
            print("Please install ffprobe (usually comes with ffmpeg)")
            sys.exit(1)
        else:
            print("✓ ffprobe is available")

        print("")

    # Pass arguments to process_all function and get results
    results = process_all(
        dry_run=args.dry_run,
        cpu_only=args.cpu_only,
        no_duration_check=args.no_duration_check,
        auto_fix_invalid=args.auto_fix_invalid,
        duration_tolerance=args.duration_tolerance,
    )

    # Only run post-processing if we're not in dry run mode, not skipping post-processing,
    # and if we created videos OR validated existing ones without failures
    if not args.dry_run and not args.skip_post_processing:
        # Check if any experiments had successful video creation
        should_run_post_processing = False
        if results:
            for exp_name, exp_stats in results.items():
                created_ok = exp_stats.get("created_successfully", 0) > 0
                existing_all_valid = (
                    exp_stats.get("skipped_existing", 0) > 0
                    and exp_stats.get("failed", 0) == 0
                )
                if created_ok or existing_all_valid:
                    should_run_post_processing = True
                    break

        if should_run_post_processing:
            script_dir = Path(__file__).resolve().parent
            conda_path = "/home/matthias/miniconda3/bin/activate"
            CheckVideos_path = script_dir / "CheckVideos.py"
            command = f". {conda_path} processing && python {CheckVideos_path}"

            print("Running post-processing video checks...")
            try:
                subprocess.run(command, shell=True, executable="/bin/bash", check=True)
                print("Post-processing completed successfully")
            except subprocess.CalledProcessError as e:
                print(
                    f"Warning: Post-processing failed with return code {e.returncode}"
                )
            except Exception as e:
                print(f"Warning: Post-processing failed: {e}")
        else:
            print(
                "Skipping post-processing because no new/valid videos were detected for finalization"
            )
            print(
                f"Fix video creation issues before running again. Log files are available at {OUTPUT_PATHS}."
            )
    elif args.skip_post_processing:
        print("Skipping post-processing as requested")
    else:
        print(
            "DRY RUN: would conditionally run post-processing based on video creation success"
        )


if __name__ == "__main__":
    main()

# CHANGELOG:
# - Added comprehensive data protection: original images are NEVER deleted
# - Added CUDA failsafe with automatic CPU fallback
# - Added duration validation against duration.npy data
# - Added detailed error logging and progress reporting
# - Added command line options for different processing modes
# - Improved video integrity checking with multiple validation steps

# TODO: Add a way to resume an aborted processing in a given folder, by checking already existing videos integrity, skipping them and processing folder not yet done.
# TODO: Make the script run as a background process, always checking for non processed videos
# TODO: Add email/notification system for processing completion/failures
# TODO: Add support for different video codecs and quality settings
