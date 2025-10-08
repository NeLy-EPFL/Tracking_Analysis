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
]

# Backwards-compatible single-variable for scripts that reference `output_path`.
# It will be set per-experiment below when a matching folder is found.
output_path = None

# fps = "29"


def check_video_integrity(video_path):
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


def create_video_from_images(
    images_folder, output_folder, video_name, fps, rotation=None, dry_run=False
):
    video_path = output_folder / f"{video_name}.mp4"
    if not video_path.exists():
        # Get the current date and time
        now = datetime.now()
        # Format the date and time as a string
        now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        # Use the date and time to create a unique log file name
        log_file_name = f"ffmpeg_log_{now_str}.txt"
        #
        terminal_call = f"/usr/bin/ffmpeg -loglevel panic -nostats -hwaccel cuda -r {fps} -i {images_folder.as_posix()}/image%d_cropped.jpg -pix_fmt yuv420p -c:v libx265 -crf 15 {video_path.as_posix()}"
        if dry_run:
            print(f"DRY RUN: would run ffmpeg to create {video_path}")
            if rotation:
                print(f"DRY RUN: would apply rotation {rotation} to {video_path}")
            return True
        try:
            with open(log_file_name, "w") as f:
                subprocess.run(
                    terminal_call, shell=True, stdout=f, stderr=subprocess.STDOUT
                )
            # If the script completes without errors, remove the log file
            os.remove(log_file_name)

            # Apply rotation if specified
            if rotation:
                rotated_video_path = output_folder / f"{video_name}_rotated.mp4"
                if rotation == "rotater":
                    print("Rotating video 90 degrees clockwise")
                    ffmpeg_command = (
                        f"ffmpeg -i {video_path} -vf 'transpose=1' {rotated_video_path}"
                    )
                subprocess.run(ffmpeg_command, shell=True)
                video_path.unlink()  # Remove the original video file
                rotated_video_path.rename(
                    video_path
                )  # Rename the rotated video file to the original name

            return True
        except Exception as e:
            print(f"An error occurred: {e}")
            # If an error occurs, keep the log file and return False
            return False
    else:
        return False


# -loglevel panic -nostats >  This is to remove the output of the ffmpeg command from the terminal, to add right after the ffmpeg command


def search_folder_for_images(folder_path, output_folder, fps, dry_run=False):
    subdirs = []
    for subdir in folder_path.glob("**/*"):
        if subdir.is_dir() and any(
            file.name.endswith(".jpg") for file in subdir.glob("*")
        ):
            subdirs.append(subdir)
    with tqdm(total=len(subdirs), desc="Processing videos") as pbar:
        for subdir in subdirs:
            relative_subdir = subdir.relative_to(folder_path)
            video_output_folder = output_folder / relative_subdir
            video_name = relative_subdir.name
            video_path = video_output_folder / f"{video_name}.mp4"

            # In dry run, just report what would be done
            if dry_run:
                print(f"DRY RUN: found images in {subdir}")
                print(f"DRY RUN: would ensure output folder {video_output_folder} exists")
                print(f"DRY RUN: would create video named {video_path} with fps={fps}")
                pbar.update(1)
                continue

            video_output_folder.mkdir(parents=True, exist_ok=True)
            if not video_path.exists() or not check_video_integrity(
                video_path.as_posix()
            ):
                if video_path.exists():
                    print(f"Video {video_name} is corrupted.")
                    remove_video = input(
                        f"Do you want to remove the corrupted video {video_name}? (y/n): "
                    )
                    if remove_video.lower() == "y":
                        print(f"Removing corrupted video: {video_path.as_posix()}")
                        video_path.unlink()
                create_video_from_images(subdir, video_output_folder, video_name, fps)
            pbar.update(1)


def process_all(dry_run=False):
    # Gather experiments and matches first when in dry run to provide a clean summary
    recorded_folders = [f for f in data_folder.iterdir() if f.is_dir() and f.name.endswith("_Checked")]

    if dry_run:
        experiments = []
        matched_map = {}
        missing = []
        for folder in recorded_folders:
            output_folder_name = folder.name.replace("_Cropped_Checked", "")
            experiments.append(output_folder_name)
            matched_output_folder = None
            matched_output_root = None
            for root in OUTPUT_PATHS:
                candidate = root / output_folder_name
                if candidate.exists() and (candidate / 'metadata.json').exists():
                    matched_output_root = root
                    matched_output_folder = candidate
                    break
            if matched_output_folder:
                matched_map[output_folder_name] = str(matched_output_folder)
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

        print("")
        if missing:
            print("Experiments missing prefilled output directories (no metadata.json found):")
            for e in missing:
                print(f" - {e}")
            print("")
            print(f"Searched output roots: {[str(p) for p in OUTPUT_PATHS]}")

        # End dry run without performing any actions
        return

    # Non-dry run processing
    for folder in recorded_folders:
        print(f"Processing folder: {folder.name}")
        output_folder_name = folder.name.replace("_Cropped_Checked", "")

        # Search known output roots for an existing experiment folder with metadata.json
        matched_output_root = None
        matched_output_folder = None
        for root in OUTPUT_PATHS:
            candidate = root / output_folder_name
            if candidate.exists() and (candidate / 'metadata.json').exists():
                matched_output_root = root
                matched_output_folder = candidate
                break

        if matched_output_folder is None:
            print(
                "Warning: this experiment found in the data folder doesn't have a prefilled output directory in any of the output paths known.\n"
                f"Searched paths: {[str(p) for p in OUTPUT_PATHS]}\n"
                f"Experiment folder name: {output_folder_name}\n"
                "Expected a folder with a metadata.json inside one of the output roots."
            )
            # Skip this experiment; user can create the experiment folder in one of the OUTPUT_PATHS
            continue

        # Use the matched output folder for further processing
        output_path_local = matched_output_root
        output_folder = matched_output_folder
        processing_output_folder = output_path_local / f"{output_folder_name}_Processing"
        if not processing_output_folder.exists():
            if dry_run:
                print(f"DRY RUN: would rename {output_folder} -> {processing_output_folder}")
            else:
                output_folder.rename(processing_output_folder)

        # Load the fps value from the fps.npy file in the experiment directory
        fps_file = processing_output_folder / "fps.npy"
        if fps_file.exists():
            fps = np.load(fps_file)
            fps = str(fps)
        else:
            print(f"Error: fps.npy file not found in {folder}")
            continue

        search_folder_for_images(folder, processing_output_folder, fps, dry_run=dry_run)
        # Rename the output folder after all videos have been created
        print(f"Processing of {folder.name} complete.")
        new_output_folder_name = f"{output_folder_name}_Videos"
        new_output_folder = output_path_local / new_output_folder_name
        if dry_run:
            print(f"DRY RUN: would rename {processing_output_folder} -> {new_output_folder}")
        else:
            processing_output_folder.rename(new_output_folder)

def main():
    parser = argparse.ArgumentParser(description="Create videos from cropped images")
    parser.add_argument("--dry-run", action="store_true", help="Do a dry run and show planned actions without making changes")
    args = parser.parse_args()

    process_all(dry_run=args.dry_run)

    script_dir = Path(__file__).resolve().parent

    conda_path = "/home/matthias/miniconda3/bin/activate"
    CheckVideos_path = script_dir / "CheckVideos.py"
    command = f". {conda_path} processing && python {CheckVideos_path}"
    if args.dry_run:
        print(f"DRY RUN: would run post-processing command: {command}")
    else:
        subprocess.run(command, shell=True, executable="/bin/bash")


if __name__ == '__main__':
    main()

# TODO: Add a way to resume an aborted processing in a given folder, by checking already existing videos integrity, skipping them and processing folder not yet done.

# TODO : make the script run as a background process, always checking for non processed videos
