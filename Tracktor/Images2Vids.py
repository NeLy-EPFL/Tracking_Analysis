from pathlib import Path
from tqdm import tqdm
import subprocess
import os
import sys

data_folder = Path("/home/matthias/Videos/")
output_path = Path("/mnt/labserver/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Videos/")

fps = "30"

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

def create_video_from_images(images_folder, output_folder, video_name, fps):
    video_path = output_folder / f"{video_name}.mp4"
    if not video_path.exists():
        terminal_call = f"ffmpeg -loglevel panic -nostats -hwaccel cuda -r {fps} -i {images_folder.as_posix()}/image%d_cropped.jpg -pix_fmt yuv420p -c:v libx265 -x265-params log-level=error -vsync 0 -crf 15 {video_path.as_posix()}"
        subprocess.run(terminal_call, shell=True)
        return True
    else:
        return False

def search_folder_for_images(folder_path, output_folder, fps):
    subdirs = []
    for subdir in folder_path.glob('**/*'):
        if subdir.is_dir() and any(file.name.endswith('.jpg') for file in subdir.glob('*')):
            subdirs.append(subdir)
    with tqdm(total=len(subdirs), desc="Processing videos") as pbar:
        for subdir in subdirs:
            relative_subdir = subdir.relative_to(folder_path)
            video_output_folder = output_folder / relative_subdir
            video_output_folder.mkdir(parents=True, exist_ok=True)
            video_name = relative_subdir.name
            video_path = video_output_folder / f"{video_name}.mp4"
            if not video_path.exists() or not check_video_integrity(video_path.as_posix()):
                if video_path.exists():
                    print(f"Video {video_name} is corrupted.")
                    remove_video = input(f"Do you want to remove the corrupted video {video_name}? (y/n): ")
                    if remove_video.lower() == 'y':
                        print(f"Removing corrupted video: {video_path.as_posix()}")
                        video_path.unlink()
                create_video_from_images(subdir, video_output_folder, video_name, fps)
            pbar.update(1)

for folder in data_folder.iterdir():
    if folder.is_dir() and folder.name.endswith("_Checked"):
        print(f"Processing folder: {folder.name}")
        output_folder_name = folder.name.replace("_Cropped_Checked", "")
        output_folder = output_path / f"{output_folder_name}"
        output_folder.mkdir(exist_ok=True)
        search_folder_for_images(folder, output_folder, fps)
        # Rename the output folder after all videos have been created
        print(f"Processing of {folder.name} complete.")
        new_output_folder_name = f"{output_folder_name}_Videos"
        new_output_folder = output_path / new_output_folder_name
        output_folder.rename(new_output_folder)

if os.isatty(sys.stdin.fileno()):
    run_checkcrops = input(
        "Launch verification of processed folders integrity? (y/n): "
    )
    if run_checkcrops.lower() == "y":
        subprocess.run(["/home/matthias/Tracking_Analysis/Tracktor/CheckVideos.sh"])

#TODO: Add a way to resume an aborted processing in a given folder, by checking already existing videos integrity, skipping them and processing folder not yet done.

#TODO : make the script run as a background process, always checking for non processed videos