import os
import subprocess
from pathlib import Path

def rotate_videos(input_folder, output_folder):
    """
    Rotates all videos in the given input folder 90 degrees clockwise.

    Parameters
    ----------
    input_folder : str
        The path to the folder containing the input videos.
    """
    # Convert input_folder to a Path object and get all mp4 files
    input_folder = Path(input_folder)
    video_files = list(input_folder.glob("*.mp4"))

    # Iterate over all video files in the folder
    for video_file in video_files:
        # Create output file path
        output_file = output_folder + video_file.name

        # Construct ffmpeg command
        cmd = [
            "ffmpeg",
            "-i", str(video_file),  # Input file
            "-vf", "transpose=1",  # Filter for 90 degree clockwise rotation
            str(output_file)  # Output file
        ]

        # Run ffmpeg command
        subprocess.run(cmd)

# Example usage:
rotate_videos(input_folder = "/mnt/labserver/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Grids/Newcrop", output_folder = "/mnt/labserver/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Grids/Newcrop_rotated/")
