from moviepy.editor import *
import os
from pathlib import Path

# Set the input folder path
input_folder = Path("/mnt/labserver/DURRIEU_Matthias/Videos/GridClipsRotated/")
# Get all video files from the input folder
input_files = list(input_folder.glob("*clip*.mp4"))
num_black_frames = 10

for file in input_files:
    video = VideoFileClip(file.as_posix())
    black_frame = ColorClip(size=video.size, color=(0, 0, 0), duration=1 / video.fps)
    black_frames = [black_frame] * num_black_frames
    final_clip = concatenate_videoclips([video] + black_frames)
    final_clip.write_videofile(str(file.with_name(f"black_{file.name}")))