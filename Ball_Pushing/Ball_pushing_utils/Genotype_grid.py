from moviepy.editor import VideoFileClip, concatenate_videoclips
import os

# Define the input directory and output file
input_dir = "/mnt/labserver/DURRIEU_Matthias/Videos/240129_TNT_Fine/TNTxDDC"
output_file = "/mnt/labserver/DURRIEU_Matthias/Videos/Genotype_grids/test.mp4"

# Get a list of all video files in the input directory
video_files = [
    f for f in os.listdir(input_dir) if f.endswith((".mp4", ".avi", ".mov"))
]  # Add more extensions if needed

# Create a list of video clips
clips = [VideoFileClip(os.path.join(input_dir, f)).rotate(90) for f in video_files]

# Concatenate all clips horizontally
final_clip = concatenate_videoclips(clips, method="compose")

# Write the result to the output file
final_clip.write_videofile(output_file)
