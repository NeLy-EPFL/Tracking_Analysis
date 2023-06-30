from moviepy.editor import VideoFileClip, clips_array
import os
import re

def create_grid_video(folder_path, start_time=None, end_time=None, output_path=None):
    # Get all video files in the folder
    video_files = [
        f for f in os.listdir(folder_path) if f.endswith((".mp4", ".avi", ".mov"))
    ]

    # Sort the video files by arena and corridor numbers
    def sort_key(video_file):
        match = re.search(r"arena(\d+)_corridor_(\d+)", video_file)
        if match:
            return int(match.group(1)), int(match.group(2))
        else:
            return 0, 0

    video_files.sort(key=sort_key)

    # Create a list of video clips
    clips = []
    for video_file in video_files:
        clip = VideoFileClip(os.path.join(folder_path, video_file))
        if start_time is not None or end_time is not None:
            if end_time == 'end':
                end_time = clip.duration
            clip = clip.subclip(start_time, end_time)
        clips.append(clip)

    # Create a grid from the clips
    grid = clips_array([clips])

    # Set the output path for the grid video
    if not output_path:
        output_path = os.path.join(folder_path, "grid.mp4")

    # Write the grid video to a file
    grid.write_videofile(output_path)
    
# Write last minute of the videos 
    
create_grid_video("/Users/ulric/Movies/Videos_NumOrdered/",
                  start_time=-60,
                  end_time='end',
                  output_path="/Users/ulric/Movies/Lastgrid.mp4")

# Write 2 mins in the middle of the video
create_grid_video("/Users/ulric/Movies/Videos_NumOrdered/",
                  start_time='middle',
                  end_time=120,
                  output_path="/Users/ulric/Movies/Midgrid.mp4")