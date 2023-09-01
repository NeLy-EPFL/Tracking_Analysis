from moviepy.editor import VideoFileClip, clips_array, TextClip
import os
import re


def create_grid_video(folder_path, duration=None, output_path=None):
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
        if duration:
            clip = clip.subclip(0, duration)
        
        # Label the clip with arena and corridor numbers
        match = re.search(r"arena(\d+)_corridor_(\d+)", video_file)
        if match:
            arena, corridor = match.groups()
            label = f"Arena {arena}, Corridor {corridor}"
            text_clip = TextClip(label, fontsize=24, color='white')
            text_clip = text_clip.set_position(('center', 'bottom')).set_duration(clip.duration)
            clip = clips_array([[clip], [text_clip]])
        
        clips.append(clip)

    # Create a grid from the clips
    grid = clips_array([clips])

    # Set the output path for the grid video
    if not output_path:
        output_path = os.path.join(folder_path, "grid.mp4")

    # Write the grid video to a file
    grid.write_videofile(output_path)


create_grid_video("/Users/ulric/Movies/Videos_NumOrdered/",
                  duration=3,
                  output_path="/Users/ulric/Movies/grid.mp4")