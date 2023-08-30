import math
import subprocess
from pathlib import Path


def get_video_size(video_path):
    """
    Returns the width and height of the video at the given path.

    Parameters
    ----------
    video_path : Path
    The path to the video file.

    Returns
    -------
    width : int
    The width of the video.
    height : int
    The height of the video.

    Returns None if there was an error.
    """
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Error running ffprobe on {video_path}: {result.stderr}")
        return None
    else:
        return map(int, result.stdout.strip().split(","))


def create_blank_video(width, height, output_path):
    subprocess.run(
        [
            "ffmpeg",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={width}x{height}:d=1:r=25",
            "-c:v",
            "libx264",
            "-tune",
            "stillimage",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
    )


def create_grid_video(input_folder, output_path, keyword):
    """
    Creates a grid video from the videos in the given input folder.

    Parameters
    ----------
    input_folder : Path
    The path to the folder containing the input videos.
    output_path : Path
    The path to the output video.
    keyword : str
    The keyword to use to find the input videos.

    Returns
    -------
    None
    """
    # Set the input folder path
    input_folder = Path(input_folder)
    # Get all video files from the input folder
    input_files = list(input_folder.glob(f"*{keyword}*.mp4"))

    # Sort the input files by the number in their name
    input_files.sort(key=lambda f: int(f.stem.split("_")[1]))

    print(f"input_files: {input_files}")

    # Get the width and height of the first video
    width, height = get_video_size(input_files[0])

    # Calculate the number of columns and rows for the grid layout
    num_videos = len(input_files)
    aspect_ratio = 16 / 9
    best_layout = (1, num_videos)
    best_diff = float("inf")
    for num_cols in range(1, num_videos + 1):
        num_rows = math.ceil(num_videos / num_cols)
        diff = abs(aspect_ratio - ((num_cols * width) / (num_rows * height)))
        if diff < best_diff:
            best_diff = diff
            best_layout = (num_rows, num_cols)
    num_rows, num_cols = best_layout

    print(f"num_videos: {num_videos}")
    print(f"num_cols: {num_cols}")
    print(f"num_rows: {num_rows}")

    # Create blank videos if necessary
    while num_videos < num_cols * num_rows:
        blank_video = input_folder / f"blank{num_videos}.mp4"
        create_blank_video(width, height, blank_video)
        input_files.append(blank_video)
        num_videos += 1

    # Set the scale factor
    scale_factor = 0.5

    # Create the filter_complex argument for the ffmpeg command
    filter_complex = "".join(
        f"[{i}:v]scale=iw*{scale_factor}:ih*{scale_factor}[s{i}];[s{i}]copy[v{i}];"
        for i in range(num_cols * num_rows)
    )
    for row in range(num_rows):
        row_inputs = "".join(f"[v{row * num_cols + col}]" for col in range(num_cols))
        filter_complex += f"{row_inputs}hstack=inputs={num_cols}[h{row}];"
    vstack_inputs = "".join(f"[h{row}]" for row in range(num_rows))
    filter_complex += f"{vstack_inputs}vstack=inputs={num_rows}[v]"
    filter_complex += f";[v]pad=ceil(iw/2)*2:ceil(ih/2)*2[v]"

    print(f"filter_complex : {filter_complex}")

    # Create the ffmpeg command arguments
    ffmpeg_args = ["ffmpeg"]
    for input_file in input_files:
        ffmpeg_args.extend(["-i", str(input_file)])
    ffmpeg_args.extend(
        ["-filter_complex", filter_complex, "-map", "[v]", str(output_path)]
    )

    # Run the ffmpeg command
    subprocess.run(ffmpeg_args)


# Example usage:
create_grid_video(
    input_folder="/mnt/labserver/DURRIEU_Matthias/Videos/GridClipsRotated/",
    output_path="/mnt/labserver/DURRIEU_Matthias/Videos/GridClipsRotated/black_grid_video.mp4",
    keyword="blackclip",
)
