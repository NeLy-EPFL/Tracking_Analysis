import math
import os
import subprocess
import json
from pathlib import Path
from operator import itemgetter
from itertools import groupby
import re


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


def create_grid_video(input_folder, output_path, keyword=None):
    """
    Creates a grid video from the videos in the given input folder.

    Parameters
    ----------
    input_folder : Path
        The path to the folder containing the input videos.
    output_path : Path
        The path to the output video.
    keyword : str, optional
        The keyword to use to find the input videos.

    Returns
    -------
    None
    """
    # Set the input folder path
    input_folder = Path(input_folder)

    # Get all video files from the input folder
    if keyword:
        input_files = list(input_folder.glob(f"*{keyword}*.mp4"))
        # Sort the input files by the number in their name
        input_files.sort(key=lambda f: int(f.stem.split("_")[1]))
    else:
        # If no keyword is provided, sort by numbers in file name
        input_files = sorted(
            list(input_folder.glob("*.mp4")),
            key=lambda f: int(re.findall(r"\d+", f.stem)[0]),
        )

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


def get_video_dimensions(video_path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    dimensions = json.loads(result.stdout)["streams"][0]
    return dimensions["width"], dimensions["height"]


def create_horizontal_video(
    source, output_path, date=None, arena=None, keyword=None, test_mode=False
):
    """Creates a horizontal video from the videos in the given input folder or files.

    Parameters
    ----------
    source : Path or list of Path
        The path to the folder containing the input videos or a list of video file paths.
    output_path : Path
        The path to the output video.
    date : str, optional
        The date to add to the video.
    arena : str, optional
        The arena to add to the video.
    keyword : str, optional
        The keyword to use to find the input videos.
    test_mode : bool, optional
        If True, runs the command in test mode.

    Returns
    -------
    None
    """
    # Specify the path to the font file
    fontfile = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

    # Check if source is a directory path or a list of file paths
    if isinstance(source, (str, Path)):
        input_folder = Path(source)
        # Get all video files from the input folder
        if keyword:
            input_files = list(input_folder.glob(f"*{keyword}*.mp4"))
            # Sort the input files by the number in their name
            input_files.sort(key=lambda f: int(f.stem.split("_")[1]))
        else:
            # If no keyword is provided, sort by numbers in file name
            input_files = sorted(
                list(input_folder.glob("*.mp4")),
                key=lambda f: int(re.findall(r"\d+", f.stem)[0]),
            )
    else:
        # If source is a list of file paths, use it directly
        input_files = source

    print(f"input_files: {input_files}")

    # Get the width and height of the first video
    width, height = get_video_size(input_files[0])

    # Calculate the number of videos
    num_videos = len(input_files)

    print(f"num_videos: {num_videos}")

    # Create the filter_complex argument for the ffmpeg command
    filter_complex = "".join(
        f"[{i}:v]transpose=2[s{i}];[s{i}]scale=-1:{width}[v{i}];"
        for i in range(num_videos)
    )
    hstack_inputs = "".join(f"[v{i}]" for i in range(num_videos))
    filter_complex += f"{hstack_inputs}hstack=inputs={num_videos}[v]"

    # Add padding and label if date and arena are provided
    if date is not None and arena is not None:
        filter_complex += f";[v]pad=iw:ih+50:0:50:black,drawtext=fontfile={fontfile}:text='{date} {arena}':fontsize=30:fontcolor=white:x=(w-text_w)/2:y=25[v]"

    print(f"filter_complex : {filter_complex}")

    # Create the ffmpeg command arguments
    ffmpeg_args = ["ffmpeg", "-y"]
    for input_file in input_files:
        ffmpeg_args.extend(["-i", str(input_file)])
    if test_mode:
        ffmpeg_args.append("-t")
        ffmpeg_args.append("10")
    ffmpeg_args.extend(
        ["-filter_complex", filter_complex, "-map", "[v]", str(output_path)]
    )
    # Run the ffmpeg command
    subprocess.run(ffmpeg_args)


def make_bundles(input_folder, output_folder, keyword=None, test_mode=False):
    # Convert input_folder and output_folder to Path objects
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    # Get all video files from the input folder
    video_files = sorted(input_folder.glob("*.mp4"))

    # Group the video files by date and arena
    video_groups = []
    for key, group in groupby(
        video_files, key=lambda f: (f.stem.split("_")[0], f.stem.split("_")[-3])
    ):
        video_groups.append(list(group))

    # Create a horizontal video for each group
    for group in video_groups:
        date, arena = (
            group[0].stem.split("_")[0],
            group[0].stem.split("_")[-3],
        )  # Get the date and arena from the first video in the group

        # Sort the videos in the group by corridor number
        group.sort(key=lambda f: int(f.stem.split("_")[-2].replace("corridor", "")))

        output_path = output_folder / f"bundle_{date}_{arena}.mp4"
        create_horizontal_video(group, output_path, date, arena, test_mode=test_mode)


# Example usage:
# create_horizontal_video(
#     input_folder="/mnt/labserver/DURRIEU_Matthias/Videos/240129_TNT_Fine/TNTxDDC",
#     output_path="/mnt/labserver/DURRIEU_Matthias/Videos/Genotype_grids/test_short.mp4",
#     test_mode=True,
#     # keyword="black_clip",
# )

make_bundles(
    input_folder=Path("/mnt/labserver/DURRIEU_Matthias/Videos/240129_TNT_Fine/TNTxDDC"),
    output_folder=Path("/mnt/labserver/DURRIEU_Matthias/Videos/Genotype_grids"),
    test_mode=True,
)

# TODO: Implement this as a more general function that can create both horizontal and grid videos without having to duplicate code.
