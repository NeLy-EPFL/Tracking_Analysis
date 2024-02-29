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
        input_files = sorted(list(input_folder.glob("*.mp4")), key=lambda f: int(re.findall(r'\d+', f.stem)[0]))

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


def create_horizontal_video(input_folder, output_path, test_mode=False):
    # Step 1: Get all the videos from the input folder
    input_folder = Path(input_folder)
    input_files = list(input_folder.glob("*.mp4"))

    # Step 2: Parse the date, arena, and corridor from the video names and make groups based on date and arena
    video_info = []
    for input_file in input_files:
        match = re.match(r"(\d{6})_.*_arena(\d+)_corridor(\d+)_.*", input_file.stem)
        if match:
            date, arena, corridor = match.groups()
            video_info.append((date, int(arena), int(corridor), input_file))
    video_info.sort()
    video_groups = [list(group) for _, group in groupby(video_info, itemgetter(0, 1))]
    
    # Step 3: Make bundle videos of each group and add a label on top of the bundle
    for i, group in enumerate(video_groups):
        date, arena, _, _ = group[0]
        label_video = input_folder / f"label_{date}_arena{arena}.mp4"

        # Sort the videos in the group by corridor
        group.sort(key=itemgetter(2))

        # Transpose and horizontally stack the videos in the group
        group_videos = [str(video) for _, _, _, video in group]
        
        print(group_videos)
        if len(group_videos) > 1:
            filter_complex = ";".join(
                [f"[{i}:v]transpose=2[v{i}]" for i in range(len(group_videos))]
            ) + ";" + "".join([f"[v{i}]" for i in range(len(group_videos))]) + f"hstack={len(group_videos)}"
        else:
            filter_complex = f"[0:v]transpose=2"

        ffmpeg_command = [
            "ffmpeg",
            *sum([["-i", video] for video in group_videos], []),
            "-filter_complex",
            filter_complex,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-y",  # Overwrite output files without asking
        ]
        if test_mode:
            ffmpeg_command.extend(["-ss", "0", "-t", "10"])
        ffmpeg_command.append(f"bundle_{i}.mp4")
            
        print (f"ffmpeg_command : {ffmpeg_command}")
        subprocess.run(ffmpeg_command)

        # Create a label video
        subprocess.run(
            [
                "ffmpeg",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=1280x720:d=1:r=25",
                "-vf",
                f"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='{date} Arena {arena}':fontsize=30:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2",
                "-c:v",
                "libx264",
                "-tune",
                "stillimage",
                "-pix_fmt",
                "yuv420p",
                "-y",  # Overwrite output files without asking
                str(label_video),
            ]
        )

        # Vertically stack the label video on top of the bundle
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(label_video),
                "-i",
                f"bundle_{i}.mp4",
                "-filter_complex",
                "[0:v][1:v]vstack",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-y",  # Overwrite output files without asking
                f"bundle_{i}.mp4",
            ]
        )

    # Step 4: Stack the bundles together
    bundle_files = " ".join([f"bundle_{i}.mp4" for i in range(len(video_groups))])
    subprocess.run(
        f"ffmpeg -i 'concat:{bundle_files}' -c copy {output_path}", shell=True
    )

    # Remove the temporary bundle files
    for i in range(len(video_groups)):
        os.remove(f"bundle_{i}.mp4")


# TODO: Implement this as a more general function that can create both horizontal and grid videos without having to duplicate code.

# Example usage:
create_horizontal_video(
    input_folder="/mnt/labserver/DURRIEU_Matthias/Videos/240129_TNT_Fine/TNTxDDC",
    output_path="/mnt/labserver/DURRIEU_Matthias/Videos/Genotype_grids/test_labeled.mp4",
    test_mode=True,
    # keyword="black_clip",
)
