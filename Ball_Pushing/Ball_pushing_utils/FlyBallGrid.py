import math
import os
import subprocess
import json
from pathlib import Path
from operator import itemgetter
from itertools import groupby
import re
import time

FONT_FILE = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
VIDEO_EXT = "*.mp4"
BUNDLE_KEYWORD = "*bundle*.mp4"


def get_video_size(video: Path) -> tuple:
    """
    Returns the width and height of the video at the given path.

    Parameters
    ----------
    video : Path
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
            str(video),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Error running ffprobe on {video}: {result.stderr}")
        return None
    else:
        try:
            return tuple(map(int, result.stdout.strip().split(",")))
        except ValueError:
            print(f"Warning: Unable to get size of video {video}")
            return None

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
            list(input_folder.glob(VIDEO_EXT)),
            key=lambda f: int(re.findall(r"\d+", f.stem)[0]),
        )
        
    # Check if the video files are valid, meaning their size can be obtained
    valid_files = []
    for file in input_files:
        size = get_video_size(file)
        if size is not None:
            valid_files.append(file)
        else:
            print(f"Skipping video {file} due to invalid size")
    input_files = valid_files

    print(f"input_files: {input_files}")

    # Get the width and height of the first video
    width, height = get_video_size(input_files[0].as_posix())

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
    source, output_path, transpose=True, date=None, arena=None, spacing=None, test_mode=False
):
    """
    Stack videos horizontally and add date and arena labels if provided.

    Parameters
    ----------
    source : Path or list of Path
        The path to the folder containing the input videos or a list of video file paths.
    output_path : Path
        The path to the output video.
    transpose : bool, optional
        If True, transposes the videos.
    date : str, optional
        The date to add to the video.
    arena : str, optional
        The arena to add to the video.
    spacing : int, optional
        The spacing to add between the videos.
    test_mode : bool, optional
        If True, runs the command in test mode, which limits the output video to 10 seconds.

    Returns
    -------
    None

    Raises
    ------
    subprocess.CalledProcessError
        If the ffmpeg command fails.
    """

    # If source is a directory path, get all .mp4 files from the directory
    if isinstance(source, (str, Path)):
        input_folder = Path(source)
        input_files = list(input_folder.glob("*.mp4"))
        input_files.sort(key=lambda f: int(re.findall(r"\d+", f.stem)[0]))
    else:
        # If source is a list or tuple of file paths, use it directly
        input_files = [f for f in source if f.suffix == ".mp4"]
        input_files.sort(key=lambda f: int(re.findall(r"\d+", f.stem)[0]))
        
    print(f"input_files: {input_files}")

    # Get the width and height of the first video
    width, height = get_video_size(input_files[0].as_posix())

    # Calculate the number of videos
    num_videos = len(input_files)

    # Create the filter_complex argument for the ffmpeg command
    filter_complex = "".join(
        f"[{i}:v]{'transpose=2[s{i}];[s{i}]' if transpose else ''}scale=-1:{width}[v{i}];"
        + (f"[v{i}]pad=iw+{spacing}:ih[v{i}];" if spacing else "")
        for i in range(num_videos)
    )
    filter_complex += f"{''.join(f'[v{i}]' for i in range(num_videos))}hstack=inputs={num_videos}[v]"

    # Add padding and label if date and arena are provided
    if date and arena:
        filter_complex += f";[v]pad=iw:ih+50:0:50:black,drawtext=fontfile={FONT_FILE}:text='{date} {arena}':fontsize=30:fontcolor=white:x=(w-text_w)/2:y=25[v]"

    # Create the ffmpeg command arguments
    ffmpeg_args = [
        "ffmpeg",
        "-y",
        # "-loglevel",
        # "panic",
        # "-hwaccel",
        # "cuda",
    ]
    for input_file in input_files:
        ffmpeg_args.extend(["-i", str(input_file)])
        
    # ffmpeg_args.append("-c:v")
    # ffmpeg_args.append("h264_nvenc")

    if test_mode:
        ffmpeg_args.extend(["-t", "10"])
    ffmpeg_args.extend(
        ["-filter_complex", filter_complex, "-map", "[v]", str(output_path)]
    )

    # print(ffmpeg_args)

    # Run the ffmpeg command
    subprocess.run(ffmpeg_args)


def make_bundles(input_folder, output_folder, test_mode=False):
    """
    Groups videos by date and arena, and creates a horizontal video for each group.

    Parameters
    ----------
    input_folder : str or Path
        The path to the folder containing the input videos.
    output_folder : str or Path
        The path to the folder where the output videos will be saved.
    test_mode : bool, optional
        If True, runs the command in test mode, which limits the output video to 10 seconds.

    Returns
    -------
    None

    Raises
    ------
    subprocess.CalledProcessError
        If the ffmpeg command fails.
    """
    
    # Convert input_folder and output_folder to Path objects
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    # Get all video files from the input folder, excluding files smaller than 300 bytes.
    video_files = sorted(
        f for f in input_folder.glob(VIDEO_EXT) if f.stat().st_size > 300
    )

    # Group the video files by date, middle part, and arena
    video_groups = [
        list(group) for _, group in groupby(
            video_files, key=lambda f: (f.stem.split("_")[0], f.stem.split("_")[1:-3], f.stem.split("_")[-3])
        )
    ]

    # Create a horizontal video for each group
    for i, group in enumerate(video_groups):
        # Get the date, middle part, and arena from the first video in the group
        date, middle_part, arena = group[0].stem.split("_")[0], "_".join(group[0].stem.split("_")[1:-3]), group[0].stem.split("_")[-3]

        # Sort the videos in the group by corridor number
        group.sort(key=lambda f: int(f.stem.split("_")[-2].replace("corridor", "")))

        # Add an index to the output file name to differentiate between videos with the same date, middle part, and arena
        output_path = output_folder / f"bundle_{date}_{middle_part}_{arena}_{i+1}.mp4"
        create_horizontal_video(
            source=group,
            output_path=output_path,
            date=date,
            arena=arena,
            test_mode=test_mode,
            transpose=True,
        )

def assemble_bundles(input_folder, output_path, date=None, arena=None, test_mode=False):
    """
    Assembles video bundles into a single horizontal video.

    Parameters
    ----------
    input_folder : str or Path
        The path to the folder containing the input video bundles.
    output_path : str or Path
        The path to the output video.
    date : str, optional
        The date to add to the video.
    arena : str, optional
        The arena to add to the video.
    test_mode : bool, optional
        If True, runs the command in test mode, which limits the output video to 10 seconds.

    Returns
    -------
    None

    Raises
    ------
    subprocess.CalledProcessError
        If the ffmpeg command fails.
    """

    # Convert input_folder to Path object and get all bundle files
    bundle_files = sorted(
        [f for f in Path(input_folder).glob(BUNDLE_KEYWORD)],
        key=lambda f: (f.stem.split("_")[1], f.stem.split("_")[2])  # Sort by date and arena
    )

    # Create a horizontal video from the bundles without transposing them
    create_horizontal_video(
        source=bundle_files,
        output_path=output_path,
        date=date,
        arena=arena,
        spacing=10,
        test_mode=test_mode,
        transpose=False,
    )


def process_videos(input_folder, output_folder=None, output_path=None, test_mode=False):
    """
    Processes videos by making bundles, assembling them into a single video, and then removing the bundles.

    Parameters
    ----------
    input_folder : str or Path
        The path to the folder containing the input videos.
    output_folder : str or Path, optional
        The path to the folder where the output videos will be saved. If not provided, uses the input_folder.
    output_path : str or Path, optional
        The path to the output video. If not provided, creates a video in the input_folder with the name "{input_folder.stem}_All.mp4".
    test_mode : bool, optional
        If True, runs the command in test mode, which limits the output video to 10 seconds.

    Returns
    -------
    None

    Raises
    ------
    subprocess.CalledProcessError
        If the ffmpeg command fails.
    """

    print(f"processing videos in {input_folder}")
    # Convert input_folder, output_folder, and output_path to Path objects
    input_folder = Path(input_folder)
    output_folder = Path(output_folder) if output_folder else input_folder
    output_path = (
        Path(output_path)
        if output_path
        else output_folder / "processed" / f"{input_folder.stem}_All_Interactions.mp4"
    )

    # Create a Path object for the "processed" directory
    processed_folder = output_folder / "processed"

    # Check if the "processed" directory exists
    if processed_folder.exists():
        print(f"The 'processed' directory already exists in {output_folder}.")

    # Create the "processed" directory
    processed_folder.mkdir(parents=True, exist_ok=True)

    # Check if the output video file already exists
    if output_path.exists():
        print(f"The video file {output_path} already exists.")
        return

    # Step 1: Make bundles
    make_bundles(input_folder, output_folder, test_mode=test_mode)
    
    # Get the list of bundle files
    bundle_files = list(output_folder.glob(BUNDLE_KEYWORD))

    if len(bundle_files) == 1:
        # If there is only one bundle, rename it to the output file name
        bundle_files[0].rename(output_path)
    else:
        # If there are multiple bundles, assemble them into a single video
        assemble_bundles(output_folder, output_path, test_mode=test_mode)

    # Step 3: Remove the bundles
    for bundle_file in output_folder.glob(BUNDLE_KEYWORD):
        bundle_file.unlink()


# Example usage:
# create_horizontal_video(
#     input_folder="/mnt/labserver/DURRIEU_Matthias/Videos/240129_TNT_Fine/TNTxDDC",
#     output_path="/mnt/labserver/DURRIEU_Matthias/Videos/Genotype_grids/test_short.mp4",
#     test_mode=True,
#     # keyword="black_clip",
# )

# make_bundles(
#     input_folder=Path("/mnt/labserver/DURRIEU_Matthias/Videos/240129_TNT_Fine/TNTxDDC"),
#     output_folder=Path("/mnt/labserver/DURRIEU_Matthias/Videos/Genotype_grids"),
#     test_mode=True,
# )

# assemble_bundles(
#     input_folder="/mnt/labserver/DURRIEU_Matthias/Videos/Genotype_grids",
#     output_path="/mnt/labserver/DURRIEU_Matthias/Videos/Genotype_grids/240129_TNT_Fine_TNTxDDC.mp4",
#     test_mode=True,
# )

# process_videos(
#     input_folder=Path(
#         "/mnt/labserver/DURRIEU_Matthias/Videos/240129_TNT_Fine/TNTxZ1647"
#     ),
#     test_mode=True,
# )

# Find all folders in the input folder
VideoFolder = Path("/mnt/labserver/DURRIEU_Matthias/Videos/TNT_Fine_Annotated_True")

Folders = [f for f in VideoFolder.iterdir() if f.is_dir()]
print(Folders)

# For each folder, process the videos
for folder in Folders:
    process_videos(folder)
# TODO: Implement this as a more general function that can create both horizontal and grid videos without having to duplicate code.
# TODO: Implement hardwareacceleration for the ffmpeg command.