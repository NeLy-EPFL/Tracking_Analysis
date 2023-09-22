import h5py
from scipy import signal
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, "../..")
# from Utilities.Utils import *
# from Utilities.Processing import *
import cv2
from datetime import timedelta
import platform
import json

import os

os.environ["IMAGEMAGICK_BINARY"] = "/usr/bin/convert"  # Replace with your actual path


def savgol_lowpass_filter(data, window_length, polyorder):
    # Apply the Savitzky-Golay filter
    y = signal.savgol_filter(data, window_length, polyorder)
    return y


def extract_coordinates(h5_file):
    with h5py.File(h5_file, "r") as f:
        locs = f["tracks"][:].T
        y = locs[:, :, 1, :].squeeze()
        x = locs[:, :, 0, :].squeeze()
    return x, y


def replace_nans_with_previous_value(arr):
    # Find the indices of the NaN values
    nan_indices = np.where(np.isnan(arr))

    # Replace the NaN values with the previous value
    for i in nan_indices[0]:
        arr[i] = arr[i - 1]


def extract_interaction_events(ballpath, flypath, Thresh=80, min_time=60):
    """
    Extracts the interaction events from the ball and fly paths.

    Parameters
    ----------
    ballpath : str
        The path to the ball path file.
        flypath : str
        The path to the fly path file.
        Thresh : int
        The threshold distance between the ball and fly.
        min_time : int
        The minimum duration of an interaction event.

        Returns
        -------
        interaction_events : list
        A list of DataFrames containing the interaction events.
    """
    xball, yball = extract_coordinates(ballpath.as_posix())
    xfly, yfly = extract_coordinates(flypath.as_posix())

    # Replace NaNs in yball
    replace_nans_with_previous_value(yball)

    # Replace NaNs in xball
    replace_nans_with_previous_value(xball)

    # Replace NaNs in yfly
    replace_nans_with_previous_value(yfly)

    # Replace NaNs in xfly
    replace_nans_with_previous_value(xfly)

    # Combine the yball and yfly arrays into a single 2D array
    data = np.stack((yball, yfly), axis=1)

    # Create a pandas DataFrame from the data
    df = pd.DataFrame(data, columns=["yball", "yfly"])

    df["yball_smooth"] = savgol_lowpass_filter(df["yball"], 221, 1)
    df["yfly_smooth"] = savgol_lowpass_filter(df["yfly"], 221, 1)
    df = df.assign(Frame=df.index + 1)
    df["time"] = df["Frame"] / 30

    # Compute the difference between the yball and yfly positions smoothed
    df["dist"] = df["yfly_smooth"] - df["yball_smooth"]

    # Locate where the distance is below the threshold
    df["close"] = df["dist"] < Thresh

    df = df.reset_index()

    # Find the start and end indices of streaks of True values in the 'close' column
    df["block"] = (df["close"].shift(1) != df["close"]).cumsum()
    events = (
        df[df["close"]]
        .groupby("block")
        .agg(start=("index", "min"), end=("index", "max"))
    )

    # Store the interaction events as separate DataFrames
    interaction_events = [
        df.loc[start:end]
        for start, end in events[["start", "end"]].itertuples(index=False)
    ]

    # remove events that are less than min_time frames long
    interaction_events = [
        event for event in interaction_events if len(event) >= min_time
    ]

    # event_times = [(df["time"].min(), df["time"].max()) for df in interaction_events]

    return interaction_events


# Get the DataFolder

if platform.system() == "Darwin":
    DataPath = Path(
        "/Volumes/Ramdya-Lab/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Videos"
    )
# Linux Datapath
if platform.system() == "Linux":
    DataPath = Path(
        "/mnt/labserver/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Videos"
    )

print(DataPath)
# Make a list of the folders I want to use
# For instance, I want to use the folders that have the "FeedingState" in the name

Folders = []
for folder in DataPath.iterdir():
    minfolder = str(folder).lower()
    # if "tnt" in minfolder and "tracked" in minfolder and "pm" in minfolder:
    if "feedingstate" in minfolder and "tracked" in minfolder:
        # Only use the folders that have 'feedingstate' and 'tracked' but not 'dark' in the name
        Folders.append(folder)

Folders


def process_videos(ballpath, flypath, vidpath, OutFolder, vidname, threshold=10):
    def check_yball_variation(event_df, threshold=10):
        yball_segment = event_df["yball_smooth"]
        variation = yball_segment.max() - yball_segment.min()
        return variation > threshold

    interaction_events = extract_interaction_events(ballpath, flypath)
    clips = []
    for i, event_df in enumerate(interaction_events):
        start_frame, end_frame = event_df["Frame"].min(), event_df["Frame"].max()
        start_time, end_time = (
            event_df["time"].min(),
            event_df["time"].max(),
        )  # Assuming 'time' is in seconds
        start_time_str = str(timedelta(seconds=int(start_time)))
        # Load the video
        cap = cv2.VideoCapture(str(vidpath))

        # Get the video's width, height, and frames per second (fps)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Be sure to use lower case

        clip_path = OutFolder.joinpath(f"output_{i}.mp4").as_posix()

        out = cv2.VideoWriter(
            clip_path, fourcc, fps, (height, width)
        )  # Note that width and height are swapped

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for _ in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break

            # Rotate frame 90 degrees clockwise
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # Write some Text
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"Event:{i+1} - start:{start_time_str}"
            font_scale = width / 150
            thickness = int(4 * font_scale)
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

            # Position the text at the top center of the frame
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = 25
            cv2.putText(
                frame,
                text,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

            # Check if yball value varies more than threshold
            if check_yball_variation(event_df):  # You need to implement this function
                # Add red dot to segment
                dot = np.zeros((10, 10, 3), dtype=np.uint8)
                dot[:, :, 0] = 0
                dot[:, :, 1] = 0
                dot[:, :, 2] = 255
                dot = cv2.resize(dot, (20, 20))

                # Position the dot right next to the text at the top of the frame
                dot_x = (
                    text_x + text_size[0] + 10
                )  # Position the dot right next to the text with a margin of 10

                # Adjusted position for dot_y to make it slightly higher
                dot_y_adjustment_factor = 1.2
                dot_y = (
                    text_y
                    - int(dot.shape[0] * dot_y_adjustment_factor)
                    + text_size[1] // 2
                )

                frame[dot_y : dot_y + dot.shape[0], dot_x : dot_x + dot.shape[1]] = dot

            # Write the frame into the output file
            out.write(frame)

        # Release everything when done
        cap.release()
        out.release()

        clips.append(clip_path)

    cv2.destroyAllWindows()

    # Define the codec and create a VideoWriter object for the final video
    out = cv2.VideoWriter(
        OutFolder.joinpath(f"{vidname}.mp4").as_posix(), fourcc, fps, (height, width)
    )  # Note that width and height are swapped because the clips were rotated

    # Assuming 'clips' is a list of paths to the clip files
    for clip_path in clips:
        cap = cv2.VideoCapture(clip_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Write the frame into the final output file
            out.write(frame)

        cap.release()

    # Release the final output file when done
    out.release()

    # Delete the clips
    for clip_path in clips:
        os.remove(clip_path)

    print(f"Finished processing {vidname}!")


SaveFolder = Path("/mnt/labserver/DURRIEU_Matthias/Videos/WildType_BallPushing_Events")

# Folders = [
#     Folders[0]
# ]  # Troubleshooting with only one folder, comment out to run the whole list

for folder in Folders:
    print(f"Processing {folder}...")
    # Read the metadata.json file
    with open(folder / "Metadata.json", "r") as f:
        metadata = json.load(f)
        variables = metadata["Variable"]
        metadata_dict = {}
        for var in variables:
            metadata_dict[var] = {}
            for arena in range(1, 10):
                arena_key = f"Arena{arena}"
                var_index = variables.index(var)
                metadata_dict[var][arena_key] = metadata[arena_key][var_index]

        # In the metadata_dict, make all they Arena subkeys lower case

        for var in variables:
            metadata_dict[var] = {k.lower(): v for k, v in metadata_dict[var].items()}
        print(metadata_dict)

        files = list(folder.glob("**/*.mp4"))
        # files = [
        #     files[0]
        # ]  # Troubleshooting with only one video, comment out to run the whole folder

    for file in files:
        print(file.name)
        # Get the arena and corridor numbers from the parent (corridor) and grandparent (arena) folder names
        arena = file.parent.parent.name
        # print(arena)
        corridor = file.parent.name

        # Get the Genotype and Dates from the metadata, arena should have a upper case first letter

        Genotype = metadata_dict["Genotype"][arena]
        print(f"Genotype: {Genotype} for arena {arena}")

        Date = metadata_dict["Date"][arena]
        # print(f"Date: {Date} for arena {arena}")

        Light = metadata_dict["Light"][arena]
        FeedingState = metadata_dict["FeedingState"][arena]
        Period = metadata_dict["Period"][arena]

        dir = file.parent

        # Define flypath as the *flytrack*.analysis.h5 file in the same folder as the video
        flypath = list(dir.glob("*flytrack*.analysis.h5"))[0]

        # flypath = file.parent / f"flytrack.000_{corridor}.analysis.h5"
        print(flypath.name)
        # Define ballpath as the *tracked*.analysis.h5 file in the same folder as the video
        ballpath = list(dir.glob("*tracked*.analysis.h5"))[0]
        print(ballpath.name)
        vidpath = file

        Dir = f"{Genotype}_{Date}_Light_{Light}_{FeedingState}_{Period}"

        # Define the output folder as a directory in SaveFolder with same name as Genotype. If it doesn't exist, create it.
        OutFolder = SaveFolder / Dir
        OutFolder.mkdir(exist_ok=True)

        vidname = f"{Genotype}_{Date}_Light_{Light}_{FeedingState}_{Period}_{arena}_{corridor}"

        # Check if the video has already been processed
        if not OutFolder.joinpath(f"{vidname}.mp4").exists():
            print(f"Processing {vidname}...")
            process_videos(ballpath, flypath, vidpath, OutFolder, vidname=vidname)

        else:
            print(f"{vidname} already exists! Skipping...")
