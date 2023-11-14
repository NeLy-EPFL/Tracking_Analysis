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


def write_clip(input_path, start_frame, end_frame, output_path, df):
    """
    Writes a clip from the input video to the output video.

    Parameters
    ----------
    input_path : str
        The path to the input video.
    start_frame : int
        The start frame of the clip.
    end_frame : int
        The end frame of the clip.
    output_path : str
        The path to the output video.

    Returns
    -------
    None
    """
    # Open the input video
    input_video = cv2.VideoCapture(input_path)
    fps = input_video.get(cv2.CAP_PROP_FPS)
    frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set the position of the input video to the start frame
    input_video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, 140))

    # Iterate over the frames of the input video
    for frame_number in range(start_frame, end_frame + 1):
        # Read the next frame from the input video
        ret, frame = input_video.read()
        if not ret:
            break

        # Get the position of the ball at the current frame
        row = df.loc[df["Frame"] == frame_number].iloc[0]
        yball_smooth = row["yball_smooth"]

        # Calculate the cropping region
        x_min = 0
        x_max = frame_width
        y_min = int(yball_smooth - 20)
        y_max = int(yball_smooth + 120)

        # Crop the frame
        cropped_frame = frame[y_min:y_max, x_min:x_max]

        # Resize the cropped frame to match the original frame size
        resized_frame = cv2.resize(cropped_frame, (frame_width, 140))

        # Write the resized frame to the output video
        output_video.write(resized_frame)

    # Release the input and output videos
    input_video.release()
    output_video.release()

def write_clip_wholecrop(input_path, start_frame, end_frame, output_path, df):
    """
    Writes a clip from the input video to the output video. 
    In this version, the video is only cropped once based on the min and max values of the yfly_smooth and yball_smooth columns.

    Parameters
    ----------
    input_path : str
        The path to the input video.
    start_frame : int
        The start frame of the clip.
    end_frame : int
        The end frame of the clip.
    output_path : str
        The path to the output video.

    Returns
    -------
    None
    """
    # Open the input video
    input_video = cv2.VideoCapture(input_path)
    fps = input_video.get(cv2.CAP_PROP_FPS)
    frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set the position of the input video to the start frame
    input_video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, 140))

    # Calculate the cropping region based on min and max values of 'yfly_smooth' and 'yball_smooth'
    y_min = int(df[['yfly_smooth', 'yball_smooth']].min().min() - 20)
    y_max = int(df[['yfly_smooth', 'yball_smooth']].max().max() + 120)

    # Iterate over the frames of the input video
    for frame_number in range(start_frame, end_frame + 1):
        # Read the next frame from the input video
        ret, frame = input_video.read()
        if not ret:
            break

        # Crop the frame
        cropped_frame = frame[y_min:y_max, :]

        # Calculate the padding size
        pad_size = (y_max - y_min) - frame_width

        # Add black padding to the left and right sides of the cropped frame
        padded_frame = cv2.copyMakeBorder(cropped_frame, 0, 0, pad_size // 2, pad_size - pad_size // 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # Resize the padded frame to match the original frame size
        resized_frame = cv2.resize(padded_frame, (frame_width, 140))

        # Write the resized frame to the output video
        output_video.write(resized_frame)

    # Release the input and output videos
    input_video.release()
    output_video.release()


# Example usage:
ballpath = Path(
    "/mnt/labserver/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Videos/230721_Feedingstate_4_PM_Videos_Tracked/arena5/corridor3/corridor3_tracked_ball.000_corridor3.analysis.h5"
)

flypath = Path(
    "/mnt/labserver/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Videos/230721_Feedingstate_4_PM_Videos_Tracked/arena5/corridor3/tracked_fly.000_corridor3.analysis.h5"
)

vidpath = Path(
    "/mnt/labserver/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Videos/230721_Feedingstate_4_PM_Videos_Tracked/arena5/corridor3/corridor3.mp4"
)

interaction_events = extract_interaction_events(ballpath, flypath)

OutFolder = Path("/mnt/labserver/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Grids/Newcrop")

for i, event in enumerate(interaction_events):
    start_frame = event["Frame"].min()
    end_frame = event["Frame"].max()
    output_path = OutFolder / f"clip_{i}.mp4"
    write_clip_wholecrop(vidpath.as_posix(), start_frame, end_frame, output_path.as_posix(), event)
