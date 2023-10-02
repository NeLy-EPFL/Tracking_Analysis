import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import traceback


sys.path.insert(0, "..")
sys.path.insert(0, "../..")
from Utilities.Utils import *
from Utilities.Processing import *

from datetime import timedelta


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


def extract_interaction_events(
    ballpath, flypath, Thresh=80, min_time=60, mark_in_df=False
):
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

    if mark_in_df:
        # Create a new column 'Event' and initialize it with None
        df["Event"] = None

        # For each event, assign 'EventN' to the rows that are part of the event
        for i, event in enumerate(interaction_events, start=1):
            df.loc[event.index, "Event"] = f"Event{i}"

        return df
    else:
        return interaction_events
