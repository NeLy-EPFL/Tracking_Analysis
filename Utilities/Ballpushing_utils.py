import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import traceback
import json
import datetime

sys.path.insert(0, "..")
sys.path.insert(0, "../..")
from Utilities.Utils import *
from Utilities.Processing import *

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


def generate_dataset(Folders, fly=True, ball=True, xvals=False, fps=30):
    """Generates a dataset from a list of folders containing videos, tracking files and metadata files


    Args:
        Folders (list): A list of folders containing videos, tracking files and metadata files
        fly (bool, optional): Whether to extract the fly coordinates. Defaults to True.
        ball (bool, optional): Whether to extract the ball coordinates. Defaults to True.
        xvals (bool, optional): Whether to extract the x coordinates. Defaults to False.
        fps (int, optional): The frame rate of the videos. Defaults to 30.

    Returns:
        Dataset (pandas dataframe): A dataframe containing the data from all the videos in the list of folders
    """

    Flycount = 0
    Dataset_list = []

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
                metadata_dict[var] = {
                    k.lower(): v for k, v in metadata_dict[var].items()
                }
            # print(metadata_dict)

            files = list(folder.glob("**/*.mp4"))

        for file in files:
            # print(file.name)
            # Get the arena and corridor numbers from the parent (corridor) and grandparent (arena) folder names
            arena = file.parent.parent.name
            # print(arena)
            corridor = file.parent.name

            start, end = np.load(file.parent / "coordinates.npy")

            dir = file.parent

            # Define flypath as the *flytrack*.analysis.h5 file in the same folder as the video
            try:
                flypath = list(dir.glob("*flytrack*.analysis.h5"))[0]
                # print(flypath.name)
            except IndexError:
                # print(f"No fly tracking file found for {file.name}, skipping...")

                continue

            # Define ballpath as the *tracked*.analysis.h5 file in the same folder as the video
            try:
                ballpath = list(dir.glob("*tracked*.analysis.h5"))[0]
                # print(ballpath.name)
            except IndexError:
                print(f"No ball tracking file found for {file.name}, skipping...")

                continue

            try:
                # Extract interaction events and mark them in the DataFrame

                data = get_coordinates(
                    ballpath, flypath, ball=ball, fly=fly, xvals=xvals
                )
                
                #print(data.head())
                # Apply savgol_lowpass_filter to each column that is not Frame or time
                # for col in data.columns:
                #     if col not in ["Frame", "time"]:
                #         data[f"{col}_smooth"] = savgol_lowpass_filter(data[col], 221, 1)
                    
                data["start"] = start
                data["end"] = end
                data["arena"] = arena
                data["corridor"] = corridor
                Flycount += 1
                data["Fly"] = f"Fly {Flycount}"
                
                if "Flipped" in folder.name:
                    print(f'Flipped video, flipping ball and fly y coordinates, flipping start and end.')
                    data["yball_smooth"] = -data["yball_smooth"]
                    data["yfly_smooth"] = -data["yfly_smooth"]
                    start = -start
                
                # Compute yball_relative relative to start
                data["yball_relative"] = abs(data["yball_smooth"] - data["start"])

                # Fill missing values using linear interpolation
                data["yball_relative"] = data["yball_relative"].interpolate(
                    method="linear"
                )

                # Add all the metadata categories to the DataFrame
                for var in variables:
                    data[var] = metadata_dict[var][arena]

                # Append the data to the all_data DataFrame
                Dataset_list.append(data)

            except Exception as e:
                error_message = str(e)
                traceback_message = traceback.format_exc()
                # print(f"Error processing video {vidname}: {error_message}")
                print(traceback_message)

    # Concatenate all dataframes in the list into a single dataframe
    Dataset = pd.concat(Dataset_list, ignore_index=True)

    return Dataset


def get_coordinates(ballpath=None, flypath=None, ball=True, fly=True, xvals=False):
    """Extracts the coordinates from the ball and fly paths.

    Parameters:
        ballpath (str): The path to the ball path file.
        flypath (str): The path to the fly path file.
        ball (bool): Whether to extract the ball coordinates.
        fly (bool): Whether to extract the fly coordinates.

    Returns:
        data (pd.DataFrame): The coordinates of the ball and fly.
    """
    data = []
    columns = []

    if ball:
        xball, yball = extract_coordinates(ballpath.as_posix())

        # Replace NaNs in yball
        replace_nans_with_previous_value(yball)

        # Replace NaNs in xball
        replace_nans_with_previous_value(xball)

        data.append(yball)
        columns.append("yball")

        if xvals:
            data.append(xball)
            columns.append("xball")

    if fly:
        xfly, yfly = extract_coordinates(flypath.as_posix())

        # Replace NaNs in yfly
        replace_nans_with_previous_value(yfly)

        # Replace NaNs in xfly
        replace_nans_with_previous_value(xfly)

        data.append(yfly)
        columns.append("yfly")

        if xvals:
            data.append(xfly)
            columns.append("xfly")

    # Combine the x and y arrays into a single 2D array
    data = np.stack(data, axis=1)

    # Convert the 2D array into a DataFrame
    data = pd.DataFrame(data, columns=columns)
    
    data = data.assign(Frame=data.index + 1)
    
    data["Frame"] = data["Frame"].astype(int)

    data["time"] = data["Frame"] / 30
    
    if ball:
        data["yball_smooth"] = savgol_lowpass_filter(data["yball"], 221, 1)
        if xvals:
            data["xball_smooth"] = savgol_lowpass_filter(data["xball"], 221, 1)
    if fly:
        data["yfly_smooth"] = savgol_lowpass_filter(data["yfly"], 221, 1)
        if xvals:
            data["xfly_smooth"] = savgol_lowpass_filter(data["xfly"], 221, 1)

    return data


def extract_interaction_events(
    source, Thresh=80, min_time=60, mark_in_df=False
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

    if isinstance(source, Path):
        print(f'Path: {source}')
        flypath = next(source.glob("*flytrack*.analysis.h5"))
        ballpath = next(source.glob("*tracked*.analysis.h5"))
        df = get_coordinates(flypath = flypath, ballpath = ballpath)
    # TODO : implement the code below    
    # elif isinstance(source, str):
    #     print(f'String: {source}')
    #     df = get_coordinates(flypath = source, xvals=True)
        
    elif isinstance(source, pd.DataFrame):
        print(f'DataFrame: {source.shape}')
        df = source
    else:
        raise TypeError("Invalid source format: source must be a pathlib Path, string or a pandas DataFrame")

    
    #TODO: fix event detection to detect within Fly, not across whole dataset.
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


def extract_pauses(source, min_time=200, threshold_y=0.05, threshold_x=0.05):
    """
    Extracts the pause events from the fly path.

    Parameters
    ----------
    source : pathlib Path, str or pandas DataFrame
        The path to the fly path file or a DataFrame containing the fly positions
    min_time : int
        The minimum duration of a pause event.
    threshold : float
        The threshold for the absolute difference in yfly_smooth values.

    Returns
    -------
    pause_events : list
        A list of DataFrames containing the pause events.
    """
    
    if isinstance(source, Path):
        print(f'Path: {source}')
        df = get_coordinates(flypath = source, ball=False, xvals=True)
        
    elif isinstance(source, str):
        print(f'String: {source}')
        df = get_coordinates(flypath = source, ball=False, xvals=True)
        
    elif isinstance(source, pd.DataFrame):
        print(f'DataFrame: {source.shape}')
        df = source
    else:
        raise TypeError("Invalid source format: source must be a pathlib Path, string or a pandas DataFrame")

    # Compute the absolute difference in yfly_smooth values
    df["yfly_diff"] = df["yfly_smooth"].diff().abs()
    df["xfly_diff"] = df["xfly_smooth"].diff().abs()

    # Identify periods where the difference is less than threshold for at least min_time frames
    df["Pausing"] = ((df["yfly_diff"] < threshold_y) #& df['xfly_diff'] < threshold_x
                     ).rolling(min_time).sum() == min_time

    # Replace NaN values with False
    df["Pausing"].fillna(False, inplace=True)

    # Create a new column 'PauseGroup' where change in 'Pausing' is detected
    df["PauseGroup"] = (df["Pausing"] != df["Pausing"].shift()).cumsum()

    # Filter rows where 'Pausing' is True
    pauses = df[df["Pausing"] == True]

    # Store the pause events as separate DataFrames
    pause_events = [group for _, group in pauses.groupby("PauseGroup")]

    # Group by 'PauseGroup' and get the first and last frame of each pause event
    pause_groups = pauses.groupby("PauseGroup")["time"].agg(["first", "last"])

    # Convert 'first' and 'last' columns to datetime format
    pause_groups['first'] = pd.to_datetime(pause_groups['first'], unit='s').dt.time
    pause_groups['last'] = pd.to_datetime(pause_groups['last'], unit='s').dt.time

    # Print the pause events
    print(pause_groups)

    return pause_events

#TODO: implement icecream