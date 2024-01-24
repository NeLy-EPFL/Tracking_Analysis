import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import holoviews as hv
from bokeh.models import HoverTool


from scipy.ndimage import median_filter, gaussian_filter
from pathlib import Path
import sys
import traceback
import json
import datetime
import subprocess
from collections import Counter
import pickle

import cv2
from moviepy.editor import VideoClip
from moviepy.editor import VideoFileClip
from moviepy.video.fx import all as vfx
import pygame


sys.path.insert(0, "..")
sys.path.insert(0, "../..")
from Utilities.Utils import *
from Utilities.Processing import *
from HoloviewsTemplates import hv_main

brain_regions_path = (
    "/mnt/labserver/DURRIEU_Matthias/Experimental_data/Region_map_240122.csv"
)


def save_object(obj, filename):
    """Save a custom object as a pickle file.

    Args:
        obj (object): The object to be saved.
        filename (Pathlib path): the path where to save the object. No need to add the .pkl extension.
    """

    # If the filename does not end with .pkl, add it
    if not filename.endswith(".pkl"):
        filename = filename.with_suffix(".pkl")

    with open(filename, "wb") as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    """Load a custom object from a pickle file.

    Args:
        filename (Pathlib path): the path to the object. No need to add the .pkl extension.
    """

    # If the filename does not end with .pkl, add it
    if not filename.endswith(".pkl"):
        filename = filename.with_suffix(".pkl")

    with open(filename, "rb") as input:
        obj = pickle.load(input)

    return obj


# def generate_dataset(Folders, fly=True, ball=True, xvals=False, fps=30, Events=None):
#     """Generates a dataset from a list of folders containing videos, tracking files and metadata files


#     Args:
#         Folders (list): A list of folders containing videos, tracking files and metadata files
#         fly (bool, optional): Whether to extract the fly coordinates. Defaults to True.
#         ball (bool, optional): Whether to extract the ball coordinates. Defaults to True.
#         xvals (bool, optional): Whether to extract the x coordinates. Defaults to False.
#         fps (int, optional): The frame rate of the videos. Defaults to 30.

#     Returns:
#         Dataset (pandas dataframe): A dataframe containing the data from all the videos in the list of folders
#     """

#     Flycount = 0
#     Dataset_list = []

#     for folder in Folders:
#         print(f"Processing {folder}...")
#         # Read the metadata.json file
#         with open(folder / "Metadata.json", "r") as f:
#             metadata = json.load(f)
#             variables = metadata["Variable"]
#             metadata_dict = {}
#             for var in variables:
#                 metadata_dict[var] = {}
#                 for arena in range(1, 10):
#                     arena_key = f"Arena{arena}"
#                     var_index = variables.index(var)
#                     metadata_dict[var][arena_key] = metadata[arena_key][var_index]

#             # In the metadata_dict, make all they Arena subkeys lower case

#             for var in variables:
#                 metadata_dict[var] = {
#                     k.lower(): v for k, v in metadata_dict[var].items()
#                 }
#             # print(metadata_dict)

#             files = list(folder.glob("**/*.mp4"))

#         for file in files:
#             # print(file.name)
#             # Get the arena and corridor numbers from the parent (corridor) and grandparent (arena) folder names
#             arena = file.parent.parent.name
#             # print(arena)
#             corridor = file.parent.name

#             start, end = np.load(file.parent / "coordinates.npy")

#             dir = file.parent

#             # Define flypath as the *tracked_fly*.analysis.h5 file in the same folder as the video
#             try:
#                 flypath = list(dir.glob("*tracked_fly*.analysis.h5"))[0]
#                 # print(flypath.name)
#             except IndexError:
#                 # print(f"No fly tracking file found for {file.name}, skipping...")

#                 continue

#             # Define ballpath as the *tracked*.analysis.h5 file in the same folder as the video
#             try:
#                 ballpath = list(dir.glob("*tracked*.analysis.h5"))[0]
#                 # print(ballpath.name)
#             except IndexError:
#                 print(f"No ball tracking file found for {file.name}, skipping...")

#                 continue

#             try:
#                 # Extract interaction events and mark them in the DataFrame

#                 data = get_coordinates(
#                     ballpath, flypath, ball=ball, fly=fly, xvals=xvals
#                 )

#                 # print(data.head())
#                 # Apply savgol_lowpass_filter to each column that is not Frame or time
#                 # for col in data.columns:
#                 #     if col not in ["Frame", "time"]:
#                 #         data[f"{col}_smooth"] = savgol_lowpass_filter(data[col], 221, 1)

#                 data["start"] = start
#                 data["end"] = end
#                 data["arena"] = arena
#                 data["corridor"] = corridor
#                 Flycount += 1
#                 data["Fly"] = f"Fly {Flycount}"

#                 if "Flipped" in folder.name:
#                     # print(
#                     #     f"Flipped video, flipping ball and fly y coordinates, flipping start and end."
#                     # )
#                     data["yball_smooth"] = -data["yball_smooth"]
#                     data["yfly_smooth"] = -data["yfly_smooth"]
#                     # start = -start

#                 # Compute yball_relative relative to start
#                 data["yball_relative"] = abs(data["yball_smooth"] - data["start"])

#                 # Fill missing values using linear interpolation
#                 data["yball_relative"] = data["yball_relative"].interpolate(
#                     method="linear"
#                 )

#                 # Add all the metadata categories to the DataFrame
#                 for var in variables:
#                     data[var] = metadata_dict[var][arena]

#                 # Append the data to the all_data DataFrame
#                 if Events == "interactions":
#                     # Compute interaction events for all data
#                     interaction_events = find_interaction_events(data)

#                     # Assign an event number to each event
#                     for i, (start_time, end_time) in enumerate(
#                         interaction_events, start=1
#                     ):
#                         data.loc[
#                             (data.Frame >= start_time) & (data.Frame <= end_time),
#                             "Event",
#                         ] = i

#                 Dataset_list.append(data)

#             except Exception as e:
#                 error_message = str(e)
#                 traceback_message = traceback.format_exc()
#                 # print(f"Error processing video {vidname}: {error_message}")
#                 print(traceback_message)

#     # Concatenate all dataframes in the list into a single dataframe
#     Dataset = pd.concat(Dataset_list, ignore_index=True).reset_index()

#     return Dataset


# def get_coordinates(ballpath=None, flypath=None, ball=True, fly=True, xvals=True):
#     """Extracts the coordinates from the ball and fly paths.

#     Parameters:
#         ballpath (str): The path to the ball path file.
#         flypath (str): The path to the fly path file.
#         ball (bool): Whether to extract the ball coordinates.
#         fly (bool): Whether to extract the fly coordinates.

#     Returns:
#         data (pd.DataFrame): The coordinates of the ball and fly.
#     """
#     data = []
#     columns = []

#     if ball:
#         xball, yball = extract_coordinates(ballpath.as_posix())

#         # Replace NaNs in yball
#         replace_nans_with_previous_value(yball)

#         # Replace NaNs in xball
#         replace_nans_with_previous_value(xball)

#         data.append(yball)
#         columns.append("yball")

#         if xvals:
#             data.append(xball)
#             columns.append("xball")

#     if fly:
#         xfly, yfly = extract_coordinates(flypath.as_posix())

#         # Replace NaNs in yfly
#         replace_nans_with_previous_value(yfly)

#         # Replace NaNs in xfly
#         replace_nans_with_previous_value(xfly)

#         data.append(yfly)
#         columns.append("yfly")

#         if xvals:
#             data.append(xfly)
#             columns.append("xfly")

#     # Combine the x and y arrays into a single 2D array
#     data = np.stack(data, axis=1)

#     # Convert the 2D array into a DataFrame
#     data = pd.DataFrame(data, columns=columns)

#     data = data.assign(Frame=data.index + 1)

#     data["Frame"] = data["Frame"].astype(int)

#     data["time"] = data["Frame"] / 30

#     if ball:
#         data["yball_smooth"] = savgol_lowpass_filter(data["yball"], 221, 1)
#         if xvals:
#             data["xball_smooth"] = savgol_lowpass_filter(data["xball"], 221, 1)
#     if fly:
#         data["yfly_smooth"] = savgol_lowpass_filter(data["yfly"], 221, 1)
#         if xvals:
#             data["xfly_smooth"] = savgol_lowpass_filter(data["xfly"], 221, 1)

#     return data


# def extract_interaction_events(source, Thresh=80, min_time=60, as_df=False):
#     if isinstance(source, Path):
#         print(f"Path: {source}")
#         flypath = next(source.glob("*tracked_fly*.analysis.h5"))
#         ballpath = next(source.glob("*tracked*.analysis.h5"))
#         df = get_coordinates(flypath=flypath, ballpath=ballpath)

#     elif isinstance(source, pd.DataFrame):
#         print(f"DataFrame: {source.shape}")
#         df = source

#     else:
#         raise TypeError(
#             "Invalid source format: source must be a pathlib Path, string or a pandas DataFrame"
#         )

#     # Create a new column 'Event' and initialize it with None
#     df.loc[:, "Event"] = None

#     # Compute interaction events for all data
#     interaction_events = find_interaction_events(df, Thresh, min_time)

#     # Assign an event number to each event
#     for i, (start_time, end_time) in enumerate(interaction_events, start=1):
#         df.loc[(df.index >= start_time) & (df.index <= end_time), "Event"] = i

#     if "Fly" in df.columns:
#         # Compute the maximum event number for each fly
#         max_event_per_fly = (
#             df.groupby("Fly")["Event"].max().shift(fill_value=0).cumsum()
#         )

#         # Adjust event numbers for each fly
#         df["Event"] -= df["Fly"].map(max_event_per_fly)

#     else:
#         # Compute interaction events for all data
#         interaction_events = find_interaction_events(df, Thresh, min_time)

#         # Assign an event number to each event
#         for i, (start_time, end_time) in enumerate(interaction_events, start=1):
#             df.loc[(df.index >= start_time) & (df.index <= end_time), "Event"] = i

#     if as_df:
#         return df
#     else:
#         return interaction_events


# def find_interaction_events(df, Thresh=80, min_time=60):
#     df.loc[:, "dist"] = df.loc[:, "yfly_smooth"] - df.loc[:, "yball_smooth"]
#     df.loc[:, "close"] = df.loc[:, "dist"] < Thresh
#     df.loc[:, "block"] = (df.loc[:, "close"].shift(1) != df.loc[:, "close"]).cumsum()
#     events = (
#         df[df["close"]]
#         .groupby("block")
#         .agg(start=("Frame", "min"), end=("Frame", "max"))
#     )
#     interaction_events = [
#         (start, end) for start, end in events[["start", "end"]].itertuples(index=False)
#     ]
#     interaction_events = [
#         event for event in interaction_events if event[1] - event[0] >= min_time
#     ]
#     return interaction_events


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
        print(f"Path: {source}")
        df = get_coordinates(flypath=source, ball=False, xvals=True)

    elif isinstance(source, str):
        print(f"String: {source}")
        df = get_coordinates(flypath=source, ball=False, xvals=True)

    elif isinstance(source, pd.DataFrame):
        print(f"DataFrame: {source.shape}")
        df = source
    else:
        raise TypeError(
            "Invalid source format: source must be a pathlib Path, string or a pandas DataFrame"
        )

    # Compute the absolute difference in yfly_smooth values
    df["yfly_diff"] = df["yfly_smooth"].diff().abs()
    df["xfly_diff"] = df["xfly_smooth"].diff().abs()

    # Identify periods where the difference is less than threshold for at least min_time frames
    df["Pausing"] = (
        (df["yfly_diff"] < threshold_y)  # & df['xfly_diff'] < threshold_x
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
    pause_groups["first"] = pd.to_datetime(pause_groups["first"], unit="s").dt.time
    pause_groups["last"] = pd.to_datetime(pause_groups["last"], unit="s").dt.time

    # Print the pause events
    print(pause_groups)

    return pause_events


# TODO: implement icecream


def filter_experiments(source, **criteria):
    """Generates a list of Experiment objects based on criteria.

    Args:
        source (list): A list of flies, experiments or folders to create Experiment objects from.
        criteria (dict): A dictionary of criteria to filter the experiments.

    Returns:
        list: A list of Experiment objects that match the criteria.
    """

    flies = []

    # If the source is a list of flies, check directly for the criteria in flies
    if isinstance(source[0], Fly):
        for fly in source:
            if all(
                fly.arena_metadata.get(key) == value for key, value in criteria.items()
            ):
                flies.append(fly)

    else:
        if isinstance(source[0], Experiment):
            Exps = source

        else:
            # Create Experiment objects from the folders
            Exps = [Experiment(f) for f in source]

        # Get a list of flies based on the criteria
        for exp in Exps:
            for fly in exp.flies:
                if all(
                    fly.arena_metadata.get(key) == value
                    for key, value in criteria.items()
                ):
                    flies.append(fly)

    return flies


class Fly:
    """
    A class for a single fly. This represents a folder containing a video, associated tracking files, and metadata files. It is usually contained in an Experiment object, and inherits the Experiment object's metadata.
    """

    def __init__(self, directory, experiment=None):
        """
        Initialize a Fly object.

        Args:
            directory (Path): The path to the fly directory.
            experiment (Experiment, optional): An optional Experiment object. If not provided,
                                               an Experiment object will be created based on the parent directory of the given directory.

        Attributes:
            directory (Path): The path to the fly directory.
            experiment (Experiment): The Experiment object associated with the Fly.
            arena (str): The name of the parent directory of the fly directory.
            corridor (str): The name of the fly directory.
            name (str): A string combining the name of the experiment directory, the arena, and the corridor.
            arena_metadata (dict): Metadata for the arena, obtained by calling the get_arena_metadata method.
            video (Path): The path to the video file in the fly directory.
            flytrack (Path): The path to the fly tracking file in the fly directory. If not found, a message is printed and this attribute is not set.
            balltrack (Path): The path to the ball tracking file in the fly directory. If not found, a message is printed and this attribute is not set.
            flyball_positions (DataFrame): The coordinates of the fly and the ball, obtained by calling the get_coordinates function with the flytrack and balltrack paths.
        """

        self.directory = Path(directory)
        self.experiment = (
            experiment
            if experiment is not None
            else Experiment(self.directory.parent.parent)
        )
        self.arena = self.directory.parent.name
        self.corridor = self.directory.name
        self.name = f"{self.experiment.directory.name}_{self.arena}_{self.corridor}"
        self.arena_metadata = self.get_arena_metadata()
        # For each value in the arena metadata, add it as an attribute of the fly
        for var, data in self.arena_metadata.items():
            setattr(self, var, data)

        try:
            self.video = list(self.directory.glob(f"{self.corridor}.mp4"))[0]
        except IndexError:
            raise FileNotFoundError(f"No video found for {self.name}.")

        try:
            self.flytrack = list(directory.glob("*tracked_fly*.analysis.h5"))[0]
            # print(flypath.name)
        except IndexError:
            self.flytrack = None
            # print(f"No fly tracking file found for {self.name}, skipping...")

        try:
            self.balltrack = list(directory.glob("*tracked_ball*.analysis.h5"))[0]
            # print(ballpath.name)
        except IndexError:
            self.balltrack = None
            # print(f"No ball tracking file found for {self.name}, skipping...")

        # Check if the coordinates.npy file exists in the fly directory

        if not (self.directory / "coordinates.npy").exists():
            # Run the detect_boundaries function on the Fly associated experiment to generate the coordinates.npy file for all flies in the experiment
            print(
                f"No boundaries found. Generating coordinates.npy file for {self.experiment.directory}..."
            )
            self.detect_boundaries()

        self.start, self.end = np.load(self.directory / "coordinates.npy")

        # Compute distance between fly and ball

        if self.flytrack is not None and self.balltrack is not None:
            self.flyball_positions = self.get_coordinates(self.balltrack, self.flytrack)
        else:
            self.flyball_positions = None

    def __str__(self):
        # Get the genotype from the metadata
        genotype = self.arena_metadata["Genotype"]

        return f"Fly: {self.name}\nArena: {self.arena}\nCorridor: {self.corridor}\nVideo: {self.video}\nFlytrack: {self.flytrack}\nBalltrack: {self.balltrack}\nGenotype: {genotype}"

    def __repr__(self):
        return f"Fly({self.directory})"

    def get_arena_metadata(self):
        """
        Retrieve the metadata for the Fly object's arena.

        This method looks up the arena's metadata in the experiment's metadata dictionary.
        The arena's name is converted to lowercase and used as the key to find the corresponding metadata.

        Returns:
            dict: A dictionary containing the metadata for the arena. The keys are the metadata variable names and the values are the corresponding metadata values. If no metadata is found for the arena, an empty dictionary is returned.
        """
        # Get the metadata for this fly's arena
        arena_key = self.arena.lower()
        return {
            var: data[arena_key]
            for var, data in self.experiment.metadata.items()
            if arena_key in data
        }

    def display_metadata(self):
        """
        Print the metadata for the Fly object's arena.

        This method iterates over the arena's metadata dictionary and prints each key-value pair.

        Prints:
            str: The metadata variable name and its corresponding value, formatted as 'variable: value'.
        """
        # Print the metadata for this fly's arena
        for var, data in self.arena_metadata.items():
            print(f"{var}: {data}")

    def detect_boundaries(self, threshold=100):
        """Detects the start and end of the corridor in the video. This is later used to compute the relative distance of the fly from the start of the corridor.

        Args:
            threshold (int, optional): the pixel value threshold to used for the thresholding operation. Defaults to 100. Change value if boundaries are not correctly detected.

        Returns:
            frame (np.array): the first frame of the video.
            min_row (int): the index of the minimum value in the thresholded summed pixel values.
        """

        video_file = self.video

        if not video_file.exists():
            print(f"Error: Video file {video_file} does not exist")
            return None, None

        # open the first frame of the video
        cap = cv2.VideoCapture(str(video_file))
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"Error: Could not read frame from video {video_file}")
            return None, None
        elif frame is None:
            print(f"Error: Frame is None for video {video_file}")
            return None, None

        # Convert to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply a median filter to smooth out noise and small variations
        frame = median_filter(frame, size=3)

        # Apply a Gaussian filter to smooth out noise and small variations
        frame = gaussian_filter(frame, sigma=1)

        # Compute the summed pixel values and apply a threshold
        summed_pixel_values = frame.sum(axis=1)
        summed_pixel_values[summed_pixel_values < threshold] = 0

        # Find the index of the minimum value in the thresholded summed pixel values
        min_row = np.argmin(summed_pixel_values)

        # Save a .npy file with the start and end coordinates in the video folder
        np.save(video_file.parent / "coordinates.npy", [min_row - 30, min_row - 320])

        return frame, min_row

    def get_coordinates(self, ball=True, fly=True, xvals=True):
        """Extracts the coordinates from the ball and fly h5 sleap data.

        Parameters:
            ball (bool): Whether to extract the ball coordinates.
            fly (bool): Whether to extract the fly coordinates.
            xvals (bool): Whether to extract the x coordinates.

        Returns:
            data (pd.DataFrame): The coordinates of the ball and fly.
        """
        data = []
        columns = []

        if ball:
            xball, yball = extract_coordinates(self.balltrack.as_posix())

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
            xfly, yfly = extract_coordinates(self.flytrack.as_posix())

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

        data["time"] = data["Frame"] / self.experiment.fps

        if ball:
            data["yball_smooth"] = savgol_lowpass_filter(data["yball"], 221, 1)
            if xvals:
                data["xball_smooth"] = savgol_lowpass_filter(data["xball"], 221, 1)
        if fly:
            data["yfly_smooth"] = savgol_lowpass_filter(data["yfly"], 221, 1)
            if xvals:
                data["xfly_smooth"] = savgol_lowpass_filter(data["xfly"], 221, 1)

        if self.start:
            data["yfly_relative"] = abs(data["yfly_smooth"] - self.start)

            # Fill missing values using linear interpolation
            data["yfly_relative"] = data["yfly_relative"].interpolate(method="linear")

            data["yball_relative"] = abs(data["yball_smooth"] - self.start)

            data["yball_relative"] = data["yball_relative"].interpolate(method="linear")

        return data

    def find_interaction_events(
        self,
        gap_between_events=4,
        event_min_length=2,
        thresh=[0, 70],
        omit_events=None,
        plot_signals=False,
        signal_name="",
    ):
        """
        This function finds events in a signal derived from the flyball_positions attribute based on certain criteria.

        Parameters:
        gap_between_events (int): The minimum gap required between two events, expressed in seconds.
        event_min_length (int): The minimum length of an event, expressed in seconds.
        thresh (list): The lower and upper limit values (in pixels) for the signal to be considered an event.
        omit_events (list, optional): A range of events to omit. Defaults to None.
        plot_signals (bool, optional): Whether to plot the signals or not. Defaults to False.
        signal_name (str, optional): The name of the signal. Defaults to "".

        Returns:
        list: A list of events found in the signal. Each event is a list containing the start frame, end frame and duration of the event.
        """

        # Convert the gap between events and the minimum event length from seconds to frames
        gap_between_events = gap_between_events * self.experiment.fps
        event_min_length = event_min_length * self.experiment.fps

        distance = (
            self.flyball_positions.loc[:, "yfly_smooth"]
            - self.flyball_positions.loc[:, "yball_smooth"]
        ).values

        # Initialize the list of events
        events = []

        # Find all frames where the signal is within the limit values
        all_frames_above_lim = np.where(
            (np.array(distance) > thresh[0]) & (np.array(distance) < thresh[1])
        )[0]

        # If no frames are found within the limit values, return an empty list
        if len(all_frames_above_lim) == 0:
            if plot_signals:
                print(f"Any point is between {thresh[0]} and {thresh[1]}")
                plt.plot(signal, label=f"{signal_name}-filtered")
                plt.legend()
                plt.show()
            return events

        # Find the distance between consecutive frames
        distance_betw_frames = np.diff(all_frames_above_lim)

        # Find the points where the distance between frames is greater than the gap between events
        split_points = np.where(distance_betw_frames > gap_between_events)[0]

        # Add the first and last points to the split points
        split_points = np.insert(split_points, 0, -1)
        split_points = np.append(split_points, len(all_frames_above_lim) - 1)

        # Plot the signal if required
        if plot_signals:
            limit_value = thresh[0] if thresh[1] == np.inf else thresh[1]
            print(all_frames_above_lim[split_points])
            plt.plot(signal, label=f"{signal_name}-filtered")

        # Iterate over the split points to find events
        for f in range(0, len(split_points) - 1):
            # If the gap between two split points is less than 2, skip to the next iteration
            if split_points[f + 1] - split_points[f] < 2:
                continue

            # Define the start and end of the region of interest (ROI)
            start_roi = all_frames_above_lim[split_points[f] + 1]
            end_roi = all_frames_above_lim[split_points[f + 1]]

            # If there are events to omit and the start of the ROI is within these events, adjust the start of the ROI
            if omit_events:
                if (
                    start_roi >= omit_events[0]
                    and start_roi < omit_events[1]
                    and end_roi < omit_events[1]
                ):
                    continue
                elif (
                    start_roi >= omit_events[0]
                    and start_roi < omit_events[1]
                    and end_roi > omit_events[1]
                ):
                    start_roi = int(omit_events[1])

            # Calculate the duration of the event
            duration = end_roi - start_roi

            # Calculate the mean and median of the signal within the ROI
            mean_signal = np.mean(np.array(distance[start_roi:end_roi]))
            median_signal = np.median(np.array(distance[start_roi:end_roi]))

            # Calculate the proportion of the signal within the ROI that is within the limit values
            signal_within_limits = len(
                np.where(
                    (np.array(distance[start_roi:end_roi]) > thresh[0])
                    & (np.array(distance[start_roi:end_roi]) < thresh[1])
                )[0]
            ) / len(np.array(distance[start_roi:end_roi]))

            # If the duration of the event is greater than the minimum length and more than 75% of the signal is within the limit values, add the event to the list
            if duration > event_min_length and signal_within_limits > 0.75:
                events.append([start_roi, end_roi, duration])
                if plot_signals:
                    print(
                        start_roi,
                        end_roi,
                        duration,
                        mean_signal,
                        median_signal,
                        signal_within_limits,
                    )
                    plt.plot(start_roi, limit_value, "go")
                    plt.plot(end_roi, limit_value, "rx")

        # Plot the limit value if required
        if plot_signals:
            plt.plot([0, len(distance)], [limit_value, limit_value], "c-")
            plt.legend()
            plt.show()

        # Return the list of events
        return events

    def annotate_events(self):
        """Creates a new column in the flyball_positions DataFrame containing the event number for each frame. If no event is found for a frame, the value in the column is set to None."""

        events = self.find_interaction_events()

        self.flyball_positions["event"] = None

        for i, event in enumerate(events, start=1):
            start, end = event[0], event[1]

            self.flyball_positions.loc[start:end, "event"] = i

    def check_yball_variation(self, event, threshold=10):
        """
        Check if the variation in the 'yball_smooth' value during an event exceeds a given threshold.

        This method extracts the 'yball_smooth' values for the duration of the event from the 'flyball_positions' DataFrame. It then calculates the variation as the difference between the maximum and minimum 'yball_smooth' values. If this variation exceeds the threshold, the method returns True; otherwise, it returns False.

        Args:
            event (list): A list containing the start and end indices of the event in the 'flyball_positions' DataFrame.
            threshold (int, optional): The minimum variation (in pixels) required for the method to return True. Defaults to 10.

        Returns:
            bool: True if the variation in 'yball_smooth' during the event exceeds the threshold, False otherwise.
        """
        # Get the yball_smooth segment corresponding to an event
        yball_event = self.flyball_positions.loc[event[0] : event[1], "yball_smooth"]

        variation = yball_event.max() - yball_event.min()

        return variation > threshold

    def generate_clip(self, event, outpath, fps, width, height):
        """
        Generate a video clip for a given event.

        This method creates a video clip from the original video for the duration of the event. It also adds text to each frame indicating the event number and start time. If the 'yball_smooth' value varies more than a certain threshold during the event, a red dot is added to the frame.

        Args:
            event (list): A list containing the start and end indices of the event in the 'flyball_positions' DataFrame.
            outpath (Path): The directory where the output video clip should be saved.
            fps (int): The frames per second of the original video.
            width (int): The width of the output video frames.
            height (int): The height of the output video frames.

        Returns:
            str: The path to the output video clip.
        """
        start_frame, end_frame = event[0], event[1]
        cap = cv2.VideoCapture(str(self.video))
        try:
            start_time = start_frame / fps
            start_time_str = str(datetime.timedelta(seconds=int(start_time)))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            # Get the index of the event in the list to apply it to the output file name
            event_index = self.find_interaction_events().index(event)

            clip_path = outpath.joinpath(f"output_{event_index}.mp4").as_posix()
            out = cv2.VideoWriter(clip_path, fourcc, fps, (height, width))
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                for _ in range(start_frame, end_frame):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                    # Write some Text
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text = f"Event:{event_index+1} - start:{start_time_str}"
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
                    if self.check_yball_variation(
                        event
                    ):  # You need to implement this function
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

                        frame[
                            dot_y : dot_y + dot.shape[0], dot_x : dot_x + dot.shape[1]
                        ] = dot

                    # Write the frame into the output file
                    out.write(frame)

            # Release everything when done
            finally:
                out.release()
        finally:
            cap.release()
        return clip_path

    def concatenate_clips(self, clips, outpath, fps, width, height, vidname):
        """
        Concatenate multiple video clips into a single video.

        This method takes a list of video clip paths, reads each clip frame by frame, and writes the frames into a new video file. The new video file is saved in the specified output directory with the specified name.

        Args:
            clips (list): A list of paths to the video clips to be concatenated.
            outpath (Path): The directory where the output video should be saved.
            fps (int): The frames per second for the output video.
            width (int): The width of the output video frames.
            height (int): The height of the output video frames.
            vidname (str): The name of the output video file (without the extension).

        """
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            outpath.joinpath(f"{vidname}.mp4").as_posix(), fourcc, fps, (height, width)
        )
        try:
            for clip_path in clips:
                cap = cv2.VideoCapture(clip_path)
                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        out.write(frame)
                finally:
                    cap.release()
        finally:
            out.release()

    def generate_interactions_video(self, outpath=None):
        """
        Generate a video of the fly's interactions with the ball.

        This method detects interaction events, generates a video clip for each event, concatenates the clips into a single video, and saves the video in the specified output directory. The video is named after the fly's name and genotype. If the genotype is not defined, 'undefined' is used instead. After the video is created, the individual clips are deleted.

        Args:
            outpath (Path, optional): The directory where the output video should be saved. If None, the video is saved in the fly's directory. Defaults to None.
        """

        if outpath is None:
            outpath = self.directory
        events = self.find_interaction_events()
        clips = []

        cap = cv2.VideoCapture(str(self.video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        vidname = f"{self.name}_{self.Genotype if self.Genotype else 'undefined'}"

        for i, event in enumerate(events):
            clip_path = self.generate_clip(event, outpath, fps, width, height)
            clips.append(clip_path)
        self.concatenate_clips(clips, outpath, fps, width, height, vidname)
        for clip_path in clips:
            os.remove(clip_path)
        print(f"Finished processing {vidname}!")

    def draw_circles(self, frame, flyball_coordinates):
        """
        Draw circles at the positions of the fly and ball in a video frame.

        Parameters:
        frame (numpy.ndarray): The video frame.
        fly_coordinates (pandas.Series): The tracking data for the fly.
        ball_coordinates (pandas.Series): The tracking data for the ball.
        frame_number (int): The number of the frame.

        Returns:
        numpy.ndarray: The video frame with the circles.
        """

        # Extract the x and y coordinates from the pandas Series
        fly_pos = tuple(
            map(int, [flyball_coordinates["xfly"], flyball_coordinates["yfly"]])
        )
        ball_pos = tuple(
            map(int, [flyball_coordinates["xball"], flyball_coordinates["yball"]])
        )

        # Draw a circle at each position
        cv2.circle(frame, fly_pos, 10, (0, 0, 255), -1)
        cv2.circle(frame, ball_pos, 10, (255, 0, 0), -1)

        return frame

    def generate_preview(
        self, speed=60.0, save=False, preview=False, output_path=None, tracks=True
    ):
        """
        Generate an accelerated version of the video using moviepy.

        This method uses the moviepy library to speed up the video, add circles at the positions of the fly and ball in each frame if tracking data is provided, and either save the resulting video or preview it using Pygame.

        Parameters:
        speed (float): The speedup factor. For example, 2.0 will double the speed of the video.
        save (bool, optional): Whether to save the sped up video. If True, the video is saved. If False, the video is previewed using Pygame. Defaults to False.
        output_path (str, optional): The path to save the sped up video. If not provided and save is True, a default path will be used. Defaults to None.
        tracks (dict, optional): A dictionary containing the tracking data for the fly and ball. Each key should be a string ('fly' or 'ball') and each value should be a numpy array with the x and y coordinates for each frame. Defaults to None.
        """

        if save and output_path is None:
            # Use the default output path
            output_path = (
                get_labserver()
                / "Videos"
                / "Previews"
                / f"{self.name}_{self.Genotype if self.Genotype else 'undefined'}_x{speed}.mp4"
            )

        # Load the video file
        clip = VideoFileClip(self.video.as_posix())

        # If tracks is True, add circles to the video

        # Create a new video clip with the draw_circles function applied to each frame
        if tracks:
            # Check if tracking data is available
            if self.flyball_positions is None:
                raise ValueError("No tracking data available.")

            # Create a new video clip with the draw_circles function applied to each frame
            marked_clip = VideoClip(
                lambda t: self.draw_circles(
                    clip.get_frame(t),
                    self.flyball_positions.loc[
                        round(t * clip.fps)  # Ensure correct frame is accessed
                    ],
                ),
                duration=clip.duration,
            )
        else:
            marked_clip = clip

        # Apply speed effect
        sped_up_clip = marked_clip.fx(vfx.speedx, speed)

        # If saving, write the new video clip to a file
        if save:
            print(f"Saving {self.video.name} at {speed}x speed in {output_path.parent}")
            sped_up_clip.write_videofile(
                str(output_path), fps=clip.fps
            )  # Save the sped-up clip

        # If not saving, preview the new video clip
        if preview:
            # Check if running over SSH
            if "SSH_CLIENT" in os.environ or "SSH_TTY" in os.environ:
                raise EnvironmentError(
                    "Preview mode shouldn't be run over SSH. Set preview argument to False or run locally."
                )

            # Initialize Pygame display
            pygame.display.init()

            # Set the title of the Pygame window
            pygame.display.set_caption(f"Preview (speed = x{speed})")

            print(f"Previewing {self.video.name} at {speed}x speed")

            sped_up_clip.preview(fps=self.experiment.fps * speed)

            # Manually close the preview window
            pygame.quit()

        # Close the video file to release resources
        clip.close()

        if not save and not preview:
            print("No action specified. Set save or preview argument to True.")


class Experiment:
    """
    A class for an experiment. This represents a folder containing multiple flies, each of which is represented by a Fly object.
    """

    def __init__(self, directory):
        """
        Parameters
        ----------
        directory : Path
            The path to the experiment directory.

        Attributes
        ----------
        directory : Path
            The path to the experiment directory.
            metadata : dict
            A dictionary containing the metadata for the experiment.
            fps : str
            The frame rate of the videos.
        """
        self.directory = directory
        self.metadata = self.load_metadata()
        self.fps = self.load_fps()
        self.flies = self.load_flies()

    def __str__(self):
        # Generate a list of unique genotypes from the flies in the experiment
        tested_genotypes = list(set([fly.Genotype for fly in self.flies]))

        return f"Experiment: {self.directory.name}\n  Genotypes: {', '.join(tested_genotypes)}\n  Flies: {len(self.flies)}\n  FPS: {self.fps}"

    def __repr__(self):
        return f"Experiment({self.directory})"

    def load_metadata(self):
        with open(self.directory / "Metadata.json", "r") as f:
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
            return metadata_dict

    def load_fps(self):
        # Load the fps value from the fps.npy file in the experiment directory
        fps_file = self.directory / "fps.npy"
        if fps_file.exists():
            fps = np.load(fps_file)

        else:
            fps = 30
            print(
                f"Warning: fps.npy file not found in {self.directory}; Defaulting to 30 fps."
            )

        return fps

    def load_flies(self):
        # Find all directories containing at least one .mp4 file
        mp4_directories = [
            dir for dir in self.directory.glob("**/*") if any(dir.glob("*.mp4"))
        ]

        # Find all .mp4 files that are named the same as their parent directory
        mp4_files = [
            mp4_file
            for dir in mp4_directories
            if (mp4_file := dir / f"{dir.name}.mp4").exists()
        ]

        # Create a Fly object for each .mp4 file
        flies = [Fly(mp4_file.parent, experiment=self) for mp4_file in mp4_files]

        return flies

    def generate_grid(self, preview=False, overwrite=False):
        # Check if the grid image already exists
        if (self.directory / "grid.png").exists() and not overwrite:
            print(f"Grid image already exists for {self.directory.name}")
            return
        else:
            print(f"Generating grid image for {self.directory.name}")

            frames = []
            min_rows = []
            paths = []

            for fly in self.flies:
                frame, min_row = fly.detect_boundaries()
                if frame is not None and min_row is not None:
                    frames.append(frame)
                    min_rows.append(min_row)
                    paths.append(fly.video)

            # Set the number of rows and columns for the grid

            nrows = 9
            ncols = 6

            # Create a figure with subplots
            fig, axs = plt.subplots(nrows, ncols, figsize=(20, 20))

            # Loop over the frames, minimum row indices, and video paths
            for i, (frame, min_row, flypath) in enumerate(zip(frames, min_rows, paths)):
                # Get the row and column index for this subplot
                row = i // ncols
                col = i % ncols

                # Plot the frame on this subplot
                try:
                    axs[row, col].imshow(frame, cmap="gray", vmin=0, vmax=255)
                except:
                    print(f"Error: Could not plot frame {i} for video {flypath}")
                    # go to the next folder
                    continue

                # Plot the horizontal lines on this subplot
                axs[row, col].axhline(min_row - 30, color="red")
                axs[row, col].axhline(min_row - 320, color="blue")

            # Remove the axis of each subplot and draw them closer together
            for ax in axs.flat:
                ax.axis("off")
            plt.subplots_adjust(wspace=0, hspace=0)

            # Save the grid image in the main folder
            plt.savefig(self.directory / "grid.png")

            if preview == True:
                plt.show()


class Dataset:
    def __init__(
        self,
        source,
        interactions=True,
        brain_regions_path="/mnt/labserver/DURRIEU_Matthias/Experimental_data/Region_map_240122.csv",
    ):
        """
        A class to generate a dataset Experiments and Fly objects.

        Parameters
        ----------
        source : can either be a list of Experiment objects, one Experiment object, a list of Fly objects or one Fly object.

        """
        self.source = source
        self.interactions = interactions
        # Define the experiments and flies attributes
        if isinstance(source, list):
            # If the source is a list, check if it contains Experiment or Fly objects, otherwise raise an error
            if isinstance(source[0], Experiment):
                # If the source contains Experiment objects, generate a dataset from the experiments
                self.experiments = source

                self.flies = [
                    fly for experiment in self.experiments for fly in experiment.flies
                ]

            elif isinstance(source[0], Fly):
                # make a list of distinct experiments associated with the flies
                self.experiments = list(set([fly.experiment for fly in source]))

                self.flies = source

            else:
                raise TypeError(
                    "Invalid source format: source must be a (list of) Experiment objects or a list of Fly objects"
                )

        elif isinstance(source, Experiment):
            # If the source is an Experiment object, generate a dataset from the experiment
            self.experiments = [source]

            self.flies = source.flies

        elif isinstance(source, Fly):
            # If the source is a Fly object, generate a dataset from the fly
            self.experiments = [source.experiment]

            self.flies = [source]
        else:
            raise TypeError(
                "Invalid source format: source must be a (list of) Experiment objects or a list of Fly objects"
            )

        self.brain_regions_path = brain_regions_path
        self.regions_map = pd.read_csv(self.brain_regions_path)
        
        self.metadata = []

        self.data = self.generate_dataset()

    def __str__(self):
        # Look for recurring words in the experiment names
        experiment_names = [
            experiment.directory.name for experiment in self.experiments
        ]
        experiment_names = "_".join(experiment_names)
        experiment_names = experiment_names.split("_")  # Split by "_"

        # Ignore certain labels
        labels_to_ignore = {"Tracked", "Videos"}
        experiment_names = [
            name for name in experiment_names if name not in labels_to_ignore
        ]

        # Ignore words that are found only once
        experiment_names = [
            name for name in experiment_names if experiment_names.count(name) > 1
        ]

        experiment_names = Counter(experiment_names)
        experiment_names = experiment_names.most_common(3)
        experiment_names = [name for name, _ in experiment_names]
        experiment_names = ", ".join(experiment_names)

        return f"Dataset with {len(self.flies)} flies and {len(self.experiments)} experiments\nkeyword: {experiment_names}"

    def __repr__(self):
        # Adapt the repr function to the source attribute
        # If the source is a list, check if it is Fly or Experiment objects
        if isinstance(self.source, list):
            if isinstance(self.source[0], Experiment):
                return f"Dataset({[experiment.directory for experiment in self.experiments]})"
            elif isinstance(self.source[0], Fly):
                return f"Dataset({[fly.directory for fly in self.flies]})"
        elif isinstance(self.source, Experiment):
            return f"Dataset({self.experiments[0].directory})"
        elif isinstance(self.source, Fly):
            return f"Dataset({self.flies[0].directory})"

    def generate_dataset(self):
        """Generates a pandas DataFrame from a list of Experiment objects. The dataframe contains the smoothed fly and ball positions for each experiment.

        Args:
            experiments (list): A list of Experiment objects.
        """

        try:
            dataset_list = [self._prepare_dataset(fly) for fly in self.flies]

            self.data = pd.concat(dataset_list, ignore_index=True).reset_index(
                drop=True
            )

        except Exception as e:
            print(f"An error occurred: {e}")
            self.data = pd.DataFrame()

        return self.data

    def _prepare_dataset(self, fly):
        """Helper function to prepare individual fly dataset."""

        dataset = fly.flyball_positions

        if self.interactions:
            fly.annotate_events()

        # Add a column with the fly name as categorical data
        dataset["fly"] = fly.name
        # dataset["fly"] = dataset["fly"].astype("str")
        dataset["fly"] = dataset["fly"].astype("category")

        # Add a column with the experiment name as categorical data
        dataset["experiment"] = fly.experiment.directory.name
        dataset["experiment"] = dataset["experiment"].astype("category")

        # Add the metadata for the fly's arena as columns
        for var, data in fly.arena_metadata.items():
            dataset[var] = data
            dataset[var] = dataset[var].astype("category")
            # If the variable name is not in the metadata list, add it
            if var not in self.metadata:
                self.metadata.append(var)

        # If one of the columns is 'Genotype', use the brain regions csv file to add the associated brain region
        if "Genotype" in dataset.columns:
            dataset = dataset.merge(self.regions_map, on="Genotype", how="left")
            dataset["Simplified region"] = dataset["Simplified region"].astype(
                "category"
            )

        return dataset

    def get_event_numbers(self):
        # Check that events have been annotated
        if "event" not in self.data.columns:
            raise ValueError(
                "No events have been annotated. Run the annotate_events method first."
            )

        # Group the data by Fly and Event
        GroupedData = (
            self.data.groupby(["fly", "Nickname", "Simplified region"])
            .nunique(["event"])
            .reset_index()
        )

        # Calculate sample size
        SampleSize = (
            self.data.groupby(["Nickname", "Simplified region"])
            .nunique()["fly"]
            .reset_index()
            .rename(columns={"fly": "SampleSize"})
        )

        # Merge GroupedData and SampleSize
        GroupedData = pd.merge(
            GroupedData, SampleSize, on=["Nickname", "Simplified region"]
        )

        # Modify Nickname column
        GroupedData["Nickname"] = (
            GroupedData["Nickname"]
            + " (n = "
            + GroupedData["SampleSize"].astype(str)
            + ")"
        )
        
        # Add the metadata to the GroupedData
        dropdata = self.data.drop_duplicates(subset="fly")
        GroupedData.set_index("fly", inplace=True)
        dropdata.set_index("fly", inplace=True)
        
        GroupedData.update(dropdata[self.metadata])
        GroupedData.reset_index(inplace=True)
        
        # TODO: Add handling when there's no simplified region, to get the simpler version of the data
        return GroupedData

    def jitter_boxplot(
        self, data, vdim, plot_options=hv_main, show=True, save=False, outpath=None
    ):
        # Get the metadata for the tooltips
        tooltips = [
            ("Fly", "@fly"),
            (vdim.capitalize(), f"@{vdim}"),
        ]

        # Add the metadata to the tooltips
        for var in self.metadata:
            tooltips.append((var.capitalize(), f"@{var}"))

        hover = HoverTool(tooltips=tooltips)

        h_NumbEvents_bp = (
            hv.BoxWhisker(
                data=data,
                vdims=vdim,
                kdims=["Nickname", "Simplified region"],
                color="Nickname",
            )
            .groupby("Simplified region")
            .opts(**plot_options["boxwhisker"])
        )

        h_NumbEvents_sc = (
            hv.Scatter(
                data=data,
                vdims=[vdim] + self.metadata + ["fly"],
                kdims=["Nickname", "Simplified region"],
                color="Nickname",
            )
            .groupby("Simplified region")
            .opts(**plot_options["scatter"], tools=[hover])
        )

        hvplot_NumbEvents = (h_NumbEvents_bp * h_NumbEvents_sc).opts(
            **plot_options["plot"]
        )

        if show:
            hv.render(hvplot_NumbEvents)
        if save:
            if outpath is None:
                now = datetime.datetime.now()  # get current date and time
                date_time = now.strftime("%Y%m%d_%H%M")  # format as a string

                output_path = (
                    get_labserver()
                    / "Experimental_data"
                    / "MultiMazeRecorder"
                    / f"{vdim}Number_{date_time}.html"
                )

            hv.save(hvplot_NumbEvents, output_path)

        return hvplot_NumbEvents
