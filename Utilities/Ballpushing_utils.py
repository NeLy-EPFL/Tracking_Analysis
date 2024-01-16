import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
import sys
import traceback
import json
import datetime
import subprocess

import cv2
from moviepy.editor import VideoFileClip
from moviepy.video.fx import all as vfx
import pygame


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
    # Check if the first value is NaN
    if np.isnan(arr[0]):
        # Find the next non-NaN value
        next_val = arr[next((i for i, x in enumerate(arr) if not np.isnan(x)), None)]
        arr[0] = next_val

    # Find the indices of the NaN values
    nan_indices = np.where(np.isnan(arr))

    # Replace the NaN values with the previous value
    for i in nan_indices[0]:
        arr[i] = arr[i - 1]


def generate_dataset(Folders, fly=True, ball=True, xvals=False, fps=30, Events=None):
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

            # Define flypath as the *tracked_fly*.analysis.h5 file in the same folder as the video
            try:
                flypath = list(dir.glob("*tracked_fly*.analysis.h5"))[0]
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

                # print(data.head())
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
                    # print(
                    #     f"Flipped video, flipping ball and fly y coordinates, flipping start and end."
                    # )
                    data["yball_smooth"] = -data["yball_smooth"]
                    data["yfly_smooth"] = -data["yfly_smooth"]
                    # start = -start

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
                if Events == "interactions":
                    # Compute interaction events for all data
                    interaction_events = find_interaction_events(data)

                    # Assign an event number to each event
                    for i, (start_time, end_time) in enumerate(
                        interaction_events, start=1
                    ):
                        data.loc[
                            (data.Frame >= start_time) & (data.Frame <= end_time),
                            "Event",
                        ] = i

                Dataset_list.append(data)

            except Exception as e:
                error_message = str(e)
                traceback_message = traceback.format_exc()
                # print(f"Error processing video {vidname}: {error_message}")
                print(traceback_message)

    # Concatenate all dataframes in the list into a single dataframe
    Dataset = pd.concat(Dataset_list, ignore_index=True).reset_index()

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


def extract_interaction_events(source, Thresh=80, min_time=60, as_df=False):
    if isinstance(source, Path):
        print(f"Path: {source}")
        flypath = next(source.glob("*tracked_fly*.analysis.h5"))
        ballpath = next(source.glob("*tracked*.analysis.h5"))
        df = get_coordinates(flypath=flypath, ballpath=ballpath)

    elif isinstance(source, pd.DataFrame):
        print(f"DataFrame: {source.shape}")
        df = source

    else:
        raise TypeError(
            "Invalid source format: source must be a pathlib Path, string or a pandas DataFrame"
        )

    # Create a new column 'Event' and initialize it with None
    df.loc[:, "Event"] = None

    # Compute interaction events for all data
    interaction_events = find_interaction_events(df, Thresh, min_time)

    # Assign an event number to each event
    for i, (start_time, end_time) in enumerate(interaction_events, start=1):
        df.loc[(df.index >= start_time) & (df.index <= end_time), "Event"] = i

    if "Fly" in df.columns:
        # Compute the maximum event number for each fly
        max_event_per_fly = (
            df.groupby("Fly")["Event"].max().shift(fill_value=0).cumsum()
        )

        # Adjust event numbers for each fly
        df["Event"] -= df["Fly"].map(max_event_per_fly)

    else:
        # Compute interaction events for all data
        interaction_events = find_interaction_events(df, Thresh, min_time)

        # Assign an event number to each event
        for i, (start_time, end_time) in enumerate(interaction_events, start=1):
            df.loc[(df.index >= start_time) & (df.index <= end_time), "Event"] = i

    if as_df:
        return df
    else:
        return interaction_events


def find_interaction_events(df, Thresh=80, min_time=60):
    df.loc[:, "dist"] = df.loc[:, "yfly_smooth"] - df.loc[:, "yball_smooth"]
    df.loc[:, "close"] = df.loc[:, "dist"] < Thresh
    df.loc[:, "block"] = (df.loc[:, "close"].shift(1) != df.loc[:, "close"]).cumsum()
    events = (
        df[df["close"]]
        .groupby("block")
        .agg(start=("Frame", "min"), end=("Frame", "max"))
    )
    interaction_events = [
        (start, end) for start, end in events[["start", "end"]].itertuples(index=False)
    ]
    interaction_events = [
        event for event in interaction_events if event[1] - event[0] >= min_time
    ]
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


class Fly:
    """
    A class for a single fly.

    Parameters
    ----------
    directory : Path
        The path to the fly directory.

    Attributes
    ----------
    directory : Path
        The path to the fly directory.
        experiment : Experiment
        The experiment that the fly belongs to.
    """

    def __init__(self, directory, experiment=None):
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

        self.video = list(self.directory.glob(f"{self.corridor}.mp4"))[0]

        try:
            self.flytrack = list(directory.glob("*tracked_fly*.analysis.h5"))[0]
            # print(flypath.name)
        except IndexError:
            print(f"No fly tracking file found for {self.name}, skipping...")

        try:
            self.balltrack = list(directory.glob("*tracked_ball*.analysis.h5"))[0]
            # print(ballpath.name)
        except IndexError:
            print(f"No ball tracking file found for {self.name}, skipping...")

        # Compute distance between fly and ball
        self.flyball_positions = get_coordinates(self.balltrack, self.flytrack)

        self.interaction_events = find_interaction_events(self.flyball_positions)

    def get_arena_metadata(self):
        # Get the metadata for this fly's arena
        arena_key = self.arena.lower()
        return {
            var: data[arena_key]
            for var, data in self.experiment.metadata.items()
            if arena_key in data
        }

    def display_metadata(self):
        # Print the metadata for this fly's arena
        for var, data in self.arena_metadata.items():
            print(f"{var}: {data}")

    def find_interaction_events(
        self,
        gap_between_events=1,
        event_min_length=60,
        thresh=[0, 80],
        omit_events=None,
        plot_signals=False,
        signal_name="",
    ):
        """
        This function finds events in a given signal based on certain criteria.

        Parameters:
        signal (list): The signal in which to find events.
        thresh (list): The lower and upper limit values for the signal.
        gap_between_events (int): The minimum gap required between two events.
        event_min_length (int): The minimum length of an event.
        omit_events (list, optional): A range of events to omit. Defaults to None.
        plot_signals (bool, optional): Whether to plot the signals or not. Defaults to False.
        signal_name (str, optional): The name of the signal. Defaults to "".

        Returns:
        list: A list of events found in the signal. Each event is a list containing the start frame, end frame and duration of the event.
        """

        distance = (
            self.flyball_positions.loc[:, "yfly_smooth"]
            - self.flyball_positions.loc[:, "yball_smooth"]
        )

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

    def check_yball_variation(self, event, threshold=10):
        # Get the yball_smooth segment corresponding to an event
        yball_event = self.flyball_positions.loc[event[0] : event[1], "yball_smooth"]

        variation = yball_event.max() - yball_event.min()

        return variation > threshold

    def generate_clip(self, event, outpath, fps, width, height):
        start_frame, end_frame = event[0], event[1]
        cap = cv2.VideoCapture(str(self.video))
        try:
            start_time = start_frame / fps
            start_time_str = str(datetime.timedelta(seconds=int(start_time)))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            # Get the index of the event in the list to apply it to the output file name
            event_index = self.interaction_events.index(event)

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
        Use detected events to generate clips of the fly's interactions with the ball, then concatenate the clips to generate a video.
        """

        if outpath is None:
            outpath = self.directory
        events = self.interaction_events
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

    def generate_preview(self, speed=60, save=False, output_path=None):
        """
        Generate an accelerated version of the video using moviepy.

        Parameters:
        speed (float): The speedup factor. For example, 2.0 will double the speed of the video.
        save (bool, optional): Whether to save the sped up video. Defaults to False.
        output_path (str, optional): The path to save the sped up video. If not provided and save is True, a default path will be used. Defaults to None.
        """

        if save and output_path is None:
            # Use the default output path
            output_path = (
                get_labserver()
                / "Videos"
                / "Previews"
                / f"{self.name}_{self.Genotype if self.Genotype else 'undefined'}_x{speed}.mp4"
            )

        # Construct the ffmpeg command
        cmd = f"ffmpeg -i {self.video} -vf 'setpts={1/speed}*PTS' -loglevel panic {output_path}"

        # Execute the command
        if save:
            print(f"Saving {self.video.name} at {speed}x speed in {output_path.parent}")

            subprocess.call(cmd, shell=True)
        else:
            # Load the video file
            clip = VideoFileClip(self.video.as_posix())

            # Apply speed effect
            sped_up_clip = clip.fx(vfx.speedx, speed)
            # Preview the sped up video (perhaps with show)
            # Initialize Pygame display
            pygame.display.init()

            # Set the title of the Pygame window
            pygame.display.set_caption(f"Preview (speed = x{speed})")

            print(f"Previewing {self.video.name} at {speed}x speed")

            sped_up_clip.preview(
                fps=self.experiment.fps * speed,
            )

            # Close the video file to release resources
            clip.close()

            # Manually close the preview window
            pygame.quit()


class Experiment:
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
                f"Error: fps.npy file not found in {self.directory}; Defaulting to 30 fps."
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


class Dataset:
    def __init__(self, source):
        """
        A class to generate a dataset from mazerecorder videos.

        Parameters
        ----------
        source : can either be a list of Experiment objects, one Experiment object, a list of Fly objects or one Fly object.

        """

        if isinstance(source, list):
            # If the source is a list, check if it contains Experiment objects or Fly objects
            if isinstance(source[0], Experiment):
                # If the source contains Experiment objects, generate a dataset from the experiments
                self.dataset = self.generate_dataset_from_experiments(source)
            elif isinstance(source[0], Fly):
                # If the source contains Fly objects, generate a dataset from the flies
                self.dataset = self.generate_dataset_from_flies(source)
            else:
                raise TypeError(
                    "Invalid source format: source must be a list of Experiment objects or a list of Fly objects"
                )
        elif isinstance(source, Experiment):
            # If the source is an Experiment object, generate a dataset from the experiment
            self.dataset = self.generate_dataset_from_experiments([source])
        elif isinstance(source, Fly):
            # If the source is a Fly object, generate a dataset from the fly
            self.dataset = self.generate_dataset_from_flies([source])
        else:
            raise TypeError(
                "Invalid source format: source must be a list of Experiment objects or a list of Fly objects"
            )

    # def generate_dataset_from_experiments(self, experiments):
    # TODO: implement this function
