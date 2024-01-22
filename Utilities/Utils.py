import datetime
import numpy as np
import pandas as pd
from traceback import format_tb
import requests
import os
import platform
from pathlib import Path


def get_labserver():
    """
    Returns the appropriate data path based on the platform.

    Returns:
    Path: The data path.
    """

    if platform.system() == "Darwin":
        return Path("/Volumes/Ramdya-Lab/DURRIEU_Matthias/")
    elif platform.system() == "Linux":
        return Path("/mnt/labserver/DURRIEU_Matthias/")
    else:
        raise ValueError("Unsupported platform")


def get_data_path(setup="mazerecorder"):
    """
    Returns the data path for the specified setup.

    """

    labserver = get_labserver()

    if setup == "mazerecorder":
        datapath = labserver.joinpath("Experimental_data/MultiMazeRecorder/Videos")

    return datapath


def get_folders(path, keywords):
    """Generates a list of Experiment objects based on keywords.

    Args:
        path (str): The path where to look for experiments.
        keywords (list): A list of keywords to filter the experiments.

    Returns:
        list: A list of folders.
    """

    # Get all folders that have all keywords in their name
    Folders = [
        f for f in path.iterdir() if all(keyword.lower() in f.name.lower() for keyword in keywords)
    ]

    return Folders


def checksave(path, object, file):
    """Checks if a file exists and asks the user if they want to overwrite it.
    Arguments:
        file: string. The name of the file to be saved.
        object: string. The kind of the file to be saved. will adapt the format to be used
            Currently, supported objects are:
            > "parameter" : numpy array or list to be saved as .npy
            > "dataframe" : pandas dataframe to be saved as .feather
    """
    # check if provided path is a pathlib Path object

    try:
        if object == "parameter":
            if path.exists() is True:
                choice = input("File already exists! Overwrite? [y/n]")

                if choice == "n":
                    print("File unchanged.")

                elif choice == "y":
                    np.save(path, file)
                    print("File updated.")

                else:
                    print("invalid input")

            else:
                np.save(path, file)

        elif object == "dataframe":
            if path.exists() is True:
                choice = input("File already exists! Overwrite? [y/n]")

                if choice == "n":
                    print("File unchanged.")

                elif choice == "y":
                    file.to_feather(path)
                    print("File updated.")

                else:
                    print("invalid input")

            else:
                file.to_feather(path)

        else:
            print("Invalid object type")

    except AttributeError as err:
        raise AttributeError("path argument must be a pathlib object")


def frame2time(time, fps, reverse=False, clockformat=False):
    """Converts a framecount to time and vice-versa
    Arguments:
        time: Either an integer (framecount or seconds) or a string of format '%Hours:%Minutes:%Seconds'.
        fps: integer. Frames per second of the video
        reverse: Boolean. determine if you are trying to convert frames to time or time to frame.
        clockformat: Boolean. If True, will deal with time as hours, minutes, seconds. else, will deal with time as seconds.

    Returns:
        A timestamp, which can be in the form of an integer (seconds or frames) or a tuple of hours, minutes, seconds.

    """
    if reverse is False:
        if clockformat is False:
            try:
                timestamp = round(time / fps)
                print(timestamp)
            except TypeError:
                print("Wrong variable type entered. Provide integers values.")

        else:
            try:
                s = time / fps
                hours, remainder = divmod(s, 3600)
                minutes, seconds = divmod(remainder, 60)

                timestamp = (int(hours), int(minutes), round(seconds))
                print("%s:%s:%s" % (timestamp[0], timestamp[1], timestamp[2]))
            except TypeError:
                print("Wrong variable type entered. Provide integers values.")

    else:
        if clockformat is False:
            try:
                timestamp = round(time * fps)
                print(timestamp)
            except TypeError:
                print("Wrong variable type entered. Provide integers values.")

        else:
            try:
                transtime = datetime.datetime.strptime(time, "%H:%M:%S")
                timestamp = (
                    (transtime.hour * 3600) + (transtime.minute * 60) + transtime.second
                ) * fps
                print(timestamp)
            except TypeError:
                print(
                    'Wrong variable type entered. Provide a string with "%Hours:%Minutes:%Seconds" format.'
                )

    return timestamp


def add_note(path, note):
    """Adds a note to a file or update an existing note.
    Arguments:
        path: pathlib object. Path to the file to be modified
        note: string. The note to be added to the file
    """
    if path.parent.joinpath("notes.txt").exists():
        with open(path.parent.joinpath("notes.txt").as_posix(), "a") as f:
            f.write(note + "\n")
        print("note file updated")
    else:
        with open(path.parent.joinpath("notes.txt").as_posix(), "w") as f:
            f.write(note + "\n")
        print("note file created")


def notify_me():
    """Sends a notification to your phone when a script is done running.
    it works with IFTTT android app and a webhook trigger. More information on how to set this up is available here:
    https://github.com/DrGFreeman/IFTTT-Webhook

    """
    ifttt_url = os.getenv("IFTTT_URL")
    if ifttt_url is not None:
        requests.post(ifttt_url)
    else:
        print(
            'IFTTT_URL environment variable not set. You can set it up using the following command: \n export IFTTT_URL="https://maker.ifttt.com/trigger/{event}/with/key/{your-IFTTT-key}" Note that you need to replace {event} and {your-IFTTT-key} with your own values. If you want to permanently set this variable, you can add the line above to your .bashrc or .zshrc file.'
        )
