import datetime
import numpy as np
import pandas as pd

def checksave(path, object, file):
    """Checks if a file exists and asks the user if they want to overwrite it.
    Arguments:
        file: string. The name of the file to be saved.
        object: string. The kind of the file to be saved. will adapt the format to be used
            Currently, supported objects are: 
            > "parameter" : numpy array or list to be saved as .npy 
            > "dataframe" : pandas dataframe to be saved as .feather
    """
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
    
    else: print("Invalid object type")

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
                print (timestamp)
            except TypeError:
                print('Wrong variable type entered. Provide integers values.')


        else:
            try:
                s = time/fps
                hours, remainder = divmod(s, 3600)
                minutes, seconds = divmod(remainder, 60)

                timestamp = (int(hours), int(minutes), round(seconds))
                print('%s:%s:%s' %(timestamp[0], timestamp[1], timestamp[2]))
            except TypeError:
                print('Wrong variable type entered. Provide integers values.')

    else:
        if clockformat is False:

            try:
                timestamp = round(time * fps)
                print (timestamp)
            except TypeError:
                print('Wrong variable type entered. Provide integers values.')

        else:

            try:
                transtime = datetime.datetime.strptime(time, '%H:%M:%S')
                timestamp = ((transtime.hour * 3600) + (transtime.minute * 60) + transtime.second) * fps
                print (timestamp)
            except TypeError:
                print('Wrong variable type entered. Provide a string with "%Hours:%Minutes:%Seconds" format.')

    return timestamp