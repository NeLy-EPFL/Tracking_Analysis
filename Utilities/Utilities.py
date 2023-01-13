import datetime

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