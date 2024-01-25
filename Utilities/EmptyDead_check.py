import cv2
import platform
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from Ballpushing_utils import *


def check_arena_empty(video_path):
    # Load the first frame of video
    Vid = cv2.VideoCapture(str(video_path))
    Vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = Vid.read()
    Vid.release()

    # Convert to grayscale
    Vid = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load the coordinates.npy file
    start, end = np.load(video_path.parent / "coordinates.npy")

    arena = video_path.parent.parent.name

    # Crop the frames to the chamber location, which is any y value above the start position
    crop = Vid[start + 40 :, :]

    # Detect the edges of the arena and crop the image to the edges
    edges = cv2.Canny(crop, 100, 200)
    # Find the non zero pixels
    nz = np.nonzero(edges)
    # Crop the image to the edges
    crop = crop[np.min(nz[0]) : np.max(nz[0]), np.min(nz[1]) : np.max(nz[1])]

    # Binarise the images with a threshold of 50
    crop_bin = crop < 60

    # Create a kernel
    kernel = np.ones((3, 3), np.uint8)

    # Apply an opening operation
    crop_bin = cv2.morphologyEx(crop_bin.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    # If there's a peak, the arena is not empty
    if np.any(crop_bin > 0):
        print(f"{arena}/{video_path.name} is not empty")
        return False
    else:
        print(f"{arena}/{video_path.name} is empty")
        return True


def check_arena_and_fly(video_path):
    # First, check if the arena is empty
    dir = video_path.parent
    arena = video_path.parent.parent.name

    if check_arena_empty(video_path):
        print(f"{arena}/{video_path.name} is empty, skipping...")
        return
    # If the arena is not empty, check if the fly is dead

    # Define flypath as the *tracked_fly*.analysis.h5 file in the same folder as the video
    try:
        flypath = list(dir.glob("*tracked_fly*.analysis.h5"))[0]
    except IndexError:
        print(f"No fly tracking file found for {video_path.name}, skipping...")
        return

    # Load the fly tracking data
    fly_coordinates = get_coordinates(flypath=flypath, ball=False, xvals=True)

    # Get fps from the Folder
    # For this test I use just 30
    fps = 30

    # Crop the fly coordinates to remove the first minute of the video
    fly_coordinates = fly_coordinates.iloc[fps * 60 :]

    # Check if at any time the fly position was at least 10 px away from the first position
    if np.any(
        np.abs(fly_coordinates["yfly_smooth"][61] - fly_coordinates["yfly_smooth"]) > 20
    ):
        print(f"Fly moved more than 20 px in {arena}/{video_path.name}, skipping...")
        return

    print(f"Potential dead fly in {arena}/{video_path.name}")
    # Ask the user if the fly is dead
    dead = input("Is the fly dead? (y/n)")
    if dead == "y":
        # Add the dead tag to the video name
        new_name = video_path.name + "_dead.mp4"
        video_path.rename(dir / new_name)
        print(f"Renamed {video_path.name} to {new_name}")
    else:
        print(f"Skipped {video_path.name}")


# Get the DataFolder
if platform.system() == "Darwin":
    DataPath = Path(
        "/Volumes/Ramdya-Lab/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Videos"
    )
elif platform.system() == "Linux":
    DataPath = Path(
        "/mnt/labserver/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Videos"
    )

print(f"Loading {DataPath}")

# For each folder in the DataPath containing a .mp4 file, load the corresponding fly tracking data
Folder = DataPath / "230719_TNTscreen_Broad_1_Videos_Tracked"
print(f"Checking for empty arenas and dead flies in {Folder}")

files = list(Folder.glob("**/*.mp4"))

for file in files:
    check_arena_and_fly(file)
