# This script is used to detect potential dead flies in experiments. First it loads an experiments and look at the fly tracking data. If the fly doesn't move more than 10 px for the whole video, it will be considered as potentially dead and prompt the user to check the video. If the user confirms death, a "dead" tag will be added to the experiment.

from Ballpushing_utils import *
import platform

# Load the experiment

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

print(f"Loading {DataPath}")

# For each folder in the DataPath containing a .mp4 file, load the corresponding fly tracking data

Folder = DataPath / "230719_TNTscreen_Broad_1_Videos_Tracked"

print(f"Checking for dead flies in {Folder}")

files = list(Folder.glob("**/*.mp4"))

for file in files:
    dir = file.parent

    arena = file.parent.parent.name

    # Define flypath as the *tracked_fly*.analysis.h5 file in the same folder as the video
    try:
        flypath = list(dir.glob("*tracked_fly*.analysis.h5"))[0]
        # print(flypath.name)
    except IndexError:
        print(f"No fly tracking file found for {file.name}, skipping...")

        continue

    # Load the fly tracking data
    fly_coordinates = get_coordinates(flypath=flypath, ball=False, xvals=True)
    
    # Get fps from the Folder
    # For this test I use just 30
    fps = 30
    
    # Crop the fly coordinates to remove the first minute of the video

    fly_coordinates = fly_coordinates.iloc[fps * 60 :]
    
    # Check if at any time the fly position was atleast 10 px away from the first position
    if np.any(
        np.abs(fly_coordinates["yfly_smooth"][0] - fly_coordinates["yfly_smooth"]) > 20
    ):
        print(f"Fly moved more than 20 px in {arena}/{file.name}, skipping...")
        continue

    else:
        print(f"Potential dead fly in {arena}/{file.name}")
        # Ask the user if the fly is dead
        dead = input("Is the fly dead? (y/n)")
        if dead == "y":
            # Add the dead tag to the video name
            new_name = file.name + "_dead.mp4"
            file.rename(dir / new_name)
            print(f"Renamed {file.name} to {new_name}")
        else:
            print(f"Skipped {file.name}")
