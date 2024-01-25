import h5py
from pathlib import Path
import cv2
import numpy as np
import random


def load_h5_data(filepath):
    with h5py.File(filepath, "r") as f:
        dset_names = list(f.keys())
        locs = f["tracks"][:].T
        node_names = [n.decode() for n in f["node_names"][:]]
    return locs


def create_marked_video(directory, start_frame, end_frame, sample=False):
    # Define the paths
    dirpath = Path(directory)
    ballpath = list(dirpath.glob("*tracked_ball*.analysis.h5"))[0]
    flypath = list(dirpath.glob("*tracked_fly*.analysis.h5"))[0]
    vidpath = list(dirpath.glob("corridor*.mp4"))[0]
    outpath = dirpath / "ballflytrack_sample.mp4"

    # Load the h5 files
    ball_locs = load_h5_data(ballpath.as_posix())
    fly_locs = load_h5_data(flypath.as_posix())

    yball = ball_locs[:, :, 1, :]
    xball = ball_locs[:, :, 0, :]

    yfly = fly_locs[:, :, 1, :]
    xfly = fly_locs[:, :, 0, :]

    # Open the input video file
    cap = cv2.VideoCapture(vidpath.as_posix())

    # Get the video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create the output video file
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(outpath.as_posix(), fourcc, fps, (width, height))

    # Set the position of the video to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Process each frame of the video
    for i in range(start_frame, end_frame + 1):
        # If sample is True, randomly skip frames
        if sample and random.random() < 0.95:
            continue

        # Read the next frame from the input video
        ret, frame = cap.read()

        # Check if the frame was successfully read
        if not ret:
            break

        # Get the positions of the fly and ball in this frame
        fly_pos = (int(xfly[i]), int(yfly[i]))
        ball_pos = (int(xball[i]), int(yball[i]))

        # Draw a circle at each position
        cv2.circle(frame, fly_pos, 10, (0, 0, 255), -1)
        cv2.circle(frame, ball_pos, 10, (255, 0, 0), -1)

        # Write the frame to the output video
        out.write(frame)

        # Wait for 1 millisecond to update the window
        cv2.waitKey(1)

    # Release the video capture and writer
    cap.release()
    out.release()


# Call the function
create_marked_video(
    "/Volumes/Ramdya-Lab/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Videos/231121_TNT_Fine_3_Videos_Tracked/arena5/corridor6",
    2000,
    3200,
    sample=True,
)
