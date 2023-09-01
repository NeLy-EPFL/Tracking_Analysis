from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter
from tqdm import tqdm

# Set the path to the data folder
data_folder = Path(
    "/mnt/labserver/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Videos"
)

# Set the threshold value
threshold = 100

# Loop over all main folders in the data folder that end with _Tracked
for main_folder in tqdm(
    list(data_folder.glob("*_Tracked")), desc="Processing main folders"
):
    # Check if the grid.png file already exists in the main folder
    if (main_folder / "grid.png").exists():
        # Check if all coordinates.npy files already exist in the video folders
        all_coordinates_exist = all(
            (file.parent / "coordinates.npy").exists()
            for file in main_folder.rglob("*.mp4")
        )

        # If all coordinates.npy files exist, skip this main folder
        if all_coordinates_exist:
            print(
                f"Skipping main folder {main_folder} because grid.png and coordinates.npy already exist"
            )
            continue

    # Print the current main folder being processed
    print(f"Processing main folder: {main_folder}")

    # Create a list to store the frames, minimum row indices, and video paths
    frames = []
    min_rows = []
    video_paths = []

    # Recursively traverse the directory tree
    for file in tqdm(list(main_folder.rglob("*.mp4")), desc="Processing videos"):
        # Set the path to the video file
        Videopath = file

        # open the first frame of the video
        cap = cv2.VideoCapture(Videopath.as_posix())
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"Error: Could not read frame from video {Videopath}")
        elif frame is None:
            print(f"Error: Frame is None for video {Videopath}")
        else:
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

            # Store the frame, minimum row index, and video path
            frames.append(frame)
            min_rows.append(min_row)
            video_paths.append(Videopath)

    # Set the number of rows and columns for the grid
    nrows = 9
    ncols = 6

    # Create a figure with subplots
    fig, axs = plt.subplots(nrows, ncols, figsize=(20, 20))

    # Loop over the frames, minimum row indices, and video paths
    for i, (frame, min_row, Videopath) in enumerate(zip(frames, min_rows, video_paths)):
        # Get the row and column index for this subplot
        row = i // ncols
        col = i % ncols

        # Plot the frame on this subplot
        axs[row, col].imshow(frame, cmap="gray", vmin=0, vmax=255)

        # Plot the horizontal lines on this subplot
        axs[row, col].axhline(min_row - 30, color="red")
        axs[row, col].axhline(min_row - 320, color="blue")

        # Save a .npy file with the start and end coordinates in the video folder
        np.save(Videopath.parent / "coordinates.npy", [min_row - 30, min_row - 320])

    # Remove the axis of each subplot and draw them closer together
    for ax in axs.flat:
        ax.axis("off")
    plt.subplots_adjust(wspace=0, hspace=0)

    # Save the grid image in the main folder
    plt.savefig(main_folder / "grid.png")
