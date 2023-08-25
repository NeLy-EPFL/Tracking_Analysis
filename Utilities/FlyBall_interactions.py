from pathlib import Path
import cv2
import h5py
import numpy as np
from imageio import mimwrite


def process_folder(folder_path, threshold):
    folder_path = Path(folder_path)

    # Get all video files in the folder
    video_files = [f for f in folder_path.glob("*.mp4")]

    # Create a list to store the extracted frames
    extracted_frames = []

    for video_file in video_files:
        # Construct the file paths for the position data files
        pos_file_1 = video_file.with_name(video_file.stem + "_pos1.h5")
        pos_file_2 = video_file.with_name(video_file.stem + "_pos2.h5")

        # Load the position data
        with h5py.File(pos_file_1, "r") as f:
            pos_data_1 = f["tracks"][:, :, 1, :].T
        with h5py.File(pos_file_2, "r") as f:
            pos_data_2 = f["tracks"][:, :, 1, :].T

        # Compute the distances between the two objects along the y-axis
        distances = np.abs(pos_data_1 - pos_data_2)

        # Find the frames where the distance is below the threshold
        close_frames = np.where(distances < threshold)[0]

        # Load the video
        cap = cv2.VideoCapture(str(video_file))

        # Extract the frames where the objects are close
        for frame_idx in close_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                extracted_frames.append(frame)

    # Create a grid of frames
    n_rows = int(np.ceil(np.sqrt(len(extracted_frames))))
    n_cols = int(np.ceil(len(extracted_frames) / n_rows))

    frame_height, frame_width, _ = extracted_frames[0].shape

    grid = np.zeros((frame_height * n_rows, frame_width * n_cols, 3), dtype=np.uint8)

    for i, frame in enumerate(extracted_frames):
        row = i // n_cols
        col = i % n_cols

        grid[
            row * frame_height : (row + 1) * frame_height,
            col * frame_width : (col + 1) * frame_width,
        ] = frame

    # Save the grid as a video
    mimwrite(str(folder_path / "grid.mp4"), [grid], fps=30)


process_folder("/path/to/my/folder", 50)
