import numpy as np
import pandas as pd
import sys

sys.path.insert(0, "../Tracktor")

import tracktor as tr
import cv2
import sys
import os
import random
from pathlib import Path


def track_objects_in_folder(input_folder, trim=True, filter_keywords=[]):
    """Track objects in all video files in the given folder."""
    input_folder = Path(input_folder)
    files = list(input_folder.glob("*.mp4"))  # Assuming the videos are in mp4 format

    for file in files:
        # Check if file name contains any keyword in filter_keywords
        if any(keyword in file.name for keyword in filter_keywords):
            continue

        print(f"Processing file: {file.as_posix()}")

        if trim:
            # Trim the video
            trimmed_path = trim_video(file)
            # Track objects in the trimmed video
            track_objects_in_video(trimmed_path)
        else:
            # Track objects in the original video
            track_objects_in_video(file)


def trim_video(file):
    """Trim the video to remove the part where the arena is opened initially."""
    startpoint = "00:00:00"
    finishpoint = "00:10:00"
    trimmed_path = file.with_stem(file.stem + "_Trimmed")

    os.system(
        f"ffmpeg -hide_banner -loglevel error -i {file.as_posix()} -ss {startpoint} -to {finishpoint} -c copy {trimmed_path.as_posix()}"
    )

    return trimmed_path


def track_objects_in_video(input_vidpath):
    """Track objects in the given video file."""
    # Set up parameters for tracking
    output_vidpath = input_vidpath.with_stem(input_vidpath.stem + "_tracked")
    output_filepath = output_vidpath.with_suffix(".csv")
    codec = "mp4v"
    n_inds = 1
    colours = [
        (0, 0, 255),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 255),
        (255, 255, 0),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 0),
    ]
    scaling = 1.0
    min_area = 470
    max_area = 8000
    mot = False
    kernel = np.ones((5, 5), np.uint8)

    # Create a background subtractor
    fgbg5 = cv2.bgsegm.createBackgroundSubtractorGSOC()

    # Generate a background image
    bg_path = generate_background_image(input_vidpath, fgbg5, scaling, codec)

    # Track objects
    track_objects(
        input_vidpath,
        output_vidpath,
        output_filepath,
        codec,
        n_inds,
        colours,
        scaling,
        min_area,
        max_area,
        mot,
        kernel,
        bg_path,
    )


def generate_background_image(input_vidpath, fgbg5, scaling, codec):
    """Generate a background image for the given video."""
    cap = cv2.VideoCapture(input_vidpath.as_posix())
    frame_size = (
        int(cap.read()[1].shape[1] * scaling),
        int(cap.read()[1].shape[0] * scaling),
    )
    fourcc = cv2.VideoWriter_fourcc(*codec)
    bg_gen_path = input_vidpath.parent.joinpath("Background_Generator.mp4")

    # Write a video with random frames taken in the input video
    bg_generator = cv2.VideoWriter(
        filename=bg_gen_path.as_posix(), fourcc=fourcc, fps=80, frameSize=frame_size
    )
    try:
        for _ in range(300):
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            random_frame_number = random.randint(0, total_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)
            success, image = cap.read()

            if success:
                bg_generator.write(image)
                fgbg5.apply(image)  # Apply the background subtractor to the frame
    finally:
        bg_generator.release()

    cap.release()

    # Create a background image from the generated video
    cap = cv2.VideoCapture(bg_gen_path.as_posix())
    bg_path = input_vidpath.parent.joinpath("Background.jpg")

    while True:
        ret, img = cap.read()
        if not ret:
            break

        fgbg5.apply(img)  # Apply the background subtractor to the frame

    # Now you can get the background image
    bg = fgbg5.getBackgroundImage()
    cv2.imwrite(bg_path.as_posix(), bg)

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    return bg_path


def track_objects(
    input_vidpath,
    output_vidpath,
    output_filepath,
    codec,
    n_inds,
    colours,
    scaling,
    min_area,
    max_area,
    mot,
    kernel,
    bg_path,
):
    """Track objects in the given video and write the positions to a CSV file."""
    background = cv2.imread(bg_path.as_posix())
    cap = cv2.VideoCapture(input_vidpath.as_posix())
    if not cap.isOpened():
        sys.exit(
            "Video file cannot be read! Please check input_vidpath to ensure it is correctly pointing to the video file"
        )

    frame_size = (
        int(cap.read()[1].shape[1] * scaling),
        int(cap.read()[1].shape[0] * scaling),
    )
    fourcc = cv2.VideoWriter_fourcc(*codec)
    meas_last = list(np.zeros((n_inds, 2)))
    meas_now = list(np.zeros((n_inds, 2)))
    df = []

    with cv2.VideoWriter(
        filename=output_vidpath.as_posix(),
        fourcc=fourcc,
        fps=80.0,
        frameSize=frame_size,
        isColor=True,
    ) as out:
        last = 0

        while True:
            ret, frame = cap.read()
            this = cap.get(1)

            if ret:
                frame = cv2.resize(
                    frame, None, fx=scaling, fy=scaling, interpolation=cv2.INTER_LINEAR
                )
                subtracted = cv2.absdiff(frame, background)
                thresh = tr.colour_to_thresh_binary(subtracted, 22)
                final, contours, meas_last, meas_now = tr.detect_and_draw_contours(
                    frame, thresh, meas_last, meas_now, min_area, max_area
                )
                row_ind, col_ind = tr.hungarian_algorithm(meas_last, meas_now)
                final, meas_now, df = tr.reorder_and_draw(
                    final, colours, n_inds, col_ind, meas_now, df, mot, this
                )

                for i in range(n_inds):
                    df.append([this, meas_now[i][0], meas_now[i][1]])

                out.write(final)

                if (
                    cv2.waitKey(1) == 27
                    or meas_now[0][0] < 20
                    or meas_now[0][0] > cap.get(3) - 20
                    or meas_now[0][1] < 20
                    or meas_now[0][1] > cap.get(4) - 20
                ):
                    break

            if last >= this:
                break

            last = this

    # Write positions to file
    df = pd.DataFrame(np.matrix(df), columns=["frame", "pos_x", "pos_y"])
    df.to_csv(output_filepath.as_posix(), sep=",")

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
