import numpy as np
import pandas as pd
import sys

sys.path.insert(0, "../../Tracktor")

import tracktor as tr
import cv2
import sys
import os
import random
from pathlib import Path

# Specify where to look for shell ffmpeg
# Ffmpeg = "/usr/local/Cellar/ffmpeg/5.0-with-options_1/bin/ffmpeg"

DataPath = Path(
    "/mnt/labserver/DURRIEU_Matthias/Experimental_data/Irene_Optobot/20_days/Irene_20d/SynjRQ-Atg18/m3_20d/221220/171357_s0a0_p6-0"
)


Filters = ["Results", "BadExp", "Trimmed", "tracked"]

Files = [
    path
    for path in DataPath.rglob("*.mp4")
    if any(match in path.as_posix() for match in Filters) is False
]

for file in Files:
    print(file.as_posix())

    path = file.parent
    source = file.name

    # Here are the arguments to set in order to trim out the part where the arena is opened initially.

    startpoint = "00:00:00"  # Start point is the timepoint where the arena is fully opened and static
    finishpoint = "00:10:00"  # Finish point is the last timepoint of the video
    TrimmedPath = file.with_stem(file.stem + "_Trimmed")

    os.system(  # Ffmpeg +
        "ffmpeg -hide_banner -loglevel error -i "
        + file.as_posix()
        + " -ss "
        + startpoint
        + " -to "
        + finishpoint
        + " -c copy "
        + TrimmedPath.as_posix()
    )

    input_vidpath = TrimmedPath
    output_vidpath = file.with_stem(file.stem + "_tracked")
    output_filepath = output_vidpath.with_suffix(".csv")
    codec = "mp4v"
    # try other codecs if the default doesn't work ('DIVX', 'avc1', 'XVID') note: this list is non-exhaustive

    # colours is a vector of BGR values which are used to identify individuals in the video
    # number of elements in colours should be greater than n_inds (THIS IS NECESSARY FOR VISUALISATION ONLY)
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

    # this is the block_size and offset used for adaptive thresholding (block_size should always be odd)
    # these values are critical for tracking performance
    # block_size = 35 # These should be only set if using adaptive thresholding
    # offset = 10

    # the scaling parameter can be used to speed up tracking if video resolution is too high (use value 0-1)
    scaling = 1.0

    # minimum area and maximum area occupied by the animal in number of pixels
    # this is used to get rid of other objects in view that might be hard to threshold out but are differently sized
    min_area = 470
    max_area = 8000

    # mot determines whether the tracker is being used in noisy conditions to track a single object or for multi-object
    # using this will enable k-means clustering to force n_inds number of animals
    mot = False

    # kernel for erosion and dilation
    # useful since thin spider limbs are sometimes detected as separate objects
    kernel = np.ones((5, 5), np.uint8)

    fgbg5 = cv2.bgsegm.createBackgroundSubtractorGSOC(
        # nSamples=2,
        # replaceRate=0.900,
        # propagationRate=0.003 ,
        # noiseRemovalThresholdFacFG = 0.45,
        # noiseRemovalThresholdFacBG= 0.45,
        # hitsThreshold=5
    )  # default : 0.003

    cap = cv2.VideoCapture(input_vidpath.as_posix())

    # Set framesize as the same one as the images read from input video
    BG_framesize = (
        int(cap.read()[1].shape[1] * scaling),
        int(cap.read()[1].shape[0] * scaling),
    )

    fourcc = cv2.VideoWriter_fourcc(*codec)
    # Create a Video writer with the desired parameters

    BGen_Path = file.parent.joinpath("Background_Generator.mp4")

    Background_Generator = cv2.VideoWriter(
        filename=BGen_Path.as_posix(),
        fourcc=fourcc,
        fps=80,
        frameSize=BG_framesize,
        # isColor=True,
    )

    # Write a video with random frames taken in the input video
    f = 0
    while f <= 600:
        # get total number of frames
        totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        randomFrameNumber = random.randint(0, totalFrames)
        # set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, randomFrameNumber)
        success, image = cap.read()

        if success:

            Background_Generator.write(image)

        f += 1

    cap.release()
    Background_Generator.release()
    cv2.destroyAllWindows()

    # Adjust using live rendering to get a clean background image. Default : 500

    cap = cv2.VideoCapture(BGen_Path.as_posix())

    target = 0
    cap.set(
        1, target
    )  # Set the starting point, try to find a section where the fly moves a lot.

    BgPath = file.parent.joinpath("Background.jpg")
    Frame = target + 300

    while 1:
        # read frames
        ret, img = cap.read()
        this = cap.get(1)

        # apply mask for background subtraction
        # fgbg5 is a GSOC background subtraction algorithm
        fgmask5 = fgbg5.apply(img)

        bg = fgbg5.getBackgroundImage()

        # cv2.imshow("Original", img)
        # cv2.imshow("GSOC", fgmask5)
        # cv2.imshow("background", bg)
        subtracted = cv2.absdiff(img, bg)

        if this == Frame:

            cv2.imwrite(BgPath.as_posix(), bg)
            break

        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    Background = cv2.imread(BgPath.as_posix())
    BGen_Path.unlink()

    # Open video
    cap = cv2.VideoCapture(input_vidpath.as_posix())
    if not cap.isOpened():
        sys.exit(
            "Video file cannot be read! Please check input_vidpath to ensure it is correctly pointing to the video file"
        )

    # Video writer class to output video with contour and centroid of tracked object(s)
    # make sure the frame size matches size of array 'final'
    fourcc = cv2.VideoWriter_fourcc(*codec)
    output_framesize = (
        int(cap.read()[1].shape[1] * scaling),
        int(cap.read()[1].shape[0] * scaling),
    )
    out = cv2.VideoWriter(
        filename=output_vidpath.as_posix(),
        fourcc=fourcc,
        fps=80.0,
        frameSize=output_framesize,
        isColor=True,
    )

    # Individual location(s) measured in the last and current step
    meas_last = list(np.zeros((n_inds, 2)))
    meas_now = list(np.zeros((n_inds, 2)))

    last = 0
    df = []

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        this = cap.get(1)
        if ret:
            frame = cv2.resize(
                frame, None, fx=scaling, fy=scaling, interpolation=cv2.INTER_LINEAR
            )
            subtracted = cv2.absdiff(frame, Background)
            thresh = tr.colour_to_thresh_binary(subtracted, 22)

            # Lines below to produce an erode/dilate kernel if needed  to remove small noisy objects like hairs.
            # thresh = cv2.erode(thresh, kernel, iterations = 1)
            # thresh = cv2.dilate(thresh, kernel, iterations = 1)
            final, contours, meas_last, meas_now = tr.detect_and_draw_contours(
                frame, thresh, meas_last, meas_now, min_area, max_area
            )
            row_ind, col_ind = tr.hungarian_algorithm(meas_last, meas_now)
            final, meas_now, df = tr.reorder_and_draw(
                final, colours, n_inds, col_ind, meas_now, df, mot, this
            )

            # Create output dataframe
            for i in range(n_inds):
                df.append([this, meas_now[i][0], meas_now[i][1]])

            # Display the resulting frame
            out.write(final)
            # cv2.imshow("frame", final)
            # cv2.imshow("subtracted", subtracted)
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

    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
