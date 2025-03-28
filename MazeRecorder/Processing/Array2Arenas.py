from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2
import re
from tqdm import tqdm
import cv2
import numpy as np
from pathlib import Path
from scipy import signal
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil
from itertools import repeat
import subprocess
import sys
import os
from joblib import Parallel, delayed

# from multiprocessing import Pool
# from multiprocessing import set_start_method
# if __name__ == '__main__':
#     set_start_method('spawn')


# Path definitions

datafolder = Path("/home/matthias/Videos/")
# For directories and subdirectories within the datafolder, if they contain images and do not have '_Cropped' in their name, add them to the list of folders to process

# Function used by the multiprocessing Pool
# def process_image_wrapper(args):
#         return process_image(*args)


def check_process(data_folder):
    data_folder = Path(data_folder)
    for folder in data_folder.iterdir():
        if (
            folder.is_dir()
            and not folder.name.endswith("_Cropped")
            and not folder.name.endswith("_Processing")
            and not folder.name.endswith("_Checked")
            and folder.name.endswith("_Recorded")
        ):
            cropped_folder = folder.with_name(
                folder.stem.replace("_Recorded", "_Cropped")
            )
            checked_folder = folder.with_name(
                folder.stem.replace("_Recorded", "_Checked")
            )
            pending_folder = folder.with_name(
                folder.stem.replace("_Recorded", "_Processing")
            )
            cropped_folder_path = data_folder / cropped_folder
            checked_folder_path = data_folder / checked_folder
            if cropped_folder_path.exists():
                print(
                    f"{folder.name} is already processed but its integrity is not verified."
                )
            elif checked_folder_path.exists():
                print(
                    f"{folder.name} is already processed and its integrity is verified."
                )
            elif pending_folder.exists():
                print(f"{folder.name} is currently being processed.")
            else:
                print(f"{folder.name} is not processed. Processing...")
                process_folder(folder)


def process_image(image, regions_of_interest, folder, processedfolder):
    # Read and process the image
    frame = cv2.imread(str(folder / image))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert the folder name to lowercase
    folder_name = str(folder).lower()

    # Check if the folder name contains '_flip', 'rotatel' or 'rotater' and rotate the image accordingly
    if "_flip" in folder_name:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif "rotatel" in folder_name:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif "rotater" in folder_name:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    for j, region in enumerate(regions_of_interest):
        # Get the subfolder for this arena
        subfolder = processedfolder / f"arena{j+1}"

        # Crop the image
        x1, y1, x2, y2 = region
        width = x2 - x1
        height = y2 - y1

        # Adjust the width and height to be multiples of 2
        if width % 2 != 0:
            width -= 1
        if height % 2 != 0:
            height -= 1

        cropped_image = frame[y1 : y1 + height, x1 : x1 + width]

        # Save the cropped image
        cropped_image_file = f"{Path(image).stem}_cropped.jpg"
        cv2.imwrite(str(subfolder / cropped_image_file), cropped_image)


def process_folder(in_folder):
    inputfolder = in_folder

    # Create a list of all the images in the target folder
    folder = inputfolder

    processedfolder = inputfolder.with_name(
        inputfolder.stem.replace("_Recorded", "_Processing")
    )

    # Create the subfolder if it doesn't exist
    processedfolder.mkdir(exist_ok=True)

    # Load the first frame
    frame = cv2.imread(inputfolder.joinpath("image0.jpg").as_posix())

    # If it's not already, make it grayscale
    if len(frame.shape) > 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert the folder name to lowercase
    folder_name = str(folder).lower()

    # Check if the folder name contains '_flip', 'rotatel' or 'rotater' and rotate the image accordingly
    if "_flip" in folder_name:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif "rotatel" in folder_name:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif "rotater" in folder_name:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # equalize the histogram to make thresholding easier
    frame = cv2.equalizeHist(frame)

    # Crop the image to the regions of interest

    X1 = 90
    X2 = 620
    X3 = 1590
    X4 = 2110
    X5 = 3080
    X6 = 3600

    Y1 = 30
    Y2 = 600
    Y3 = 1170
    Y4 = 1750
    Y5 = 2370
    Y6 = 2900

    regions_of_interest = [
        (X1, Y1, X2, Y2),
        (X3, Y1, X4, Y2),
        (X5, Y1, X6, Y2),
        (X1, Y3, X2, Y4),
        (X3, Y3, X4, Y4),
        (X5, Y3, X6, Y4),
        (X1, Y5, X2, Y6),
        (X3, Y5, X4, Y6),
        (X5, Y5, X6, Y6),
    ]

    fig, axs = plt.subplots(3, 3, figsize=(20, 20))
    for i in range(9):
        axs[i // 3, i % 3].axis("off")
        axs[i // 3, i % 3].imshow(
            frame[
                regions_of_interest[i][1] : regions_of_interest[i][3],
                regions_of_interest[i][0] : regions_of_interest[i][2],
            ],
            cmap="gray",
            vmin=0,
            vmax=255,
        )

    # Remove the axis of each subplot and draw them closer together
    for ax in axs.flat:
        ax.axis("off")
    plt.subplots_adjust(wspace=0, hspace=0)
    # Save the figure in the output folder
    plt.savefig(
        str(processedfolder.joinpath("crop_check.png")), dpi=300, bbox_inches="tight"
    )

    # Get a list of all image files in the input folder
    images = [f.name for f in folder.glob("*.[jJ][pP][gG]") if f.is_file()]

    # Sort the list of images by their number
    images.sort(key=lambda f: int(re.sub("\D", "", f)))

    # Create the subfolders for each arena
    for j in range(len(regions_of_interest)):
        subfolder = processedfolder / f"arena{j+1}"
        subfolder.mkdir(parents=True, exist_ok=True)

    # Check if standard input is connected to a terminal
    is_tty = os.isatty(sys.stdin.fileno())

    # Prepare the arguments for the process_image function
    args = [(image, regions_of_interest, folder, processedfolder) for image in images]

    # Use joblib to run the process_image function in parallel
    results = Parallel(n_jobs=-1)(delayed(process_image)(*arg) for arg in tqdm(args))

    # rename the processed folder from _processing to _Cropped
    croppedfolder = processedfolder.with_name(
        processedfolder.stem.replace("_Processing", "_Cropped")
    )
    processedfolder.rename(croppedfolder)
    print(
        f"Processing of {in_folder.name} finished! Folder renamed to {croppedfolder.name}"
    )


check_process(datafolder)
#process_folder(datafolder / "240_F1_Recorded")


if os.isatty(sys.stdin.fileno()):
    run_checkcrops = input(
        "Launch verification of processed folders integrity? (y/n): "
    )
    if run_checkcrops.lower() == "y":
        subprocess.run(
            [
                "/home/matthias/Tracking_Analysis/Ball_Pushing/MazeRecorder/Processing/CheckCrops.sh"
            ]
        )
