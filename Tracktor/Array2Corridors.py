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


# Path definitions

datafolder = Path("/home/matthias/Videos/")
# For directories and subdirectories within the datafolder, if they contain images and do not have '_Cropped' in their name, add them to the list of folders to process


def check_process(data_folder):
    data_folder = Path(data_folder)
    for folder in data_folder.iterdir():
        if (
            folder.is_dir()
            and not folder.name.endswith("_Cropped")
            and not folder.name.endswith("_Checked")
            and folder.name.endswith("_Recorded")
        ):
            cropped_folder = folder.name + "_Cropped"
            checked_folder = folder.name + "_Checked"
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
            else:
                print(f"{folder.name} is not processed. Processing...")
                process_folder(folder)



def modify_corridors(Corridors):
    for i in range(len(Corridors)):
        for j in range(len(Corridors[i])):
            corridor = list(Corridors[i][j])
            height = corridor[3] - corridor[1]
            width = corridor[2] - corridor[0]
            if height % 2 != 0:
                corridor[3] += 1
            if width % 2 != 0:
                corridor[2] += 1
            Corridors[i][j] = tuple(corridor)
    return Corridors


def process_image(image, Corridors, folder, processedfolder):
    # Read and process the image
    frame = cv2.imread(str(folder / image))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert the folder name to lowercase
    folder_name = str(folder).lower()

    # Check if the folder name contains '_flip', 'rotatel' or 'rotater' and rotate the image accordingly
    if '_flip' in folder_name:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif 'rotatel' in folder_name:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif 'rotater' in folder_name:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    for j, subset in enumerate(Corridors):
        for k, corridor in enumerate(subset):
            # Get the subfolder for this arena and corridor
            subfolder = processedfolder / f"arena{j+1}" / f"corridor{k+1}"

            # Crop the image
            x1, y1, x2, y2 = corridor
            cropped_image = frame[y1:y2, x1:x2]

            # Save the cropped image
            cropped_image_file = f"{Path(image).stem}_cropped.jpg"
            cv2.imwrite(str(subfolder / cropped_image_file), cropped_image)


def process_folder(in_folder):
    inputfolder = in_folder

    # Create a list of all the images in the target folder
    folder = inputfolder
    
    processedfolder = inputfolder.with_name(inputfolder.stem.replace("_Recorded", "_Cropped"))

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
    if '_flip' in folder_name:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif 'rotatel' in folder_name:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif 'rotater' in folder_name:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # equalize the histogram to make thresholding easier
    frame = cv2.equalizeHist(frame)

    # Crop the image to the regions of interest

    X1 = 0
    X2 = 620
    X3 = 1450
    X4 = 2130
    X5 = 2980
    X6 = 3590

    Y1 = 0
    Y2 = 725
    Y3 = 1140
    Y4 = 1860
    Y5 = 2350
    Y6 = 2995

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

    # For each subset, find the cols and rows peaks and store the even peaks in a list
    Corridors = []
    for i in range(len(regions_of_interest)):
        subset = np.array(
            frame[
                regions_of_interest[i][1] : regions_of_interest[i][3],
                regions_of_interest[i][0] : regions_of_interest[i][2],
            ]
        )
        Thresh = subset.copy()
        # Apply an adaptive threshold to each subset to keep only the brightest pixels
        Thresh = cv2.adaptiveThreshold(
            Thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 4
        )

        cols = Thresh.sum(axis=0)
        rows = subset.sum(axis=1)

        colpeaks = signal.find_peaks(
            cols,
            distance=20,
            height=(80_000, 120_000),
            # width=(5, 30),
        )
        colpeaks = (colpeaks[0], colpeaks[1])

        rowpeaks = signal.find_peaks(
            rows,
            distance=300,
        )
        #######################################################
        Colpos = []
        Rowpos = []

        for peak_index in colpeaks[0]:
            peak_x = regions_of_interest[i][0] + peak_index
            peak_y = np.argmax(subset[peak_index])
            peak_y += regions_of_interest[i][1]
            Colpos.append((peak_x, peak_y))

        for peak_index in rowpeaks[0]:
            peak_x = np.argmax(subset[peak_index])
            peak_x += regions_of_interest[i][0]
            peak_y = regions_of_interest[i][1] + peak_index
            Rowpos.append((peak_x, peak_y))

        bound_x = 30
        bound_y = 60

        subcors = [
            (
                Colpos[0][0] - bound_x,
                Rowpos[0][1] - bound_y,
                Colpos[1][0] + bound_x,
                Rowpos[1][1] + bound_y,
            ),
            (
                Colpos[2][0] - bound_x,
                Rowpos[0][1] - bound_y,
                Colpos[3][0] + bound_x,
                Rowpos[1][1] + bound_y,
            ),
            (
                Colpos[4][0] - bound_x,
                Rowpos[0][1] - bound_y,
                Colpos[5][0] + bound_x,
                Rowpos[1][1] + bound_y,
            ),
            (
                Colpos[6][0] - bound_x,
                Rowpos[0][1] - bound_y,
                Colpos[7][0] + bound_x,
                Rowpos[1][1] + bound_y,
            ),
            (
                Colpos[8][0] - bound_x,
                Rowpos[0][1] - bound_y,
                Colpos[9][0] + bound_x,
                Rowpos[1][1] + bound_y,
            ),
            (
                Colpos[10][0] - bound_x,
                Rowpos[0][1] - bound_y,
                Colpos[11][0] + bound_x,
                Rowpos[1][1] + bound_y,
            ),
        ]

        # subcors = generate_subsets(subset, regions_of_interest)

        Corridors.append(subcors)

    Corridors = modify_corridors(Corridors)

    fig, axs = plt.subplots(9, 6, figsize=(20, 20))
    for i in range(9):
        for j in range(6):
            axs[i, j].axis("off")
            axs[i, j].imshow(
                frame[
                    Corridors[i][j][1] : Corridors[i][j][3],
                    Corridors[i][j][0] : Corridors[i][j][2],
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

    # Create the subfolders for each arena and corridor
    for j, subset in enumerate(Corridors):
        for k, corridor in enumerate(subset):
            subfolder = processedfolder / f"arena{j+1}" / f"corridor{k+1}"
            subfolder.mkdir(parents=True, exist_ok=True)

    # Process the images in parallel using a process pool
    with ProcessPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(
                    process_image,
                    images,
                    repeat(Corridors),
                    repeat(folder),
                    repeat(processedfolder),
                ),
                total=len(images),
            )
        )

    print(f"Processing of {in_folder.name} finished!")


check_process(datafolder)

run_checkcrops = input(
    "Launch verification of processed folders integrity? (y/n): "
)
if run_checkcrops.lower() == "y":
    subprocess.run(["/home/matthias/Tracking_Analysis/Tracktor/CheckCrops.sh"])
