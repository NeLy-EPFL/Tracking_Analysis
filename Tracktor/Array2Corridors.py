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

# Path definitions

datafolder = Path("/Volumes/Ramdya-Lab/labserver/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Videos/")

# For directories and subdirectories within the datafolder, if they contain images and do not have '_Cropped' in their name, add them to the list of folders to process
                                
inputfolder = datafolder.joinpath("/Feeding_state/230704_FeedingState_1_AM/")

# Load the first frame
frame = cv2.imread(inputfolder.joinpath("image0.jpg").as_posix())

# If it's not already, make it grayscale
if len(frame.shape) > 2:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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


Corridors = modify_corridors(Corridors)

# Create a list of all the images in the target folder
folder = inputfolder
processedfolder = inputfolder.parent / f"{inputfolder.stem}_Cropped"

# Create the subfolder if it doesn't exist
processedfolder.mkdir(exist_ok=True)

def process_image(image):
    # Read and process the image
    frame = cv2.imread(str(folder / image))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frame = np.rot90(frame)

    for j, subset in enumerate(Corridors):
        for k, corridor in enumerate(subset):
            # Get the subfolder for this arena and corridor
            subfolder = processedfolder / f"arena{j+1}" / f"corridor_{k+1}"

            # Crop the image
            x1, y1, x2, y2 = corridor
            cropped_image = frame[y1:y2, x1:x2]

            # Save the cropped image
            cropped_image_file = f"{Path(image).stem}_cropped.jpg"
            cv2.imwrite(str(subfolder / cropped_image_file), cropped_image)



# Create the subfolder if it doesn't exist
processedfolder.mkdir(exist_ok=True)

# Get a list of all image files in the input folder
images = [f.name for f in folder.glob("*.[jJ][pP][gG]") if f.is_file()]

# Sort the list of images by their number
images.sort(key=lambda f: int(re.sub("\D", "", f)))

# Create the subfolders for each arena and corridor
for j, subset in enumerate(Corridors):
    for k, corridor in enumerate(subset):
        subfolder = processedfolder / f"arena{j+1}" / f"corridor_{k+1}"
        subfolder.mkdir(parents=True, exist_ok=True)

# Process the images in parallel using a process pool
with ProcessPoolExecutor() as executor:
    results = list(tqdm(executor.map(process_image, images), total=len(images)))

