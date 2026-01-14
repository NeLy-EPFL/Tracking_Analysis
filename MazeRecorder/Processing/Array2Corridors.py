from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2
import re
from tqdm import tqdm
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import shutil
from itertools import repeat
import subprocess
import sys
import os
from joblib import Parallel, delayed
import gc
import multiprocessing as mp

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
    """Check which folders need processing and process them."""
    data_folder = Path(data_folder)
    if not data_folder.exists():
        print(f"Error: Data folder {data_folder} does not exist.")
        return
        
    for folder in data_folder.iterdir():
        if (
            folder.is_dir()
            and not folder.name.endswith("_Cropped")
            and not folder.name.endswith("_Processing")
            and not folder.name.endswith("_Checked")
            and folder.name.endswith("_Recorded")
        ):
            cropped_folder = folder.with_name(folder.stem.replace("_Recorded", "_Cropped"))
            checked_folder = folder.with_name(folder.stem.replace("_Recorded", "_Checked"))
            pending_folder = folder.with_name(folder.stem.replace("_Recorded", "_Processing"))
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
                print(
                    f"{folder.name} is currently being processed."
                )
            else:
                print(f"{folder.name} is not processed. Processing...")
                try:
                    process_folder(folder)
                except Exception as e:
                    print(f"Error processing {folder.name}: {e}")
                    print("Cleaning up incomplete processing folder...")
                    # Clean up any incomplete processing folder
                    if pending_folder.exists():
                        import shutil
                        shutil.rmtree(pending_folder)
                        print(f"Removed incomplete processing folder: {pending_folder}")
                    raise  # Re-raise the exception to stop execution


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
    """Process a single image: crop according to corridors and save."""
    try:
        # Read and process the image
        frame = cv2.imread(str(folder / image))
        if frame is None:
            print(f"Warning: Could not load image {image}")
            return
            
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
    
    except Exception as e:
        print(f"Error processing {image}: {e}")
    finally:
        # Explicitly clean up large variables
        if 'frame' in locals():
            del frame


def process_image_batch(image_batch, Corridors, folder, processedfolder):
    """Process a batch of images to reduce overhead."""
    for image in image_batch:
        process_image(image, Corridors, folder, processedfolder)


def process_folder(in_folder):
    inputfolder = in_folder

    folder = inputfolder

    # Function to extract numerical value from filename for sorting
    def extract_number(filename):
        numbers = re.findall(r'\d+', filename.stem)
        return int(numbers[0]) if numbers else 0

    # Sort files numerically based on extracted number
    image_files = sorted(inputfolder.glob('*.[jJ][pP][gG]'), key=extract_number)

    if not image_files:
        print("No image files found in the folder.")
        return

    processedfolder = inputfolder.with_name(inputfolder.stem.replace("_Recorded", "_Processing"))
    processedfolder.mkdir(exist_ok=True)

    # Debugging: Print the last image file to be processed
    print(f"Last image file to be processed: {image_files[-1]}")

    # Load the last frame
    frame = cv2.imread(str(image_files[-1]))
    if frame is None:
        print(f"Error: Could not load image {image_files[-1]}")
        return

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
        
        # Before creating subcors, check if Colpos and Rowpos have the expected number of elements
        expected_num_peaks = 12  # Adjust based on your expectations
        if len(Colpos) < expected_num_peaks or len(Rowpos) < 2:
            error_msg = f"ERROR: Peak detection failed for region {i+1}. Found {len(Colpos)} column peaks and {len(Rowpos)} row peaks, but expected at least {expected_num_peaks} column peaks and 2 row peaks."
            print(error_msg)
            print(f"Cannot proceed with processing {in_folder.name} - all regions must have valid peak detection.")
            
            # Clean up the processing folder if it was created
            if processedfolder.exists():
                import shutil
                shutil.rmtree(processedfolder)
                print(f"Cleaned up incomplete processing folder: {processedfolder}")
            
            raise RuntimeError(error_msg)

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

    # Verify all corridors were successfully detected
    if len(Corridors) != len(regions_of_interest):
        error_msg = f"ERROR: Expected {len(regions_of_interest)} corridor sets, but only got {len(Corridors)}. Peak detection failed."
        print(error_msg)
        
        # Clean up the processing folder
        if processedfolder.exists():
            import shutil
            shutil.rmtree(processedfolder)
            print(f"Cleaned up incomplete processing folder: {processedfolder}")
        
        raise RuntimeError(error_msg)

    # Verify each corridor set has the expected number of corridors
    expected_corridors_per_set = 6
    for i, corridor_set in enumerate(Corridors):
        if len(corridor_set) != expected_corridors_per_set:
            error_msg = f"ERROR: Arena {i+1} has {len(corridor_set)} corridors, expected {expected_corridors_per_set}."
            print(error_msg)
            
            # Clean up the processing folder
            if processedfolder.exists():
                import shutil
                shutil.rmtree(processedfolder)
                print(f"Cleaned up incomplete processing folder: {processedfolder}")
            
            raise RuntimeError(error_msg)

    print(f"SUCCESS: All {len(Corridors)} regions detected with {expected_corridors_per_set} corridors each.")

    # Create visualization of all detected corridors
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
    plt.close()  # Close the figure to free memory

    # Get a list of all image files in the input folder
    images = [f.name for f in folder.glob("*.[jJ][pP][gG]") if f.is_file()]

    # Sort the list of images by their number (fix the regex syntax)
    images.sort(key=lambda f: int(re.sub(r"\D", "", f)))

    # Create the subfolders for each arena and corridor
    for j, subset in enumerate(Corridors):
        for k, corridor in enumerate(subset):
            subfolder = processedfolder / f"arena{j+1}" / f"corridor{k+1}"
            subfolder.mkdir(parents=True, exist_ok=True)

    # Check if standard input is connected to a terminal
    is_tty = os.isatty(sys.stdin.fileno())

    print(f"Processing {len(images)} images...")

    # Process images in batches to reduce memory usage and overhead
    batch_size = max(1, len(images) // (mp.cpu_count() * 4))  # Adaptive batch size
    image_batches = [images[i:i + batch_size] for i in range(0, len(images), batch_size)]
    
    # Prepare the arguments for the process_image_batch function
    args = [(batch, Corridors, folder, processedfolder) for batch in image_batches]

    # Use threading backend for better resource management
    try:
        with Parallel(n_jobs=-1, backend='threading', batch_size=1) as parallel:
            results = parallel(delayed(process_image_batch)(*arg) for arg in tqdm(args, disable=not is_tty))
    except Exception as e:
        print(f"Error during parallel processing: {e}")
        print("Falling back to sequential processing...")
        # Fallback to sequential processing
        for batch in tqdm(image_batches, disable=not is_tty):
            process_image_batch(batch, Corridors, folder, processedfolder)
    
    # Force garbage collection to clean up any remaining resources
    gc.collect()
            
    # rename the processed folder from _processing to _Cropped
    croppedfolder = processedfolder.with_name(processedfolder.stem.replace("_Processing", "_Cropped"))
    processedfolder.rename(croppedfolder)
    print(f"Processing of {in_folder.name} finished! Folder renamed to {croppedfolder.name}")


check_process(datafolder)


# Main execution
if __name__ == "__main__":
    # Set multiprocessing start method to avoid resource tracker issues
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Start method already set

    try:
        check_process(datafolder)

        if os.isatty(sys.stdin.fileno()):
            run_checkcrops = input(
                "Launch verification of processed folders integrity? (y/n): "
            )
            if run_checkcrops.lower() == "y":
                # Get the directory of the current script
                script_dir = Path(__file__).resolve().parent
                # Construct the path to CheckCrops.sh
                checkcrops_path = script_dir / "CheckCrops.sh"
                # Run the script
                if checkcrops_path.exists():
                    subprocess.run([str(checkcrops_path)])
                else:
                    print(f"CheckCrops.sh not found at {checkcrops_path}")
    
    finally:
        # Final cleanup
        gc.collect()
