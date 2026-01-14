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
import json
import argparse
from joblib import Parallel, delayed
import gc
import multiprocessing as mp

# Path definitions
datafolder = Path("/home/matthias/Videos/")

# Arena coordinate definitions (used by both test and processing functions)
ARENA_COORDS = {
    'X1': 60, 'X2': 620,
    'X3': 1560, 'X4': 2110, 
    'X5': 3055, 'X6': 3600,
    'Y1': 30, 'Y2': 600,
    'Y3': 1200, 'Y4': 1770,
    'Y5': 2370, 'Y6': 2920
}

# Generate regions of interest from coordinates
REGIONS_OF_INTEREST = [
    (ARENA_COORDS['X1'], ARENA_COORDS['Y1'], ARENA_COORDS['X2'], ARENA_COORDS['Y2']),  # Arena 1
    (ARENA_COORDS['X3'], ARENA_COORDS['Y1'], ARENA_COORDS['X4'], ARENA_COORDS['Y2']),  # Arena 2  
    (ARENA_COORDS['X5'], ARENA_COORDS['Y1'], ARENA_COORDS['X6'], ARENA_COORDS['Y2']),  # Arena 3
    (ARENA_COORDS['X1'], ARENA_COORDS['Y3'], ARENA_COORDS['X2'], ARENA_COORDS['Y4']),  # Arena 4
    (ARENA_COORDS['X3'], ARENA_COORDS['Y3'], ARENA_COORDS['X4'], ARENA_COORDS['Y4']),  # Arena 5
    (ARENA_COORDS['X5'], ARENA_COORDS['Y3'], ARENA_COORDS['X6'], ARENA_COORDS['Y4']),  # Arena 6
    (ARENA_COORDS['X1'], ARENA_COORDS['Y5'], ARENA_COORDS['X2'], ARENA_COORDS['Y6']),  # Arena 7
    (ARENA_COORDS['X3'], ARENA_COORDS['Y5'], ARENA_COORDS['X4'], ARENA_COORDS['Y6']),  # Arena 8
    (ARENA_COORDS['X5'], ARENA_COORDS['Y5'], ARENA_COORDS['X6'], ARENA_COORDS['Y6']),  # Arena 9
]


def test_process_folder(folder_path, num_images=5):
    """Test processing on a specific folder with a limited number of images."""
    folder = Path(folder_path)
    
    if not folder.exists() or not folder.is_dir():
        print(f"Error: Folder {folder} does not exist or is not a directory.")
        return
    
    if not folder.name.endswith("_Recorded"):
        print(f"Error: Folder {folder.name} doesn't appear to be a recorded folder (should end with '_Recorded').")
        return
    
    print(f"Testing processing on folder: {folder.name}")
    print(f"Will process first {num_images} images...")
    
    # Create test processing folder
    test_processedfolder = folder.with_name(folder.stem.replace("_Recorded", "_Test_Processing"))
    
    # Clean up any existing test folder
    if test_processedfolder.exists():
        import shutil
        shutil.rmtree(test_processedfolder)
    
    test_processedfolder.mkdir(exist_ok=True)

    # Load the first frame for arena detection
    first_image = folder / "image0.jpg"
    if not first_image.exists():
        # Try to find any jpg file
        jpg_files = list(folder.glob("*.jpg"))
        if not jpg_files:
            print(f"Error: No jpg files found in {folder}")
            return
        first_image = jpg_files[0]
    
    frame = cv2.imread(str(first_image))
    if frame is None:
        print(f"Error: Could not load image {first_image}")
        return

    # If it's not already, make it grayscale
    if len(frame.shape) > 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert the folder name to lowercase
    folder_name = str(folder).lower()

    # Apply global rotation based on folder name
    if "_flip" in folder_name:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif "rotatel" in folder_name:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif "rotater" in folder_name:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Equalize the histogram to make thresholding easier
    frame = cv2.equalizeHist(frame)

    # Use shared arena coordinates
    regions_of_interest = REGIONS_OF_INTEREST

    # Get orientation information for each arena
    orientations = []
    for i in range(len(regions_of_interest)):
        orientation = get_orientation_from_metadata(folder, i)
        orientations.append(orientation)
        print(f"Arena {i+1} orientation: {orientation}")

    # Create visualization of detected arenas with split preview
    fig, axs = plt.subplots(3, 9, figsize=(30, 10))  # 3 rows, 9 columns (arena + left + right for each)
    
    for i in range(9):
        # Show original arena
        arena_crop = frame[
            regions_of_interest[i][1] : regions_of_interest[i][3],
            regions_of_interest[i][0] : regions_of_interest[i][2],
        ]
        axs[0, i].imshow(arena_crop, cmap="gray", vmin=0, vmax=255)
        axs[0, i].set_title(f"Arena {i+1} ({orientations[i]})")
        axs[0, i].axis("off")
        
        # Process the arena to show left and right splits
        x1, y1, x2, y2 = regions_of_interest[i]
        width = x2 - x1
        height = y2 - y1

        # Adjust the width and height to be multiples of 2
        if width % 2 != 0:
            width -= 1
        if height % 2 != 0:
            height -= 1

        arena_image = frame[y1 : y1 + height, x1 : x1 + width]
        
        # Apply rotation if the orientation is "hz" (horizontal)
        if orientations[i] == "hz":
            arena_image = cv2.rotate(arena_image, cv2.ROTATE_90_CLOCKWISE)
        
        # Split the arena into left and right halves
        arena_height, arena_width = arena_image.shape
        half_width = arena_width // 2
        
        # Ensure half_width is even
        if half_width % 2 != 0:
            half_width -= 1
            
        left_half = arena_image[:, :half_width]
        right_half = arena_image[:, arena_width - half_width:]
        
        # Rotate the right half 180 degrees
        right_half_rotated = cv2.rotate(right_half, cv2.ROTATE_180)
        
        # Show left half
        axs[1, i].imshow(left_half, cmap="gray", vmin=0, vmax=255)
        axs[1, i].set_title(f"Left {i+1}")
        axs[1, i].axis("off")
        
        # Show right half (rotated)
        axs[2, i].imshow(right_half_rotated, cmap="gray", vmin=0, vmax=255)
        axs[2, i].set_title(f"Right {i+1}")
        axs[2, i].axis("off")

    plt.tight_layout()
    plt.savefig(
        str(test_processedfolder / "test_crop_check.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()
    
    print(f"Arena detection visualization saved to: {test_processedfolder / 'test_crop_check.png'}")

    # Get a list of first few image files
    images = [f.name for f in folder.glob("*.[jJ][pP][gG]") if f.is_file()]
    images.sort(key=lambda f: int(re.sub(r"\D", "", f)))
    
    # Limit to the specified number of images
    test_images = images[:num_images]
    
    print(f"Processing {len(test_images)} test images: {test_images}")

    # Create the subfolders for each arena's left and right tracks
    for j in range(len(regions_of_interest)):
        left_subfolder = test_processedfolder / f"arena{j+1}" / "Left"
        right_subfolder = test_processedfolder / f"arena{j+1}" / "Right"
        left_subfolder.mkdir(parents=True, exist_ok=True)
        right_subfolder.mkdir(parents=True, exist_ok=True)

    # Process the test images
    for image in tqdm(test_images, desc="Processing test images"):
        process_image(image, regions_of_interest, folder, test_processedfolder, orientations)

    print(f"Test processing complete!")
    print(f"Results saved in: {test_processedfolder}")
    print(f"Check the 'test_crop_check.png' file to verify arena detection and splitting.")
    print(f"Check individual arena folders to verify the cropped images look correct.")
    
    # Ask if user wants to clean up test folder
    if os.isatty(sys.stdin.fileno()):
        cleanup = input("Do you want to delete the test processing folder? (y/n): ")
        if cleanup.lower() == "y":
            import shutil
            shutil.rmtree(test_processedfolder)
            print("Test folder cleaned up.")


def find_recorded_folders(data_folder):
    """Find all folders that end with '_Recorded' for testing."""
    data_folder = Path(data_folder)
    recorded_folders = []
    
    for folder in data_folder.iterdir():
        if folder.is_dir() and folder.name.endswith("_Recorded"):
            recorded_folders.append(folder)
    
    return recorded_folders


def check_process(data_folder):
    """Check which folders need processing and process them."""
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


def get_orientation_from_metadata(folder, arena_index):
    """Read orientation information from metadata.json file."""
    metadata_file = folder / "metadata.json"
    
    if not metadata_file.exists():
        print(f"Warning: metadata.json not found in {folder}. Using default orientation 'std'.")
        return "std"
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Arena names are typically Arena1, Arena2, etc.
        arena_name = f"Arena{arena_index + 1}"
        
        if arena_name in metadata and len(metadata[arena_name]) > 4:
            orientation = metadata[arena_name][4]
            return orientation
        else:
            print(f"Warning: Orientation not found for {arena_name} in metadata. Using default 'std'.")
            return "std"
            
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Warning: Error reading metadata file: {e}. Using default orientation 'std'.")
        return "std"


def process_image(image, regions_of_interest, folder, processedfolder, orientations):
    """Process a single image: crop arenas and split into left/right halves."""
    try:
        # Read and process the image
        frame = cv2.imread(str(folder / image))
        if frame is None:
            print(f"Warning: Could not load image {image}")
            return
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert the folder name to lowercase
        folder_name = str(folder).lower()

        # Apply global rotation based on folder name
        if "_flip" in folder_name:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif "rotatel" in folder_name:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif "rotater" in folder_name:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Process each arena
        for j, region in enumerate(regions_of_interest):
            # Get the orientation for this arena
            orientation = orientations[j]
            
            # Crop the arena
            x1, y1, x2, y2 = region
            width = x2 - x1
            height = y2 - y1

            # Adjust the width and height to be multiples of 2
            if width % 2 != 0:
                width -= 1
            if height % 2 != 0:
                height -= 1

            arena_image = frame[y1 : y1 + height, x1 : x1 + width]
            
            # Apply rotation if the orientation is "hz" (horizontal)
            if orientation == "hz":
                arena_image = cv2.rotate(arena_image, cv2.ROTATE_90_CLOCKWISE)
            
            # Split the arena into left and right halves
            arena_height, arena_width = arena_image.shape
            half_width = arena_width // 2
            
            # Ensure half_width is even
            if half_width % 2 != 0:
                half_width -= 1
                
            left_half = arena_image[:, :half_width]
            right_half = arena_image[:, arena_width - half_width:]
            
            # Rotate the right half 180 degrees (equivalent to transpose=2,transpose=2 in ffmpeg)
            right_half_rotated = cv2.rotate(right_half, cv2.ROTATE_180)
            
            # Save the cropped and split images
            image_stem = Path(image).stem
            
            # Left half
            left_subfolder = processedfolder / f"arena{j+1}" / "Left"
            left_image_file = f"{image_stem}_cropped.jpg"
            cv2.imwrite(str(left_subfolder / left_image_file), left_half)
            
            # Right half (rotated)
            right_subfolder = processedfolder / f"arena{j+1}" / "Right"
            right_image_file = f"{image_stem}_cropped.jpg"
            cv2.imwrite(str(right_subfolder / right_image_file), right_half_rotated)
    
    except Exception as e:
        print(f"Error processing {image}: {e}")
    finally:
        # Explicitly clean up large variables
        if 'frame' in locals():
            del frame
        if 'arena_image' in locals():
            del arena_image


def process_image_batch(image_batch, regions_of_interest, folder, processedfolder, orientations):
    """Process a batch of images to reduce overhead."""
    for image in image_batch:
        process_image(image, regions_of_interest, folder, processedfolder, orientations)


def process_folder(in_folder):
    """Process a folder of images, detecting arenas and splitting them into left/right tracks."""
    inputfolder = in_folder
    folder = inputfolder

    processedfolder = inputfolder.with_name(
        inputfolder.stem.replace("_Recorded", "_Processing")
    )

    # Create the subfolder if it doesn't exist
    processedfolder.mkdir(exist_ok=True)

    # Load the first frame for arena detection
    frame = cv2.imread(inputfolder.joinpath("image0.jpg").as_posix())

    # If it's not already, make it grayscale
    if len(frame.shape) > 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert the folder name to lowercase
    folder_name = str(folder).lower()

    # Apply global rotation based on folder name
    if "_flip" in folder_name:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif "rotatel" in folder_name:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif "rotater" in folder_name:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Equalize the histogram to make thresholding easier
    frame = cv2.equalizeHist(frame)

    # Use shared arena coordinates
    regions_of_interest = REGIONS_OF_INTEREST

    # Get orientation information for each arena
    orientations = []
    for i in range(len(regions_of_interest)):
        orientation = get_orientation_from_metadata(inputfolder, i)
        orientations.append(orientation)
        print(f"Arena {i+1} orientation: {orientation}")

    # Create visualization of detected arenas with split preview (3 rows x 9 columns)
    fig, axs = plt.subplots(3, 9, figsize=(30, 10))  # 3 rows, 9 columns (arena + left + right for each)
    
    for i in range(9):
        # Show original arena
        arena_crop = frame[
            regions_of_interest[i][1] : regions_of_interest[i][3],
            regions_of_interest[i][0] : regions_of_interest[i][2],
        ]
        axs[0, i].imshow(arena_crop, cmap="gray", vmin=0, vmax=255)
        axs[0, i].set_title(f"Arena {i+1} ({orientations[i]})")
        axs[0, i].axis("off")
        
        # Process the arena to show left and right splits
        x1, y1, x2, y2 = regions_of_interest[i]
        width = x2 - x1
        height = y2 - y1

        # Adjust the width and height to be multiples of 2
        if width % 2 != 0:
            width -= 1
        if height % 2 != 0:
            height -= 1

        arena_image = frame[y1 : y1 + height, x1 : x1 + width]
        
        # Apply rotation if the orientation is "hz" (horizontal)
        if orientations[i] == "hz":
            arena_image = cv2.rotate(arena_image, cv2.ROTATE_90_CLOCKWISE)
        
        # Split the arena into left and right halves
        arena_height, arena_width = arena_image.shape
        half_width = arena_width // 2
        
        # Ensure half_width is even
        if half_width % 2 != 0:
            half_width -= 1
            
        left_half = arena_image[:, :half_width]
        right_half = arena_image[:, arena_width - half_width:]
        
        # Rotate the right half 180 degrees
        right_half_rotated = cv2.rotate(right_half, cv2.ROTATE_180)
        
        # Show left half
        axs[1, i].imshow(left_half, cmap="gray", vmin=0, vmax=255)
        axs[1, i].set_title(f"Left {i+1}")
        axs[1, i].axis("off")
        
        # Show right half (rotated)
        axs[2, i].imshow(right_half_rotated, cmap="gray", vmin=0, vmax=255)
        axs[2, i].set_title(f"Right {i+1}")
        axs[2, i].axis("off")

    plt.tight_layout()
    plt.savefig(
        str(processedfolder.joinpath("crop_check.png")), dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Get a list of all image files in the input folder
    images = [f.name for f in folder.glob("*.[jJ][pP][gG]") if f.is_file()]

    # Sort the list of images by their number
    images.sort(key=lambda f: int(re.sub(r"\D", "", f)))

    # Create the subfolders for each arena's left and right tracks
    for j in range(len(regions_of_interest)):
        left_subfolder = processedfolder / f"arena{j+1}" / "Left"
        right_subfolder = processedfolder / f"arena{j+1}" / "Right"
        left_subfolder.mkdir(parents=True, exist_ok=True)
        right_subfolder.mkdir(parents=True, exist_ok=True)

    # Check if standard input is connected to a terminal
    is_tty = os.isatty(sys.stdin.fileno())

    print(f"Processing {len(images)} images...")

    # Process images in batches to reduce memory usage and overhead
    batch_size = max(1, len(images) // (mp.cpu_count() * 4))  # Adaptive batch size
    image_batches = [images[i:i + batch_size] for i in range(0, len(images), batch_size)]
    
    # Prepare the arguments for the process_image_batch function
    args = [(batch, regions_of_interest, folder, processedfolder, orientations) for batch in image_batches]

    # Use threading backend for better resource management
    with Parallel(n_jobs=-1, backend='threading', batch_size=1) as parallel:
        results = parallel(delayed(process_image_batch)(*arg) for arg in tqdm(args, disable=not is_tty))
    
    # Force garbage collection to clean up any remaining resources
    gc.collect()

    # Rename the processed folder from _Processing to _Cropped
    croppedfolder = processedfolder.with_name(
        processedfolder.stem.replace("_Processing", "_Cropped")
    )
    processedfolder.rename(croppedfolder)
    print(
        f"Processing of {in_folder.name} finished! Folder renamed to {croppedfolder.name}"
    )


# Main execution
if __name__ == "__main__":
    # Set multiprocessing start method to avoid resource tracker issues
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Start method already set
    
    parser = argparse.ArgumentParser(description="Process arena images and split into left/right tracks")
    parser.add_argument("--test", "-t", action="store_true", help="Run in test mode")
    parser.add_argument("--folder", "-f", type=str, help="Specific folder to test (for test mode)")
    parser.add_argument("--num-images", "-n", type=int, default=5, help="Number of images to process in test mode (default: 5)")
    
    args = parser.parse_args()
    
    try:
        if args.test:
            if args.folder:
                # Test specific folder
                test_process_folder(args.folder, args.num_images)
            else:
                # Show available folders and let user choose
                recorded_folders = find_recorded_folders(datafolder)
                if not recorded_folders:
                    print("No folders ending with '_Recorded' found in the data folder.")
                    sys.exit(1)
                
                print("Available folders for testing:")
                for i, folder in enumerate(recorded_folders):
                    print(f"{i+1}: {folder.name}")
                
                if os.isatty(sys.stdin.fileno()):
                    choice = input(f"Choose a folder to test (1-{len(recorded_folders)}): ")
                    try:
                        folder_index = int(choice) - 1
                        if 0 <= folder_index < len(recorded_folders):
                            test_process_folder(recorded_folders[folder_index], args.num_images)
                        else:
                            print("Invalid choice.")
                    except ValueError:
                        print("Please enter a valid number.")
                else:
                    # Non-interactive mode, test the first folder
                    print(f"Non-interactive mode: testing first folder {recorded_folders[0].name}")
                    test_process_folder(recorded_folders[0], args.num_images)
        else:
            # Normal processing mode
            check_process(datafolder)

            # Optional: Run integrity check
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
