from pathlib import Path
import cv2
import numpy as np
import re
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil
import sys
import os

# Configuration parameters
REGION_COORDINATES = [
    (90, 30, 620, 600),  # Region 1
    (1590, 30, 2110, 600),  # Region 2
    (3080, 30, 3600, 600),  # Region 3
    (90, 1170, 620, 1750),  # Region 4
    (1590, 1170, 2110, 1750),  # Region 5
    (3080, 1170, 3600, 1750),  # Region 6
    (90, 2370, 620, 2900),  # Region 7
    (1590, 2370, 2110, 2900),  # Region 8
    (3080, 2370, 3600, 2900),  # Region 9
]

ADAPTIVE_THRESH_PARAMS = {
    "block_size": 61,
    "c": 2,
    "method": cv2.ADAPTIVE_THRESH_MEAN_C,
}

CONTOUR_PARAMS = {"min_area": 15000, "max_area": 35000, "padding": 10}


def process_last_frame(image_path, rotation="rotater"):
    """Process the first frame to find rectangles"""

    # Find number of frames
    cap = cv2.VideoCapture(str(image_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read last frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
    ret, img = cap.read()

    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if rotation:
        print(rotation)
        if rotation == "flip":
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif rotation == "rotatel":
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation == "rotater":
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    rectangles_per_region = []
    for region_idx, region in enumerate(REGION_COORDINATES):
        x1, y1, x2, y2 = region
        region_img = img[y1:y2, x1:x2]

        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            region_img,
            255,
            ADAPTIVE_THRESH_PARAMS["method"],
            cv2.THRESH_BINARY,
            ADAPTIVE_THRESH_PARAMS["block_size"],
            ADAPTIVE_THRESH_PARAMS["c"],
        )

        # Contour detection
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rectangles = [
            cv2.boundingRect(c)
            for c in contours
            if CONTOUR_PARAMS["min_area"]
            < cv2.contourArea(c)
            < CONTOUR_PARAMS["max_area"]
        ]

        # Validate we found exactly 6 corridors per region
        if len(rectangles) != 6:
            print(f"Warning: Found {len(rectangles)} contours in region {region_idx+1}")
            rectangles_per_region.append([])
        else:
            rectangles_per_region.append(rectangles)

    return rectangles_per_region


def process_image(image_path, output_base, rectangles_per_region, rotation=None):
    """Process individual image using precomputed rectangles"""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if rotation:
        print(rotation)

    for region_idx, region in enumerate(REGION_COORDINATES):
        x1, y1, x2, y2 = region
        region_img = img[y1:y2, x1:x2]

        # Rotate the cropped region if needed
        if rotation:
            if rotation == "flip":
                region_img = cv2.rotate(region_img, cv2.ROTATE_180)
            elif rotation == "rotatel":
                region_img = cv2.rotate(region_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif rotation == "rotater":
                region_img = cv2.rotate(region_img, cv2.ROTATE_90_CLOCKWISE)

        rectangles = rectangles_per_region[region_idx]
        if not rectangles:
            continue

        # Process and save subcrops
        for corridor_idx, (x, y, w, h) in enumerate(rectangles):
            pad = CONTOUR_PARAMS["padding"]
            subcrop = region_img[
                max(0, y - pad) : min(region_img.shape[0], y + h + pad),
                max(0, x - pad) : min(region_img.shape[1], x + w + pad),
            ]

            # Ensure even dimensions
            subcrop = subcrop[: subcrop.shape[0] // 2 * 2, : subcrop.shape[1] // 2 * 2]

            output_dir = (
                output_base / f"arena{region_idx+1}" / f"corridor{corridor_idx+1}"
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save the image with "_cropped" suffix
            output_filename = image_path.stem + "_cropped.jpg"
            cv2.imwrite(str(output_dir / output_filename), subcrop)


def process_folder(input_folder, output_folder):
    """Process all images in a folder"""
    images = sorted(
        [f for f in input_folder.glob("*.[jJ][pP][gG]") if f.is_file()],
        key=lambda x: int(re.findall(r"\d+", x.stem)[0]),
    )

    # Determine rotation from folder name
    folder_name = input_folder.name.lower()
    rotation = None
    if "_flip" in folder_name:
        rotation = "flip"
    elif "rotatel" in folder_name:
        rotation = "rotatel"
    elif "rotater" in folder_name:
        rotation = "rotater"

    # Process the first frame to find rectangles
    rectangles_per_region = process_last_frame(images[0], rotation)

    # Parallel processing with progress bar
    Parallel(n_jobs=-1)(
        delayed(process_image)(img, output_folder, rectangles_per_region, rotation)
        for img in tqdm(images, desc=f"Processing {input_folder.name}")
    )

    # Generate verification preview
    generate_verification_preview(input_folder, output_folder)


def generate_verification_preview(input_folder, output_folder):
    """Generate grid preview of cropping results"""
    fig, axs = plt.subplots(9, 6, figsize=(20, 20))

    sample_img = next(output_folder.glob("**/*.jpg"))
    sample = cv2.imread(str(sample_img), cv2.IMREAD_GRAYSCALE)

    for region_idx in range(9):
        for corridor_idx in range(6):
            try:
                img_path = next(
                    (
                        output_folder
                        / f"arena{region_idx+1}"
                        / f"corridor{corridor_idx+1}"
                    ).glob("*.jpg")
                )
                axs[region_idx, corridor_idx].imshow(
                    cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE), cmap="gray"
                )
            except (StopIteration, IndexError):
                pass
            axs[region_idx, corridor_idx].axis("off")

    plt.savefig(output_folder / "crop_verification.png", dpi=300, bbox_inches="tight")
    plt.close()


def main_processing_flow(data_folder):
    """Main processing workflow"""
    data_folder = Path(data_folder)

    for folder in data_folder.iterdir():
        if folder.is_dir() and "_Recorded" in folder.name:
            output_folder = folder.with_name(
                folder.name.replace("_Recorded", "_Cropped")
            )

            if output_folder.exists():
                print(f"Skipping {folder.name} - already processed")
                continue

            print(f"Processing {folder.name}")
            output_folder.mkdir(exist_ok=True)

            try:
                process_folder(folder, output_folder)
                print(f"Completed {folder.name}")
            except Exception as e:
                print(f"Failed processing {folder.name}: {str(e)}")
                shutil.rmtree(output_folder)


if __name__ == "__main__":
    data_folder = Path("/home/matthias/Videos/")
    main_processing_flow(data_folder)
