from pathlib import Path
from PIL import Image
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys


def check_integrity(folder, source_folder):
    folder = Path(folder)
    source_folder = Path(source_folder)

    # Check for at least one image file in folder
    image_files = list(folder.glob("*.png")) + list(folder.glob("*.jpg"))
    if not image_files:
        print(f"Cropped check image not found!")
        remove = input("Do you want to remove the processed folder? (y/n): ")
        if remove.lower() == "y":
            shutil.rmtree(folder)
            print(f"Folder {folder.name} has been removed.")
        return False

    print(f"Cropped check images found...")
    # Display the images in a grid and ask the user if they're valid
    fig, axs = plt.subplots(len(image_files), figsize=(10, 10))
    for ax, image_file in zip(axs, image_files):
        img = mpimg.imread(str(image_file))
        ax.imshow(img)
    plt.show(block=False)  # display the image using matplotlib
    valid = input("Are the detected ROIs valid? (y/n): ")
    plt.close()  # close the image window

    if valid.lower() == "n":
        remove = input("Do you want to remove the processed folder? (y/n): ")
        if remove.lower() == "y":
            shutil.rmtree(folder)
            print(f"Folder {folder.name} has been removed.")
        return False

    print(f"Folder {folder.name} is verified.")
    return True


def process_data_folder(data_folder):
    data_folder = Path(data_folder)

    for folder in data_folder.iterdir():
        if (
            not folder.is_dir()
            or not folder.name.endswith("_Cropped")
            or folder.name.endswith("_Checked")
        ):
            continue
        source_folder_name = folder.stem.replace("_Cropped", "_Recorded")
        source_folder = data_folder / source_folder_name
        print(f"Checking integrity of folder: {folder.name}")
        verified = check_integrity(folder, source_folder)
        if verified:
            new_name = f"{folder}_Checked"
            folder.rename(new_name)
            print(f"Folder renamed to: {new_name}")
            remove_source = input("Do you want to remove the source folder? (y/n): ")
            if remove_source.lower() == "y":
                shutil.rmtree(source_folder)
                print(f"Source folder {source_folder.name} has been removed.")


process_data_folder("/home/matthias/Videos/")
