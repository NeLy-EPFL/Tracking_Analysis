from pathlib import Path
from PIL import Image
import shutil

def check_integrity(folder, source_folder):
    folder = Path(folder)
    source_folder = Path(source_folder)
    folder_items = list(folder.iterdir())
    # Check for exactly one image file in folder
    image_count = len(list(folder.glob("*.png"))) + len(list(folder.glob("*.jpg")))
    if image_count != 1:
        print(f"Cropped check image not found!")
        remove = input("Do you want to remove the processed folder? (y/n): ")
        if remove.lower() == "y":
            shutil.rmtree(folder)
            print(f"Folder {folder.name} has been removed.")
        return False
    else:
        print(f"Cropped check image found...")
        # Display the image and ask the user if it's valid
        img = Image.open(folder / "crop_check.png")
        img.show()
        valid = input("Are the detected ROIs valid? (y/n): ")
        img.close()
        if valid.lower() == "n":
            remove = input("Do you want to remove the processed folder? (y/n): ")
            if remove.lower() == "y":
                shutil.rmtree(folder)
                print(f"Folder {folder.name} has been removed.")
            return False
    # Check for exactly 9 subfolders in folder
    subfolder_count = sum(1 for item in folder_items if item.is_dir())
    if subfolder_count != 9:
        print(f"Number of subfolders in folder: {subfolder_count}")
        remove = input("Do you want to remove the processed folder? (y/n): ")
        if remove.lower() == "y":
            shutil.rmtree(folder)
            print(f"Folder {folder.name} has been removed.")
        return False
    else:
        print(f"all arenas found...")
    source_image_count = len(list(source_folder.glob("*.png"))) + len(
        list(source_folder.glob("*.jpg"))
    )
    for subfolder in folder.iterdir():
        if not subfolder.is_dir():
            continue
        if len(list(subfolder.iterdir())) != 6:
            print(
                f"Number of subsubfolders in subfolder: {len(list(subfolder.iterdir()))}"
            )
            remove = input("Do you want to remove the processed folder? (y/n): ")
            if remove.lower() == "y":
                shutil.rmtree(folder)
                print(f"Folder {folder.name} has been removed.")
            return False
        else:
            print(f"all corridors found in {subfolder.stem}...")
        for subsubfolder in subfolder.iterdir():
            if not subsubfolder.is_dir():
                return False
            image_count = len(list(subsubfolder.glob("*.png"))) + len(
                list(subsubfolder.glob("*.jpg"))
            )
            if image_count != source_image_count:
                print(
                    f"Image count for folder: {subsubfolder.name} is {image_count} instead of {source_image_count}"
                )
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
        source_folder_name = folder.stem.replace("_Cropped", "")
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
