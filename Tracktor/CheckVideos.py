import subprocess
from pathlib import Path
import shutil


def check_video_integrity(video_path):
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_format", "-show_streams", video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error checking video integrity: {e.stderr.decode('utf-8')}")
        return False


def check_folder_integrity(folder):
    folder = Path(folder)
    folder_items = list(folder.iterdir())
    # Check for exactly 9 subfolders in folder
    subfolder_count = sum(1 for item in folder_items if item.is_dir())
    if subfolder_count != 9:
        print(f"Number of subfolders in folder: {subfolder_count}")
        return False
    else:
        print(f"All arenas found...")
    for subfolder in folder.iterdir():
        if not subfolder.is_dir():
            continue
        if len(list(subfolder.iterdir())) != 6:
            print(
                f"Number of subsubfolders in subfolder: {len(list(subfolder.iterdir()))}"
            )
            return False
        else:
            print(f"All corridors found in {subfolder.stem}...")
        for subsubfolder in subfolder.iterdir():
            if not subsubfolder.is_dir():
                return False
            video_count = len(list(subsubfolder.glob("*.mp4")))
            if video_count != 1:
                print(
                    f"Video count for folder: {subsubfolder.name} is {video_count} instead of 1"
                )
                return False
            else:
                video_path = list(subsubfolder.glob("*.mp4"))[0]
                if not check_video_integrity(video_path.as_posix()):
                    print(f"Video {video_path.name} is corrupted or otherwise unusable")
                    return False
    return True


def process_data_folder(data_folder, source_data_folder):
    data_folder = Path(data_folder)

    for folder in data_folder.iterdir():
        if not folder.is_dir() or not folder.name.endswith("_Videos"):
            continue
        print(f"Checking integrity of folder: {folder.name}")
        verified = check_folder_integrity(folder)
        if verified:
            new_name = f"{folder}_Checked"
            folder.rename(new_name)
            print(f"Folder {folder.name} is verified.")
            print(f"Folder renamed to: {new_name}")
            
            # Prompt user to remove original image folder
            remove_images = input(f"Do you want to remove the original image folder for {folder.name}? (y/n): ")
            if remove_images.lower() == 'y':
                image_folder_name = folder.name.replace('_Videos', '_Cropped_Checked')
                image_folder = source_data_folder / image_folder_name
                if image_folder.exists() and image_folder.is_dir():
                    print(f"Removing original image folder: {image_folder.as_posix()}")
                    shutil.rmtree(image_folder)
        else:
            print(f"Folder {folder.name} is not verified.")


source_data_folder = Path('/home/matthias/Videos/') 

process_data_folder(
    "/mnt/labserver/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Videos/",
    source_data_folder
)