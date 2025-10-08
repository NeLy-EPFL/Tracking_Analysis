import subprocess
from pathlib import Path
import shutil
try:
    # Preferred: package exposes Utils directly
    from utils_behavior import Utils
except Exception:
    try:
        # Fallback: import the module and try to access Utils attribute
        import utils_behavior as utils_behavior_mod
        Utils = getattr(utils_behavior_mod, 'Utils', None)
    except Exception:
        Utils = None
import os

# Known output roots to search for experiment folders (same as Images2Vids)
OUTPUT_PATHS = [
    Path("/mnt/upramdya_data/MD/Infection_Exps/InfectionCorridors/Experiments"),
    Path("/mnt/upramdya_data/MD/F1_Tracks/Videos"),
    Path("/home/matthias/Videos_output"),
]


def check_video_integrity(video_path):
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-i", video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        # Check file size and duration
        file_size = os.path.getsize(video_path)
        if file_size < 1000:  # arbitrary small size threshold
            print(f"Video {video_path} is too small, possible corruption.")
            return False
        
        # Extract video duration using ffprobe
        duration_check = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", 
             "default=noprint_wrappers=1:nokey=1", video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        duration = float(duration_check.stdout.decode("utf-8").strip())
        if duration <= 0:
            print(f"Video {video_path} has zero duration, possible corruption.")
            return False
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error checking video integrity: {e.stderr.decode('utf-8')}")
        return False


def check_folder_integrity(folder):
    folder = Path(folder)

    for subfolder in folder.iterdir():
        if not subfolder.is_dir() or subfolder.name.startswith("."):
            continue
        print(f"Checking subfolder: {subfolder.name}")
        for subsubfolder in subfolder.iterdir():
            if not subsubfolder.is_dir() or subsubfolder.name.startswith("."):
                continue
            print(f"Checking subsubfolder: {subsubfolder.name}")
            video_files = list(subsubfolder.glob("*.mp4"))
            for video_file in video_files:
                if not check_video_integrity(video_file.as_posix()):
                    print(f"Video {video_file.name} is corrupted or otherwise unusable")
                    return False
    return True


def process_data_folder(data_folder, source_data_folder):

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

            image_folder_name = folder.name.replace("_Videos", "_Cropped_Checked")
            image_folder = source_data_folder / image_folder_name
            if image_folder.exists() and image_folder.is_dir():
                print(f"Removing original image folder: {image_folder.as_posix()}")
                shutil.rmtree(image_folder)
        else:
            print(f"Folder {folder.name} is not verified.")
            new_name = f"{folder}_NotChecked"


source_data_folder = Path("/home/matthias/Videos/")
if Utils is None:
    # Diagnostic: show what's importable
    try:
        import importlib
        mod = importlib.import_module('utils_behavior')
        available = dir(mod)
    except Exception as e:
        available = f"Import error: {e}"
    raise RuntimeError(f"Could not locate Utils in utils_behavior. Available attributes: {available}")
else:
    # If Utils is a class or module with get_data_path
    if hasattr(Utils, 'get_data_path'):
        remote_data_folder = Utils.get_data_path()
    elif hasattr(Utils, 'Utils') and hasattr(Utils.Utils, 'get_data_path'):
        remote_data_folder = Utils.Utils.get_data_path()
    else:
        raise RuntimeError("Found Utils but no callable get_data_path attribute")

# Build list of roots to check: include remote_data_folder from Utils if available,
# then the known OUTPUT_PATHS. Deduplicate while preserving order.
roots_to_check = []
if 'remote_data_folder' in globals():
    try:
        rd = Path(remote_data_folder)
        if rd.exists():
            roots_to_check.append(rd)
        else:
            print(f"Utils reported remote data folder {rd} but it does not exist on disk.")
    except Exception:
        pass

for p in OUTPUT_PATHS:
    if p not in roots_to_check:
        roots_to_check.append(p)

# Run integrity check on each existing root
found_any = False
for root in roots_to_check:
    if root.exists() and root.is_dir():
        print(f"Checking experiments under: {root}")
        process_data_folder(root, source_data_folder)
        found_any = True
    else:
        print(f"Skipping missing root: {root}")

if not found_any:
    print("No valid output roots found to check. Please verify OUTPUT_PATHS or Utils.get_data_path()")
