import subprocess
from pathlib import Path
import shutil
import argparse
import numpy as np

try:
    # Preferred: package exposes Utils directly
    from utils_behavior import Utils
except Exception:
    try:
        # Fallback: import the module and try to access Utils attribute
        import utils_behavior as utils_behavior_mod

        Utils = getattr(utils_behavior_mod, "Utils", None)
    except Exception:
        Utils = None
import os

# Known output roots to search for experiment folders (align with Images2Vids)
OUTPUT_PATHS = [
    Path("/mnt/upramdya_data/MD/Infection_Exps/InfectionCorridors/Experiments"),
    Path("/mnt/upramdya_data/MD/F1_Tracks/Videos"),
    Path("/home/matthias/Videos_output"),
    Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Videos"),
]


def check_ffprobe_available():
    try:
        result = subprocess.run(
            ["ffprobe", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def get_video_duration(video_path):
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return None


def validate_video_duration(video_path, expected_duration_sec, tolerance_sec=1.0):
    actual = get_video_duration(video_path)
    if actual is None:
        return False, None
    diff = abs(actual - expected_duration_sec)
    return diff <= tolerance_sec, actual


def check_video_integrity(video_path):
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                "-i",
                video_path,
            ],
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
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
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


def load_duration_data(experiment_folder: Path):
    duration_file = experiment_folder / "duration.npy"
    if not duration_file.exists():
        return None
    try:
        data = np.load(duration_file, allow_pickle=True)
        if isinstance(data, np.ndarray):
            if data.ndim == 0:
                total = float(data.item())
                return {"__default_duration__": total}
            elif len(data.shape) == 1:
                durations = {}
                for i, d in enumerate(data):
                    durations[f"video_{i:03d}"] = float(d)
                return durations
        if isinstance(data, dict):
            return {str(k): float(v) for k, v in data.items()}
    except Exception as e:
        print(f"Warning: could not load duration data from {duration_file}: {e}")
    return None


def check_folder_integrity(
    folder, no_duration_check=False, duration_tolerance=1.0, expected_durations=None
):
    folder = Path(folder)
    video_count = 0

    for subfolder in folder.iterdir():
        if not subfolder.is_dir() or subfolder.name.startswith("."):
            continue
        print(f"Checking subfolder: {subfolder.name}")
        for subsubfolder in subfolder.iterdir():
            if not subsubfolder.is_dir() or subsubfolder.name.startswith("."):
                continue
            print(f"Checking subsubfolder: {subsubfolder.name}")
            video_files = list(subsubfolder.glob("*.mp4"))

            if not video_files:
                print(
                    f"No video files found in {subsubfolder.name} - this folder has no valid videos"
                )
                return False, 0

            for video_file in video_files:
                if not check_video_integrity(video_file.as_posix()):
                    print(f"Video {video_file.name} is corrupted or otherwise unusable")
                    return False, video_count

                # Optional duration validation
                if not no_duration_check and expected_durations:
                    expected = None
                    stem = video_file.stem
                    if stem in expected_durations:
                        expected = expected_durations[stem]
                    elif "__default_duration__" in expected_durations:
                        expected = expected_durations["__default_duration__"]

                    if expected is not None:
                        ok, actual = validate_video_duration(
                            video_file.as_posix(),
                            expected,
                            tolerance_sec=duration_tolerance,
                        )
                        if not ok:
                            print(
                                f"Duration mismatch for {video_file.name}: expected {expected:.2f}s, "
                                f"got {actual if actual is not None else 'N/A'} (tolerance {duration_tolerance:.2f}s)"
                            )
                            return False, video_count

                video_count += 1

    if video_count == 0:
        print("No valid videos found in this experiment folder")
        return False, 0

    return True, video_count


def process_data_folder(
    data_folder,
    source_data_folder,
    no_duration_check=False,
    duration_tolerance=1.0,
    dry_run=False,
):

    for folder in data_folder.iterdir():
        if not folder.is_dir():
            continue
        name = folder.name
        if not (name.endswith("_Videos") or name.endswith("_Videos_NotChecked")):
            continue
        print(f"Checking integrity of folder: {folder.name}")
        # Load durations if available within this experiment folder
        durations = load_duration_data(folder)
        if durations:
            if not no_duration_check:
                print(f"Loaded duration data for {len(durations)} entries")
            else:
                print("Duration validation disabled by flag; ignoring duration.npy")
        verified, video_count = check_folder_integrity(
            folder,
            no_duration_check=no_duration_check,
            duration_tolerance=duration_tolerance,
            expected_durations=durations,
        )

        if verified and video_count > 0:
            # Normalize to *_Videos_Checked
            base = name
            if base.endswith("_Videos_NotChecked"):
                base = base[: -len("_Videos_NotChecked")]
            elif base.endswith("_Videos"):
                base = base[: -len("_Videos")]
            new_name = f"{base}_Videos_Checked"
            target_path = folder.parent / new_name
            if dry_run:
                print(f"DRY RUN: would rename {folder} -> {target_path}")
            else:
                folder.rename(target_path)
            print(f"Folder {name} is verified with {video_count} valid videos.")
            print(f"Folder renamed to: {target_path}")

            # Only remove source images if we have verified videos
            images_name = f"{base}_Cropped_Checked"
            image_folder = source_data_folder / images_name
            if image_folder.exists() and image_folder.is_dir():
                if dry_run:
                    print(
                        f"DRY RUN: would remove original image folder: {image_folder.as_posix()}"
                    )
                else:
                    print(f"Removing original image folder: {image_folder.as_posix()}")
                    shutil.rmtree(image_folder)
            else:
                print(f"Warning: Source image folder not found: {image_folder}")
        else:
            print(
                f"Folder {folder.name} is not verified or has no valid videos (found {video_count} videos)."
            )
            # Ensure name is *_Videos_NotChecked
            base = name
            if base.endswith("_Videos_NotChecked"):
                target_path = folder  # already set
            elif base.endswith("_Videos"):
                base = base[: -len("_Videos")]
                target_path = folder.parent / f"{base}_Videos_NotChecked"
            else:
                # unexpected, but keep _NotChecked suffix
                target_path = folder.parent / f"{base}_NotChecked"
            if target_path != folder:
                if dry_run:
                    print(f"DRY RUN: would rename {folder} -> {target_path}")
                else:
                    folder.rename(target_path)
                    print(f"Folder renamed to: {target_path}")
            print(
                "IMPORTANT: Source images preserved because no valid videos were found!"
            )

            # Do NOT remove source images when videos are invalid or missing
            # Derive base again for message
            images_name = f"{base}_Cropped_Checked"
            image_folder = source_data_folder / images_name
            if image_folder.exists():
                print(f"Source image folder preserved: {image_folder}")
            else:
                print(f"Note: Source image folder not found: {image_folder}")


def build_roots_to_check():
    roots_to_check = []
    if Utils is None:
        # Diagnostic: show what's importable
        try:
            import importlib

            mod = importlib.import_module("utils_behavior")
            available = dir(mod)
        except Exception as e:
            available = f"Import error: {e}"
        raise RuntimeError(
            f"Could not locate Utils in utils_behavior. Available attributes: {available}"
        )
    else:
        # If Utils is a class or module with get_data_path
        if hasattr(Utils, "get_data_path"):
            remote_data_folder = Utils.get_data_path()
        elif hasattr(Utils, "Utils") and hasattr(Utils.Utils, "get_data_path"):
            remote_data_folder = Utils.Utils.get_data_path()
        else:
            raise RuntimeError("Found Utils but no callable get_data_path attribute")

    if "remote_data_folder" in locals():
        try:
            rd = Path(remote_data_folder)
            if rd.exists():
                roots_to_check.append(rd)
            else:
                print(
                    f"Utils reported remote data folder {rd} but it does not exist on disk."
                )
        except Exception:
            pass

    for p in OUTPUT_PATHS:
        if p not in roots_to_check:
            roots_to_check.append(p)
    return roots_to_check


def main():
    parser = argparse.ArgumentParser(
        description="Verify output videos and finalize experiments"
    )
    parser.add_argument(
        "--no-duration-check",
        action="store_true",
        help="Skip duration validation even if duration.npy exists",
    )
    parser.add_argument(
        "--duration-tolerance",
        type=float,
        default=1.0,
        help="Allowed duration mismatch in seconds",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show intended actions without renaming or deleting",
    )
    args = parser.parse_args()

    if not check_ffprobe_available():
        print(
            "ERROR: ffprobe is not available or not working properly. Please install ffmpeg/ffprobe."
        )
        return

    source_data_folder = Path("/home/matthias/Videos/")
    roots_to_check = build_roots_to_check()

    found_any = False
    for root in roots_to_check:
        if root.exists() and root.is_dir():
            print(f"Checking experiments under: {root}")
            process_data_folder(
                root,
                source_data_folder,
                no_duration_check=args.no_duration_check,
                duration_tolerance=args.duration_tolerance,
                dry_run=args.dry_run,
            )
            found_any = True
        else:
            print(f"Skipping missing root: {root}")

    if not found_any:
        print(
            "No valid output roots found to check. Please verify OUTPUT_PATHS or Utils.get_data_path()"
        )


if __name__ == "__main__":
    main()
