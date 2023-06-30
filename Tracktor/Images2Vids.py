import os
from pathlib import Path
from tqdm import tqdm

folder_path = Path("/home/matthias/Videos/230606_LightReco_Cropped/")
output_folder = folder_path.parent / f"{folder_path.stem}_Videos"

output_folder.mkdir(exist_ok=True)

def create_video_from_images(images_folder, output_folder, video_name, fps):
    f = images_folder.as_posix()
    video_path = f"{output_folder.as_posix()}/{video_name}.mp4"
    if not os.path.exists(video_path):
        terminal_call = f"ffmpeg -loglevel panic -nostats -hwaccel cuda -r {fps} -i {f}/image%d_cropped.jpg -pix_fmt yuv420p -c:v libx265 -x265-params log-level=error -vsync 0 -crf 15 {video_path}"
        os.system(terminal_call)
        return True
    else:
        return False

def search_folder_for_images(folder_path, output_folder, fps):
    subdirs = []
    for subdir, dirs, files in os.walk(folder_path):
        if any(file.endswith('.jpg') for file in files):
            subdirs.append(subdir)
    with tqdm(total=len(subdirs), desc="Processing videos") as pbar:
        for subdir in subdirs:
            video_name = '_'.join(Path(subdir).parts[-2:])
            create_video_from_images(Path(subdir), output_folder, video_name, fps)
            pbar.update(1)

fps = "30"
search_folder_for_images(folder_path, output_folder, fps)