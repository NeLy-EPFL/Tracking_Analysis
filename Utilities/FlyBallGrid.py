import math
import subprocess
from pathlib import Path

# Set the input folder path
input_folder = Path("/mnt/labserver/DURRIEU_Matthias/Videos/GridClips/")

# Get all video files from the input folder
input_files = list(input_folder.glob("*clip*.mp4"))

# Sort the input files by the number in their name
input_files.sort(key=lambda f: int(f.stem.split("_")[1]))

input_files = input_files#[:5]

print(f"input_files: {input_files}")

# Calculate the number of columns and rows for the grid layout
num_videos = len(input_files)
num_cols = math.ceil(math.sqrt(num_videos))
num_rows = math.ceil(num_videos / num_cols)

print(f"num_videos: {num_videos}")
print(f"num_cols: {num_cols}")
print(f"num_rows: {num_rows}")

# Get the width and height of the first video
result = subprocess.run([
    "ffprobe", "-v", "error", "-select_streams", "v:0",
    "-show_entries", "stream=width,height",
    "-of", "csv=p=0", str(input_files[0])
], capture_output=True, text=True)
if result.returncode != 0:
    print(f"Error running ffprobe on {input_files[0]}: {result.stderr}")
else:
    width, height = map(int, result.stdout.strip().split(","))

# Create blank videos if necessary
while num_videos < num_cols * num_rows:
    blank_video = input_folder / f"blank{num_videos}.mp4"
    subprocess.run([
        "ffmpeg", "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:d=1:r=25",
        "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
        str(blank_video)
    ])
    input_files.append(blank_video)
    num_videos += 1

# Set the scale factor
scale_factor = 0.2

# Create the filter_complex argument for the ffmpeg command
filter_complex = "".join(f"[{i}:v]scale=iw*{scale_factor}:ih*{scale_factor}[s{i}];[s{i}]copy[v{i}];" for i in range(num_cols * num_rows))
for row in range(num_rows):
    row_inputs = "".join(f"[v{row * num_cols + col}]" for col in range(num_cols))
    filter_complex += f"{row_inputs}hstack=inputs={num_cols}[h{row}];"
vstack_inputs = "".join(f"[h{row}]" for row in range(num_rows))
filter_complex += f"{vstack_inputs}vstack=inputs={num_rows}[v]"

print(f"filter_complex : {filter_complex}")

# Create the ffmpeg command arguments
ffmpeg_args = ["ffmpeg"]
for input_file in input_files:
    ffmpeg_args.extend(["-i", str(input_file)])
ffmpeg_args.extend([
    "-filter_complex", filter_complex,
    "-map", "[v]",
    "/mnt/labserver/DURRIEU_Matthias/Videos/GridClips/grid_video.mp4"
])

# Run the ffmpeg command
subprocess.run(ffmpeg_args)
