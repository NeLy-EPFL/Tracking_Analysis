#!/bin/bash

# Define the base directory
base_dir="/mnt/labserver/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Videos"

# Find all directories with "Flipped" in their name
find "$base_dir" -type d -name "*Flip*" | while read -r dir; do
    # Remove all .slp and .h5 files
    find "$dir" -type f \( -name "*.slp" -o -name "*.h5" \) -exec rm {} \;

    # Rotate all .mp4 files 180°
    find "$dir" -type f -name "*.mp4" | while read -r file; do
        temp_file="${file%.mp4}_temp.mp4"
        ffmpeg -i "$file" -vf "rotate=PI" "$temp_file"
        mv "$temp_file" "$file"
    done

    # Rename the directory
    new_dir="${dir/Tracked/Checked}"
    mv "$dir" "$new_dir"
done
