#!/bin/bash

# Input and output directories
input_dir="input_folder"
output_dir="output_folder"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Find all mp4 files in the input directory and its subdirectories
find "$input_dir" -type f -name "*.mp4" | while read -r input_file; do
    # Get the base name of the file (without directory and extension)
    base_name=$(basename "$input_file" .mp4)

    # Get the relative path of the input file from the input directory
    relative_path=$(realpath --relative-to="$input_dir" "$input_file")

    # Get the directory of the input file relative to the input directory
    relative_dir=$(dirname "$relative_path")

    # Define output directories for left and right halves
    left_output_dir="$output_dir/$relative_dir/Left"
    right_output_dir="$output_dir/$relative_dir/Right"

    # Create output directories if they don't exist
    mkdir -p "$left_output_dir"
    mkdir -p "$right_output_dir"

    # Define output file paths
    left_output_file="$left_output_dir/${base_name}_left.mp4"
    right_output_file="$right_output_dir/${base_name}_right_rotated.mp4"

    # Check if the output files already exist
    if [[ -f "$left_output_file" && -f "$right_output_file" ]]; then
        echo "Skipping $input_file, already processed."
        continue
    fi

    # Read the orientation from metadata.json
    metadata_file=$(dirname "$input_file")/metadata.json
    if [[ -f "$metadata_file" ]]; then
        # Extract the arena name from the file path
        arena_name=$(basename "$(dirname "$input_file")")
        # Read the orientation for the specific arena
        orientation=$(jq -r --arg arena "$arena_name" '.[$arena][4]' "$metadata_file")
    else
        echo "Metadata file not found for $input_file. Skipping."
        continue
    fi

    # Determine the filter based on orientation
    if [[ "$orientation" == "hz" ]]; then
        filter_complex="[0:v]transpose=1,split=2[left][right]; \
                        [left]crop=iw/2:ih:0:0[left]; \
                        [right]crop=iw/2:ih:iw/2:0,transpose=2,transpose=2[right]"
    elif [[ "$orientation" == "std" ]]; then
        filter_complex="[0:v]crop=iw/2:ih:0:0[left]; \
                        [0:v]crop=iw/2:ih:iw/2:0,transpose=2,transpose=2[right]"
    else
        echo "Unknown orientation for $input_file. Skipping."
        continue
    fi

    # Process the video
    echo "Processing $input_file with orientation $orientation..."
    ffmpeg -loglevel error -i "$input_file" -filter_complex "$filter_complex" \
        -map "[left]" "$left_output_file" -map "[right]" "$right_output_file" | pv -p -t -e -N "Processing $base_name"

    echo "Processed $input_file"
done
