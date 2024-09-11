#!/bin/bash

# Enable debug mode if needed
debug=0
# Enable echo mode if needed
echo_mode=0

# Check for echo argument
if [[ "$1" == "echo" ]]; then
    echo_mode=1
    shift
fi

# Check if at least one directory is provided
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 [echo] <directory1> <directory2> ... <directoryN>"
    exit 1
fi

# Loop through each provided directory
for input_dir in "$@"; do
    # Extract the base name of the input directory
    base_dir_name=$(basename "$input_dir")

    # Define output directory based on the base name of the input directory
    output_dir="/mnt/upramdya_data/MD/F1_Tracks/Videos/$base_dir_name"

    # Create output directory if it doesn't exist
    mkdir -p "$output_dir"

    # Path to the metadata file
    metadata_file="$input_dir/metadata.json"

    # Check if the metadata file exists
    if [[ ! -f "$metadata_file" ]]; then
        echo "Metadata file not found in $input_dir. Skipping."
        continue
    fi

    # Find all mp4 files in the input directory and its subdirectories
    mapfile -t files < <(find "$input_dir" -type f -name "*.mp4")

    # Loop through each file
    for input_file in "${files[@]}"; do
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

        # Extract the arena name from the file path and capitalize it
        arena_name=$(basename "$(dirname "$input_file")")
        capitalized_arena_name=$(echo "$arena_name" | awk '{print toupper(substr($0,1,1)) tolower(substr($0,2))}')

        # Debug output for arena name
        echo "Processing arena: $capitalized_arena_name"

        # Read the orientation for the specific arena from the metadata file
        orientation=$(jq -r --arg arena "$capitalized_arena_name" '.[$arena][4]' "$metadata_file")

        # Check if the orientation was found
        if [[ -z "$orientation" || "$orientation" == "null" ]]; then
            echo "Orientation not found for arena $arena_name in $metadata_file. Skipping."
            continue
        fi

        # Debug output for orientation
        echo "Orientation for $capitalized_arena_name: $orientation"

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

        # Print details if echo_mode is enabled
        if [[ $echo_mode -eq 1 ]]; then
            echo "Found video: $input_file"
            echo "Will process with orientation: $orientation"
            echo "Left output file: $left_output_file"
            echo "Right output file: $right_output_file"
            continue
        fi

        # Process the video
        echo "Processing $input_file with orientation $orientation..."
        if [[ $debug -eq 1 ]]; then
            ffmpeg -loglevel debug -i "$input_file" -filter_complex "$filter_complex" \
                -map "[left]" "$left_output_file" -map "[right]" "$right_output_file"
        else
            ffmpeg -loglevel error -i "$input_file" -filter_complex "$filter_complex" \
                -map "[left]" "$left_output_file" -map "[right]" "$right_output_file"
        fi

        # Check the exit status of ffmpeg
        if [[ $? -eq 0 ]]; then
            echo "Processed $input_file"
        else
            echo "Error processing $input_file. Continuing with next file."
        fi
    done
done