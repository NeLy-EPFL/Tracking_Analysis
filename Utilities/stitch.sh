#!/bin/bash

# Set the paths to the two folders containing the videos
folder1="/home/matthias/Videos/Test2_Cropped_Videos/"
folder2="/home/matthias/Videos/Test_Cropped_Videos/"

# Set the output folder
output_folder="/home/matthias/Videos/Test1_2_Merged/"

# Loop through all the video files in folder1
for file1 in "$folder1"/*.mp4; do
  # Extract the filename without the extension
  filename=$(basename -- "$file1")
  filename="${filename%.*}"

  # Set the path to the corresponding video file in folder2
  file2="$folder2/$filename.mp4"

  # Check if the corresponding video file exists in folder2
  if [ -f "$file2" ]; then
    # Set the path to the output file
    output_file="$output_folder/$filename-stitched.mp4"

    # Create a temporary file with the list of files to concatenate
    echo "file '$file1'" > concat_list.txt
    echo "file '$file2'" >> concat_list.txt

    # Stitch the two video files and save the result to the output folder
    ffmpeg -f concat -safe 0 -i concat_list.txt -c copy "$output_file"

    # Remove the temporary file
    rm concat_list.txt
  fi
done