#!/bin/bash

VIDEOS_FOLDER="/Volumes/Ramdya-Lab/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Videos/230606_DarkishTest_Cropped_Videos"
PROCESSED_DATA_FOLDER="/Volumes/Ramdya-Lab/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Sleap/Datasets/230606_DarkishTest_Full"
OUTPUT_FOLDER="/Volumes/Ramdya-Lab/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Processed/230606_DarkishTest_Full"

# Create output folder if it doesn't exist
mkdir -p "$OUTPUT_FOLDER"

# Loop over all video files
for video in "$VIDEOS_FOLDER"/*; do
  # Get the video file name without the extension
  video_name=$(basename "$video" | cut -f 1 -d '.')

  # Print the current video name
  echo "Processing video: $video_name"

  # Create a subfolder in the output folder for this video
  video_output_folder="$OUTPUT_FOLDER/$video_name"
  mkdir -p "$video_output_folder"

  # Copy the video file to the output subfolder
  cp "$video" "$video_output_folder"

  # Find all processed data files that contain the video name
  processed_data_files=$(find "$PROCESSED_DATA_FOLDER" -name "*$video_name*")

  # Check if any processed data files were found
  if [ -z "$processed_data_files" ]; then
    echo "Warning: No processed data files found for video: $video_name"
  else
    # Copy all processed data files to the output subfolder
    for processed_data_file in $processed_data_files; do
      cp "$processed_data_file" "$video_output_folder"
    done
  fi
done

echo "Done!"