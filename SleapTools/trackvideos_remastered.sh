#!/bin/bash

# activate the right conda environment
source activate sleap

# Set input and output paths
input_path="/home/matthias/Videos/230606_LightReco_Cropped_Videos/"
output_path="/mnt/labserver/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Processed/230606_LightReco_Cropped_Videos/"
model_path="/mnt/labserver/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Sleap/Labels/models/230602_141343.single_instance.n=108/"

# Create output folder if it doesn't exist
mkdir -p $output_path

# Find all videos in input folder
videos=$(find $input_path -type f -name "*.mp4")

# For each video, use sleap-track terminal command with existing model to track ball positions
for video in $videos; do
    video_name=$(basename $video .mp4)
    output_folder="${output_path}/${video_name}"
    mkdir -p $output_folder
    output_file="${output_folder}/${video_name}_tracked.slp"
    mv $video $output_folder
    sleap-track $video --model $model_path --output $output_file
    
    # execute the command to extract and save .h5 files
    sleap-convert "$output_file" --format analysis
done