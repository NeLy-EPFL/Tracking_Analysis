#!/bin/bash

# activate the right conda environment
source activate sleap

# Set input and output paths
input_path="/mnt/labserver/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Processed/230606_LightReco_Cropped_Videos/"
model_path="/mnt/labserver/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Sleap/Labels/models/230602_141343.single_instance.n=108/"

# Find all videos in input folder and its subfolders
videos=$(find $input_path -type f -name "*.mp4")

# For each video, use sleap-track terminal command with existing model to track ball positions
for video in $videos; do
    video_name=$(basename $video .mp4)
    output_folder=$(dirname $video)
    output_file="${output_folder}/${video_name}_tracked_ball.slp"
    sleap-track $video --model $model_path --output $output_file
    
    # execute the command to extract and save .h5 files
    sleap-convert "$output_file" --format analysis
done