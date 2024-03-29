#!/bin/bash

source activate sleap

# Set input and output paths
input_path="/home/matthias/Videos/Test_Cropped_Videos/"
output_path="/home/matthias/Documents/Sleap/Labels/FirstExp/LongRecording/"
model_path="/home/matthias/Documents/Sleap/Labels/models/230602_141343.single_instance.n=108/"

# Find all videos in input folder
videos=$(find $input_path -type f -name "*.mp4")

# For each video, use sleap-track terminal command with existing model to track ball positions
for video in $videos; do
    video_name=$(basename $video .mp4)
    output_file="${output_path}/${video_name}_tracked_ball.slp"
    sleap-track $video --model $model_path --output $output_file
done