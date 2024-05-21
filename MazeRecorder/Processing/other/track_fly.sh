#!/bin/bash

# activate the right conda environment
source activate sleap

# Set input and output paths
datafolder="/mnt/labserver/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Videos/230721_Feedingstate_4_PM_Videos_Tracked/arena5/corridor3/"
model_path="/mnt/labserver/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Sleap/Labels/Thorax_labels.v001.slp.training_job/models/230825_101219.single_instance/"

# Print out the values of datafolder and model_path
echo "datafolder: $datafolder"
echo "model_path: $model_path"

# Find all subdirectories in datafolder
subdirs=$(find $datafolder -type d)

# For each subdirectory, check if .slp and .h5 files already exist
for subdir in $subdirs; do
    slp_file=$(find $subdir -maxdepth 1 -type f -name "*.slp")
    h5_file=$(find $subdir -maxdepth 1 -type f -name "*.h5")
    
    # If .slp and .h5 files do not exist, process videos in this folder
    if [ -z "$slp_file" ] && [ -z "$h5_file" ]; then
        # Find all videos in this folder
        videos=$(find $subdir -maxdepth 1 -type f -name "*.mp4" -print)
        
        # For each video, use sleap-track terminal command with existing model to track ball positions
        for video in $videos; do
            video_name=$(basename $video .mp4)
            output_folder=$(dirname $video)
            output_file="${output_folder}/${video_name}_tracked_fly.slp"
            sleap-track $video --model $model_path --output $output_file
            
            # execute the command to extract and save .h5 files
            sleap-convert "$output_file" --format analysis
        done
    fi
done