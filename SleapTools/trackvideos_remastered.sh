#!/bin/bash

# activate the right conda environment
source activate sleap

# Set input and output paths
input_path="/mnt/labserver/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Videos/230706_FeedingState_3_AM_Videos_Checked/"
model_path="/mnt/labserver/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Sleap/Labels/models/230602_141343.single_instance.n=108/"

# Find all videos in input folder
videos=$(find $input_path -type f -name "*.mp4" -print)

# For each video, use sleap-track terminal command with existing model to track ball positions
for video in $videos; do
    video_name=$(basename $video .mp4)
    output_folder=$(dirname $video)
    output_file="${output_folder}/${video_name}_tracked.slp"
    sleap-track $video --model $model_path --output $output_file
    
    # execute the command to extract and save .h5 files
    sleap-convert "$output_file" --format analysis
done

#TODO : make the script loop over all files. 

#TODO : make the script run as a background process, always checking for non processed videos

#TODO: Add a way to resume an aborted processing in a given folder, by checking already existing h5 files, 
#skipping them and processing folder not yet done or that would have a partial slp file.
