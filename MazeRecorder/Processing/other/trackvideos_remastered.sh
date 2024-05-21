#!/bin/bash

# activate the right conda environment
source activate sleap

# Set input and output paths
datafolder="/mnt/labserver/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Videos/"
model_path="/mnt/labserver/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Sleap/Labels/models/230602_141343.single_instance.n=108/"

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
            output_file="${output_folder}/${video_name}_tracked_ball.slp"
            sleap-track $video --model $model_path --output $output_file
            
            # execute the command to extract and save .h5 files
            sleap-convert "$output_file" --format analysis
        done
    fi
done




#TODO : make the script loop over all files. 

#TODO : make the script run as a background process, always checking for non processed videos

#TODO: Add a way to resume an aborted processing in a given folder, by checking already existing h5 files, 
#skipping them and processing folder not yet done or that would have a partial slp file.

#TODO: Make the sleap command less verbose, or at least make it so that the output is not displayed in the terminal
