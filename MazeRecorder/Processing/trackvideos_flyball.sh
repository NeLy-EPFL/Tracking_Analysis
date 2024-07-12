#!/bin/bash

# activate the right conda environment
source activate sleap

# Set input and output paths
datafolder="/mnt/upramdya_data/MD/MultiMazeRecorder/Videos/"
model_path_ball="/mnt/upramdya_data/MD/MultiMazeRecorder/Sleap/Labels/models/230602_141343.single_instance.n=108/"
model_path_fly="/mnt/upramdya_data/MD/MultiMazeRecorder/Sleap/Labels/Thorax_labels.v001.slp.training_job/models/230825_101219.single_instance/"

# Print out the values of datafolder and model_path
echo "datafolder: $datafolder"
echo "model_path_ball: $model_path_ball"
echo "model_path_fly: $model_path_fly"

# Collect all directories within datafolder
subdirs=($(find "$datafolder" -type d))

# If arguments are provided, filter subdirs to match those arguments
if [ $# -gt 0 ]; then
    filtered_subdirs=()
    for arg in "$@"; do
        for subdir in "${subdirs[@]}"; do
            if [[ "$subdir" == *"$arg"* ]]; then
                filtered_subdirs+=("$subdir")
            fi
        done
    done
    subdirs=("${filtered_subdirs[@]}")
    echo "directories to be processed: ${subdirs[@]}"
fi
#TODO: This is not working yet.

# For each subdirectory, check if .slp and .h5 files already exist
for subdir in "${subdirs[@]}"; do
    echo "Processing directory: $subdir"
    # Only process directories that have been pre-processed or fully processed
    if [[ $subdir == *_Checked* ]]; then # || [[ $subdir == *_Tracked* ]] has been removed for efficiency

        slp_file_ball=$(find $subdir -maxdepth 1 -type f -name "*_tracked_ball.slp")
        h5_file_ball=$(find $subdir -maxdepth 1 -type f -name "*_tracked_ball.h5")
        slp_file_fly=$(find $subdir -maxdepth 1 -type f -name "*_tracked_fly.slp")
        h5_file_fly=$(find $subdir -maxdepth 1 -type f -name "*_tracked_fly.h5")

        # Find all videos in this folder
        videos=$(find $subdir -maxdepth 1 -type f -name "*.mp4" -print)

        # For each video, use sleap-track terminal command with existing model to track ball positions
        for video in $videos; do
            video_name=$(basename $video .mp4)
            output_folder=$(dirname $video)
            output_file_ball="${output_folder}/${video_name}_tracked_ball.slp"
            output_file_fly="${output_folder}/${video_name}_tracked_fly.slp"

            # If .slp and .h5 files for ball do not exist, track ball
            # Print a message to the terminal to indicate that the video is being processed
            echo "Processing video: $video"

            if [ -z "$slp_file_ball" ] && [ -z "$h5_file_ball" ]; then
                echo "No tracking data found for the ball position. Tracking ball..."
                sleap-track $video --model $model_path_ball --output $output_file_ball --verbosity rich
                sleap-convert "$output_file_ball" --format analysis
            fi

            # If .slp and .h5 files for fly do not exist, track fly
            if [ -z "$slp_file_fly" ] && [ -z "$h5_file_fly" ]; then
                echo "No tracking data found for the fly position. Tracking fly..."
                sleap-track $video --model $model_path_fly --output $output_file_fly --verbosity rich
                sleap-convert "$output_file_fly" --format analysis
            fi

            echo "Processing of $video complete."
        done
    fi
done

#script_dir="$(dirname "$0")"

# Run check tracks python script

#python "$script_dir/CheckTracks.py"
