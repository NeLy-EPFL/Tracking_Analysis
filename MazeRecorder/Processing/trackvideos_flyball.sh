#!/bin/bash

# activate the right conda environment (check if already activated)
if [[ "$CONDA_DEFAULT_ENV" != "sleap" ]]; then
    source activate sleap
fi

# Set input and output paths
datafolder="/mnt/upramdya_data/MD/MultiMazeRecorder/Videos/"
model_path_ball="/mnt/upramdya_data/MD/MultiMazeRecorder/Sleap/Labels/models/230602_141343.single_instance.n=108/"
model_path_fly="/mnt/upramdya_data/MD/MultiMazeRecorder/Sleap/Labels/Thorax_labels.v001.slp.training_job/models/230825_101219.single_instance/"

# Print out the values of datafolder and model_path
echo "datafolder: $datafolder"
echo "model_path_ball: $model_path_ball"
echo "model_path_fly: $model_path_fly"

# Collect directories within datafolder
if [ $# -gt 0 ]; then
    # If arguments are provided, search directly for matching directories
    subdirs=()
    for arg in "$@"; do
        # Use find with pattern matching to directly find relevant directories
        while IFS= read -r -d '' dir; do
            subdirs+=("$dir")
        done < <(find "$datafolder" -type d -name "*${arg}*" -print0)
    done
    echo "directories to be processed: ${subdirs[*]}"
else
    # If no arguments, collect all directories (original behavior)
    mapfile -t subdirs < <(find "$datafolder" -type d)
fi
#TODO: This is not working yet.

# For each subdirectory, check if .slp and .h5 files already exist
for subdir in "${subdirs[@]}"; do
    echo "Processing directory: $subdir"
    # Only process directories that have been pre-processed or fully processed
    if [[ $subdir == *_Checked* ]]; then # || [[ $subdir == *_Tracked* ]] has been removed for efficiency
        echo "Directory matches *_Checked* pattern, proceeding..."

        # Find all videos in this folder and its subdirectories
        videos=$(find $subdir -type f -name "*.mp4" -print)
        echo "Found videos: $videos"
        
        if [ -z "$videos" ]; then
            echo "No .mp4 videos found in $subdir"
            continue
        fi

        # For each video, use sleap-track terminal command with existing model to track ball positions
        for video in $videos; do
            video_name=$(basename $video .mp4)
            output_folder=$(dirname $video)
            output_file_ball="${output_folder}/${video_name}_tracked_ball.slp"
            output_file_fly="${output_folder}/${video_name}_tracked_fly.slp"

            # Check for existing tracking files in the same directory as the video
            slp_file_ball=$(find $output_folder -maxdepth 1 -type f -name "*_tracked_ball.slp")
            h5_file_ball=$(find $output_folder -maxdepth 1 -type f -name "*_tracked_ball.h5")
            slp_file_fly=$(find $output_folder -maxdepth 1 -type f -name "*_tracked_fly.slp")
            h5_file_fly=$(find $output_folder -maxdepth 1 -type f -name "*_tracked_fly.h5")

            echo "Debug - Found tracking files in $output_folder:"
            echo "  Ball .slp: $slp_file_ball"
            echo "  Ball .h5: $h5_file_ball"
            echo "  Fly .slp: $slp_file_fly"
            echo "  Fly .h5: $h5_file_fly"

            # If .slp and .h5 files for ball do not exist, track ball
            # Print a message to the terminal to indicate that the video is being processed
            echo "Processing video: $video"

            echo "Debug - Checking ball tracking conditions:"
            echo "  slp_file_ball empty: $([ -z "$slp_file_ball" ] && echo "true" || echo "false")"
            echo "  h5_file_ball empty: $([ -z "$h5_file_ball" ] && echo "true" || echo "false")"
            
            if [ -z "$slp_file_ball" ] && [ -z "$h5_file_ball" ]; then
                echo "No tracking data found for the ball position. Tracking ball..."
                sleap-track $video --model $model_path_ball --batch_size 16 --output $output_file_ball --verbosity rich
                sleap-convert "$output_file_ball" --format analysis
            else
                echo "Ball tracking files already exist, skipping ball tracking."
            fi

            echo "Debug - Checking fly tracking conditions:"
            echo "  slp_file_fly empty: $([ -z "$slp_file_fly" ] && echo "true" || echo "false")"
            echo "  h5_file_fly empty: $([ -z "$h5_file_fly" ] && echo "true" || echo "false")"

            # If .slp and .h5 files for fly do not exist, track fly
            if [ -z "$slp_file_fly" ] && [ -z "$h5_file_fly" ]; then
                echo "No tracking data found for the fly position. Tracking fly..."
                sleap-track $video --model $model_path_fly --batch_size 16 --output $output_file_fly --verbosity rich
                sleap-convert "$output_file_fly" --format analysis
            else
                echo "Fly tracking files already exist, skipping fly tracking."
            fi

            echo "Processing of $video complete."
        done
    else
        echo "Directory $subdir does not match *_Checked* pattern, skipping..."
    fi
done

#script_dir="$(dirname "$0")"

# Run check tracks python script

#python "$script_dir/CheckTracks.py"
