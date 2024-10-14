#!/bin/bash

# Activate the right conda environment
source activate sleap

# Set input and output paths
datafolder="/mnt/upramdya_data/MD/F1_Tracks/Videos"
model_path_ball_centroid="/mnt/upramdya_data/_Tracking_models/Sleap/mazerecorder/BallTracking/models/240926_141251.centroid.n=102"
model_path_ball_centered_instance="/mnt/upramdya_data/_Tracking_models/Sleap/mazerecorder/BallTracking/models/240926_151129.centered_instance.n=102"
model_path_fly="/mnt/upramdya_data/_Tracking_models/Sleap/mazerecorder/FlyTracking/Thorax/Labels/models/240924_164931.single_instance.n=192"

# Print out the values of datafolder and model paths
echo "datafolder: $datafolder"
echo "model_path_ball_centroid: $model_path_ball_centroid"
echo "model_path_ball_centered_instance: $model_path_ball_centered_instance"
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

# Collect all videos to be processed
videos_to_process=()

# For each subdirectory, check if .slp and .h5 files already exist
for subdir in "${subdirs[@]}"; do
    echo "Processing directory: $subdir"
    # Only process directories that have been pre-processed or fully processed
    if [[ $subdir == *_Checked* ]]; then

        # Find all videos in this folder
        videos=$(find $subdir -maxdepth 1 -type f -name "*.mp4" -print)

        # For each video, check if tracking files already exist
        for video in $videos; do
            video_name=$(basename $video .mp4)
            output_folder=$(dirname $video)
            output_file_ball="${output_folder}/${video_name}_tracked_ball.slp"
            output_file_fly="${output_folder}/${video_name}_tracked_fly.slp"

            slp_file_ball=$(find $subdir -maxdepth 1 -type f -name "${video_name}_tracked_ball.slp")
            h5_file_ball=$(find $subdir -maxdepth 1 -type f -name "${video_name}_tracked_ball.h5")
            slp_file_fly=$(find $subdir -maxdepth 1 -type f -name "${video_name}_tracked_fly.slp")
            h5_file_fly=$(find $subdir -maxdepth 1 -type f -name "${video_name}_tracked_fly.h5")

            # If .slp and .h5 files for ball do not exist, add to processing list
            if [ -z "$slp_file_ball" ] && [ -z "$h5_file_ball" ]; then
                echo "Adding $video to processing list for ball tracking."
                videos_to_process+=("$video")
            else
                echo "Tracking files for ball already exist for $video. Skipping."
            fi

            # If .slp and .h5 files for fly do not exist, add to processing list
            if [ -z "$slp_file_fly" ] && [ -z "$h5_file_fly" ]; then
                echo "Adding $video to processing list for fly tracking."
                videos_to_process+=("$video")
            else
                echo "Tracking files for fly already exist for $video. Skipping."
            fi
        done
    fi
done

# Process each video
for video in "${videos_to_process[@]}"; do
    video_name=$(basename "$video" .mp4)
    output_folder=$(dirname "$video")
    output_file_ball="${output_folder}/${video_name}_tracked_ball.slp"
    output_file_fly="${output_folder}/${video_name}_tracked_fly.slp"

    # Perform tracking for ball
    echo "Tracking ball for video: $video"
    sleap-track "$video" --model "$model_path_ball_centroid" --model "$model_path_ball_centered_instance" --batch_size 16 --tracking.tracker simple --tracking.max_tracking 1 --tracking.max_tracks 2 --output "$output_file_ball" --verbosity rich
    sleap-convert "$output_file_ball" --format analysis
    echo "Ball tracking complete for video: $video"

    # Perform tracking for fly
    echo "Tracking fly for video: $video"
    sleap-track "$video" --model "$model_path_fly" --batch_size 16 --output "$output_file_fly" --verbosity rich
    sleap-convert "$output_file_fly" --format analysis
    echo "Fly tracking complete for video: $video"
done

echo "All processing complete."