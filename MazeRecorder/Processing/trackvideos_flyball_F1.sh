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

            # If .slp and .h5 files for ball do not exist, add to batch processing list
            if [ -z "$slp_file_ball" ] && [ -z "$h5_file_ball" ]; then
                videos_to_process+=("$video")
            fi

            # If .slp and .h5 files for fly do not exist, add to batch processing list
            if [ -z "$slp_file_fly" ] && [ -z "$h5_file_fly" ]; then
                videos_to_process+=("$video")
            fi
        done
    fi
done

# Function to process videos in batches
process_batch() {
    local batch=("$@")
    local batch_size=${#batch[@]}
    echo "Processing batch of $batch_size videos..."

    # Perform batch processing for ball tracking
    sleap-track "${batch[@]}" --model $model_path_ball_centroid --model $model_path_ball_centered_instance --output "${datafolder}/tracked_ball.slp" --verbosity rich
    sleap-convert "${datafolder}/tracked_ball.slp" --format analysis

    # Perform batch processing for fly tracking
    sleap-track "${batch[@]}" --model $model_path_fly --output "${datafolder}/tracked_fly.slp" --verbosity rich
    sleap-convert "${datafolder}/tracked_fly.slp" --format analysis

    echo "Batch processing complete."
}

# Define batch size
batch_size=16

# Split videos into batches and process each batch
total_videos=${#videos_to_process[@]}
for ((i=0; i<total_videos; i+=batch_size)); do
    batch=("${videos_to_process[@]:i:batch_size}")
    process_batch "${batch[@]}"
done

echo "All batch processing complete."