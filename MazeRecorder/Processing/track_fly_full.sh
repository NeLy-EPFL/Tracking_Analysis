#!/bin/bash

# activate the right conda environment
echo "$(date): Activating conda environment 'sleap_14'"
source activate sleap_14

# Set input and output paths
datafolder="/mnt/upramdya_data/MD/MultiMazeRecorder/Videos/"
model_path_ball="/mnt/upramdya_data/MD/MultiMazeRecorder/Sleap/Labels/models/230602_141343.single_instance.n=108/"
model_path_fly="/mnt/upramdya_data/_Tracking_models/Sleap/mazerecorder/FlyTracking/FullBody/models/240809_102156.single_instance.n=421"

# Print out the values of datafolder and model paths
echo "$(date): datafolder: $datafolder"
echo "$(date): model_path_ball: $model_path_ball"
echo "$(date): model_path_fly: $model_path_fly"

# Function to track videos
track_videos() {
    local videos=("$@")
    for video in "${videos[@]}"; do
        echo "$(date): Processing video: $video"
        video_name=$(basename "$video" .mp4)
        output_folder=$(dirname "$video")
        output_file_ball="${output_folder}/${video_name}_tracked_ball.slp"
        output_file_fly="${output_folder}/${video_name}_tracked_fly_full.slp"

        # Track ball if .slp and .h5 files do not exist
        if [ ! -f "$output_file_ball" ] && [ ! -f "${output_file_ball%.slp}.h5" ]; then
            echo "$(date): No tracking data found for the ball position. Tracking ball..."
            sleap-track "$video" --model "$model_path_ball" --output "$output_file_ball" --verbosity rich
            echo "$(date): Converting ball tracking data to analysis format..."
            sleap-convert "$output_file_ball" --format analysis
        fi

        # Track fly if .slp and .h5 files do not exist
        if [ ! -f "$output_file_fly" ] && [ ! -f "${output_file_fly%.slp}.h5" ]; then
            echo "$(date): No tracking data found for the fly position. Tracking fly..."
            sleap-track "$video" --model "$model_path_fly" --output "$output_file_fly" --verbosity rich
            echo "$(date): Converting fly tracking data to analysis format..."
            sleap-convert "$output_file_fly" --format analysis
        fi

        echo "$(date): Processing of $video complete."
    done
}

# Collect all videos within datafolder
echo "$(date): Collecting all videos within datafolder..."
all_videos=($(find "$datafolder" -type f -name "*.mp4"))
echo "$(date): Found ${#all_videos[@]} videos."

# If arguments are provided, filter videos to match those arguments
if [ $# -gt 0 ]; then
    echo "$(date): Filtering videos based on provided arguments..."
    filtered_videos=()
    for arg in "$@"; do
        for video in "${all_videos[@]}"; do
            if [[ "$video" == *"$arg"* ]]; then
                filtered_videos+=("$video")
            fi
        done
    done
    all_videos=("${filtered_videos[@]}")
    echo "$(date): Videos to be processed: ${#all_videos[@]}"
fi

# Track the videos
echo "$(date): Starting video tracking..."
track_videos "${all_videos[@]}"
echo "$(date): Video tracking complete."
