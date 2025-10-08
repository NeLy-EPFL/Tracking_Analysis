#!/bin/bash

# Check for dry run flag
DRY_RUN=false
if [[ "$1" == "--dry-run" || "$1" == "-n" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE - No actual processing will occur ==="
    shift  # Remove the dry-run flag from arguments
fi

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

# For each subdirectory, check if .slp files already exist
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
            output_h5_ball="${output_folder}/${video_name}_tracked_ball.h5"
            output_h5_fly="${output_folder}/${video_name}_tracked_fly.h5"

            # Check for ball tracking files using pattern matching
            slp_files_ball=($(find "$output_folder" -maxdepth 1 -name "*tracked_ball*.slp" 2>/dev/null))
            h5_files_ball=($(find "$output_folder" -maxdepth 1 -name "*tracked_ball*.h5" -o -name "*tracked_ball*.analysis.h5" 2>/dev/null))
            
            # Check for fly tracking files using pattern matching
            slp_files_fly=($(find "$output_folder" -maxdepth 1 -name "*tracked_fly*.slp" 2>/dev/null))
            h5_files_fly=($(find "$output_folder" -maxdepth 1 -name "*tracked_fly*.h5" -o -name "*tracked_fly*.analysis.h5" 2>/dev/null))
            
            # Debug output in dry run mode
            if [ "$DRY_RUN" = true ]; then
                echo "  Video: $video_name"
                echo "    Ball .slp files found: ${#slp_files_ball[@]} - ${slp_files_ball[*]}"
                echo "    Ball .h5 files found:  ${#h5_files_ball[@]} - ${h5_files_ball[*]}"
                echo "    Fly .slp files found:  ${#slp_files_fly[@]} - ${slp_files_fly[*]}"
                echo "    Fly .h5 files found:   ${#h5_files_fly[@]} - ${h5_files_fly[*]}"
            fi

            # Check ball tracking status
            if [ ${#slp_files_ball[@]} -eq 0 ]; then
                echo "Adding $video to processing list for ball tracking (missing .slp)."
                videos_to_process+=("$video:ball:slp")
            elif [ ${#h5_files_ball[@]} -eq 0 ]; then
                echo "Adding $video to processing list for ball h5 conversion (missing .h5)."
                videos_to_process+=("$video:ball:h5")
            else
                if [ "$DRY_RUN" = true ]; then
                    echo "Ball tracking files (.slp and .h5) already exist for $video. Skipping."
                fi
            fi

            # Check fly tracking status
            if [ ${#slp_files_fly[@]} -eq 0 ]; then
                echo "Adding $video to processing list for fly tracking (missing .slp)."
                videos_to_process+=("$video:fly:slp")
            elif [ ${#h5_files_fly[@]} -eq 0 ]; then
                echo "Adding $video to processing list for fly h5 conversion (missing .h5)."
                videos_to_process+=("$video:fly:h5")
            else
                if [ "$DRY_RUN" = true ]; then
                    echo "Fly tracking files (.slp and .h5) already exist for $video. Skipping."
                fi
            fi
        done
    fi
done

# Show summary and process each video
echo ""
echo "=== PROCESSING SUMMARY ==="
echo "Total items to process: ${#videos_to_process[@]}"
if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN - The following would be processed:"
    for item in "${videos_to_process[@]}"; do
        IFS=":" read -r video track_type process_type <<< "$item"
        video_name=$(basename "$video" .mp4)
        echo "  - $video_name: $track_type $process_type"
    done
    echo "=== END DRY RUN ==="
    exit 0
fi

# Process each video
for item in "${videos_to_process[@]}"; do
    IFS=":" read -r video track_type process_type <<< "$item"
    video_name=$(basename "$video" .mp4)
    output_folder=$(dirname "$video")

    if [ "$track_type" == "ball" ]; then
        output_file_ball="${output_folder}/${video_name}_tracked_ball.slp"
        
        if [ "$process_type" == "slp" ]; then
            # Perform tracking for ball
            echo "Tracking ball for video: $video"
            sleap-track "$video" --model "$model_path_ball_centroid" --model "$model_path_ball_centered_instance" --batch_size 16 --max_instances 2 --output "$output_file_ball" --verbosity rich
            echo "Ball tracking complete for video: $video"
            
            # After successful tracking, convert to h5
            echo "Converting ball tracking to h5 format for video: $video"
            sleap-convert "$output_file_ball" --format analysis
            echo "Ball h5 conversion complete for video: $video"
        elif [ "$process_type" == "h5" ]; then
            # Only convert existing slp to h5
            echo "Converting existing ball tracking to h5 format for video: $video"
            sleap-convert "$output_file_ball" --format analysis
            echo "Ball h5 conversion complete for video: $video"
        fi
        
    elif [ "$track_type" == "fly" ]; then
        output_file_fly="${output_folder}/${video_name}_tracked_fly.slp"
        
        if [ "$process_type" == "slp" ]; then
            # Perform tracking for fly
            echo "Tracking fly for video: $video"
            sleap-track "$video" --model "$model_path_fly" --batch_size 16 --output "$output_file_fly" --verbosity rich
            echo "Fly tracking complete for video: $video"
            
            # After successful tracking, convert to h5
            echo "Converting fly tracking to h5 format for video: $video"
            sleap-convert "$output_file_fly" --format analysis
            echo "Fly h5 conversion complete for video: $video"
        elif [ "$process_type" == "h5" ]; then
            # Only convert existing slp to h5
            echo "Converting existing fly tracking to h5 format for video: $video"
            sleap-convert "$output_file_fly" --format analysis
            echo "Fly h5 conversion complete for video: $video"
        fi
    fi
done

echo "All processing complete."