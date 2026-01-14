#!/bin/bash

# Check for flags
DRY_RUN=false
CHECK_MODE=false
CHECK_AND_PROCESS=false

# Parse flags
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run|-n)
            DRY_RUN=true
            echo "=== DRY RUN MODE - No actual processing will occur ==="
            shift
            ;;
        --check|-c)
            CHECK_MODE=true
            echo "=== CHECK MODE - Analyzing processing status ==="
            shift
            ;;
        --check-and-process|-cp)
            CHECK_AND_PROCESS=true
            echo "=== CHECK AND PROCESS MODE - Will show status then process unfinished items ==="
            shift
            ;;
        *)
            break  # End of flags, start of directory arguments
            ;;
    esac
done

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
processed_videos=()
directories_with_unprocessed=()
directories_fully_processed=()

# For each subdirectory, check if .slp files already exist
for subdir in "${subdirs[@]}"; do
    echo "Processing directory: $subdir"
    # Only process directories that have been pre-processed or fully processed
    if [[ $subdir == *_Checked* ]]; then

        # Find all videos in this folder
        videos=$(find $subdir -maxdepth 1 -type f -name "*.mp4" -print)
        
        # Track status for this directory
        dir_has_unprocessed=false
        dir_video_count=0
        dir_processed_count=0

        # For each video, check if tracking files already exist
        for video in $videos; do
            ((dir_video_count++))
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

            # Track if this video is fully processed
            video_fully_processed=true

            # Check ball tracking status
            if [ ${#slp_files_ball[@]} -eq 0 ]; then
                echo "Adding $video to processing list for ball tracking (missing .slp)."
                videos_to_process+=("$video:ball:slp")
                dir_has_unprocessed=true
                video_fully_processed=false
            elif [ ${#h5_files_ball[@]} -eq 0 ]; then
                echo "Adding $video to processing list for ball h5 conversion (missing .h5)."
                videos_to_process+=("$video:ball:h5")
                dir_has_unprocessed=true
                video_fully_processed=false
            else
                if [ "$DRY_RUN" = true ]; then
                    echo "Ball tracking files (.slp and .h5) already exist for $video. Skipping."
                fi
            fi

            # Check fly tracking status
            if [ ${#slp_files_fly[@]} -eq 0 ]; then
                echo "Adding $video to processing list for fly tracking (missing .slp)."
                videos_to_process+=("$video:fly:slp")
                dir_has_unprocessed=true
                video_fully_processed=false
            elif [ ${#h5_files_fly[@]} -eq 0 ]; then
                echo "Adding $video to processing list for fly h5 conversion (missing .h5)."
                videos_to_process+=("$video:fly:h5")
                dir_has_unprocessed=true
                video_fully_processed=false
            else
                if [ "$DRY_RUN" = true ]; then
                    echo "Fly tracking files (.slp and .h5) already exist for $video. Skipping."
                fi
            fi
            
            # Add to processed list if fully done
            if [ "$video_fully_processed" = true ]; then
                processed_videos+=("$video")
                ((dir_processed_count++))
            fi
        done
        
        # Track directory status
        if [ "$dir_has_unprocessed" = true ]; then
            directories_with_unprocessed+=("$subdir ($dir_processed_count/$dir_video_count videos processed)")
        else
            if [ $dir_video_count -gt 0 ]; then
                directories_fully_processed+=("$subdir ($dir_video_count/$dir_video_count videos processed)")
            fi
        fi
    fi
done

# Show summary and process each video
echo ""
echo "=== PROCESSING SUMMARY ==="
echo "Total items to process: ${#videos_to_process[@]}"
echo "Total processed videos: ${#processed_videos[@]}"

# Check mode output
if [ "$CHECK_MODE" = true ] || [ "$CHECK_AND_PROCESS" = true ]; then
    echo ""
    if [ "$CHECK_MODE" = true ]; then
        echo "=== CHECK MODE RESULTS ==="
    else
        echo "=== CHECK AND PROCESS MODE RESULTS ==="
    fi
    echo ""
    
    if [ ${#directories_fully_processed[@]} -gt 0 ]; then
        echo "✅ FULLY PROCESSED DIRECTORIES:"
        for dir in "${directories_fully_processed[@]}"; do
            echo "  - $dir"
        done
        echo ""
    fi
    
    if [ ${#directories_with_unprocessed[@]} -gt 0 ]; then
        echo "❌ DIRECTORIES WITH UNPROCESSED VIDEOS:"
        for dir in "${directories_with_unprocessed[@]}"; do
            echo "  - $dir"
        done
        echo ""
        echo "📋 DETAILED UNPROCESSED ITEMS:"
        for item in "${videos_to_process[@]}"; do
            IFS=":" read -r video track_type process_type <<< "$item"
            video_name=$(basename "$video" .mp4)
            directory=$(dirname "$video")
            directory_name=$(basename "$directory")
            echo "  - $directory_name/$video_name: $track_type $process_type"
        done
    else
        echo "🎉 ALL DIRECTORIES ARE FULLY PROCESSED!"
    fi
    
    echo ""
    echo "=== SUMMARY ==="
    echo "Directories fully processed: ${#directories_fully_processed[@]}"
    echo "Directories with unprocessed videos: ${#directories_with_unprocessed[@]}"
    echo "Videos needing processing: ${#videos_to_process[@]}"
    echo "Videos already processed: ${#processed_videos[@]}"
    
    # Exit if just checking, continue if check-and-process
    if [ "$CHECK_MODE" = true ]; then
        echo "=== END CHECK MODE ==="
        exit 0
    elif [ "$CHECK_AND_PROCESS" = true ]; then
        if [ ${#videos_to_process[@]} -eq 0 ]; then
            echo "=== NOTHING TO PROCESS - ALL VIDEOS ARE ALREADY COMPLETE ==="
            exit 0
        else
            echo ""
            echo "🚀 PROCEEDING WITH PROCESSING OF ${#videos_to_process[@]} UNFINISHED ITEMS..."
            echo "=== STARTING PROCESSING ==="
            # Continue to processing section below
        fi
    fi
fi

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

if [ "$CHECK_AND_PROCESS" = true ]; then
    echo ""
    echo "🎉 CHECK-AND-PROCESS COMPLETE!"
    echo "All unfinished videos have been processed."
else
    echo "All processing complete."
fi