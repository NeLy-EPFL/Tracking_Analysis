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

# Set input path
datafolder="/mnt/upramdya_data/MD/F1_Tracks/Videos"

# Print out the value of datafolder
echo "datafolder: $datafolder"

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

# Collect all fly slp files that need h5 conversion
files_to_convert=()

# For each subdirectory, check for fly .slp files and corresponding .h5 files
for subdir in "${subdirs[@]}"; do
    echo "Processing directory: $subdir"
    # Only process directories that have been pre-processed or fully processed
    if [[ $subdir == *_Checked* ]]; then

        # Find all fly tracking .slp files in this folder
        slp_files_fly=($(find "$subdir" -maxdepth 1 -name "*tracked_fly*.slp" 2>/dev/null))

        # For each fly .slp file, check if corresponding .h5 file exists
        for slp_file in "${slp_files_fly[@]}"; do
            slp_basename=$(basename "$slp_file" .slp)
            output_folder=$(dirname "$slp_file")
            
            # Look for corresponding .h5 or .analysis.h5 files
            h5_files=($(find "$output_folder" -maxdepth 1 -name "${slp_basename}.h5" -o -name "${slp_basename}.analysis.h5" 2>/dev/null))
            
            # Debug output in dry run mode
            if [ "$DRY_RUN" = true ]; then
                echo "  Fly .slp file: $(basename "$slp_file")"
                echo "    Corresponding .h5 files found: ${#h5_files[@]} - ${h5_files[*]}"
            fi

            # Check if h5 conversion is needed
            if [ ${#h5_files[@]} -eq 0 ]; then
                echo "Adding $slp_file to conversion list (missing .h5)."
                files_to_convert+=("$slp_file")
            else
                if [ "$DRY_RUN" = true ]; then
                    echo "H5 file already exists for $(basename "$slp_file"). Skipping."
                fi
            fi
        done
    fi
done

# Show summary and process each file
echo ""
echo "=== CONVERSION SUMMARY ==="
echo "Total fly .slp files to convert: ${#files_to_convert[@]}"
if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN - The following would be converted:"
    for slp_file in "${files_to_convert[@]}"; do
        echo "  - $(basename "$slp_file")"
    done
    echo "=== END DRY RUN ==="
    exit 0
fi

# Convert each .slp file to .h5
for slp_file in "${files_to_convert[@]}"; do
    echo "Converting fly tracking to h5 format: $(basename "$slp_file")"
    sleap-convert "$slp_file" --format analysis
    
    # Check if conversion was successful
    if [ $? -eq 0 ]; then
        echo "Successfully converted: $(basename "$slp_file")"
    else
        echo "ERROR: Failed to convert: $(basename "$slp_file")"
    fi
done

echo "All conversions complete."