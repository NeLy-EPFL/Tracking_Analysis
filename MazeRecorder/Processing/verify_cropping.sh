#!/bin/bash
# Quick bash script to verify Array2F1tracks cropping
# This provides a fast overview of image counts

echo "=== Array2F1tracks Cropping Verification ==="
echo

DATA_DIR="/home/matthias/Videos/"

if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Data directory $DATA_DIR does not exist"
    exit 1
fi

cd "$DATA_DIR"

# Find all _Recorded folders
RECORDED_FOLDERS=($(find . -maxdepth 1 -type d -name "*_Recorded" | sort))

if [ ${#RECORDED_FOLDERS[@]} -eq 0 ]; then
    echo "❌ No _Recorded folders found in $DATA_DIR"
    exit 1
fi

echo "Found ${#RECORDED_FOLDERS[@]} recorded folders to check:"
echo

total_checked=0
total_success=0
total_failed=0

for recorded in "${RECORDED_FOLDERS[@]}"; do
    # Remove ./ prefix and get folder name
    recorded_name=$(basename "$recorded")
    cropped_name="${recorded_name/_Recorded/_Cropped}"
    cropped_path="./$cropped_name"
    
    echo "────────────────────────────────────────"
    echo "Checking: $recorded_name"
    
    # Count original images
    original_count=$(find "$recorded" -maxdepth 1 -name "*.jpg" -o -name "*.JPG" | wc -l)
    echo "Original images: $original_count"
    
    if [ ! -d "$cropped_path" ]; then
        echo "❌ Cropped folder missing: $cropped_name"
        ((total_failed++))
        ((total_checked++))
        continue
    fi
    
    echo "Cropped folder: $cropped_name"
    
    # Check each arena
    all_good=true
    for arena in {1..9}; do
        left_path="$cropped_path/arena$arena/Left"
        right_path="$cropped_path/arena$arena/Right"
        
        if [ ! -d "$left_path" ] || [ ! -d "$right_path" ]; then
            echo "❌ Arena $arena: Missing Left or Right folder"
            all_good=false
            continue
        fi
        
        left_count=$(find "$left_path" -maxdepth 1 -name "*.jpg" -o -name "*.JPG" | wc -l)
        right_count=$(find "$right_path" -maxdepth 1 -name "*.jpg" -o -name "*.JPG" | wc -l)
        
        if [ "$left_count" -eq "$original_count" ] && [ "$right_count" -eq "$original_count" ]; then
            echo "✅ Arena $arena: L=$left_count, R=$right_count"
        else
            echo "❌ Arena $arena: L=$left_count, R=$right_count (expected $original_count)"
            all_good=false
        fi
    done
    
    if $all_good; then
        echo "✅ SUCCESS: All arenas processed correctly"
        ((total_success++))
    else
        echo "❌ FAILED: Some issues found"
        ((total_failed++))
    fi
    
    ((total_checked++))
    echo
done

echo "========================================"
echo "SUMMARY"
echo "========================================"
echo "Total folders checked: $total_checked"
echo "✅ Successfully verified: $total_success"
echo "❌ Failed verification: $total_failed"

if [ $total_failed -eq 0 ]; then
    echo
    echo "🎉 ALL FOLDERS VERIFIED SUCCESSFULLY!"
    exit 0
else
    echo
    echo "⚠️  SOME FOLDERS HAVE ISSUES"
    echo "Run the Python script for detailed analysis:"
    echo "python3 /home/matthias/Tracking_Analysis/MazeRecorder/Processing/verify_cropping.py"
    exit 1
fi