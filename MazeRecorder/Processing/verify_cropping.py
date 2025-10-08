#!/usr/bin/env python3
"""
Script to verify that Array2F1tracks cropping worked properly.
Checks that all images were processed and split correctly into Left/Right subfolders.
"""

import os
import sys
from pathlib import Path
import argparse
from collections import defaultdict

def count_images(folder_path):
    """Count the number of .jpg files in a folder."""
    folder = Path(folder_path)
    if not folder.exists():
        return 0
    
    # Count both .jpg and .JPG files
    jpg_files = list(folder.glob("*.[jJ][pP][gG]"))
    return len(jpg_files)

def verify_single_folder(recorded_folder, cropped_folder):
    """Verify that a single recorded folder was processed correctly."""
    print(f"\n{'='*60}")
    print(f"Verifying: {recorded_folder.name}")
    print(f"{'='*60}")
    
    # Count original images
    original_count = count_images(recorded_folder)
    print(f"Original folder images: {original_count}")
    
    if original_count == 0:
        print("❌ ERROR: No images found in original folder!")
        return False
    
    # Check if cropped folder exists
    if not cropped_folder.exists():
        print(f"❌ ERROR: Cropped folder does not exist: {cropped_folder}")
        return False
    
    print(f"Cropped folder: {cropped_folder.name}")
    
    # Check each arena
    all_good = True
    arena_summary = []
    
    for arena_num in range(1, 10):  # Arena 1 to 9
        arena_folder = cropped_folder / f"arena{arena_num}"
        left_folder = arena_folder / "Left"
        right_folder = arena_folder / "Right"
        
        # Check if arena folders exist
        if not arena_folder.exists():
            print(f"❌ Arena {arena_num}: Folder missing")
            all_good = False
            continue
            
        if not left_folder.exists():
            print(f"❌ Arena {arena_num}: Left folder missing")
            all_good = False
            continue
            
        if not right_folder.exists():
            print(f"❌ Arena {arena_num}: Right folder missing")
            all_good = False
            continue
        
        # Count images in each subfolder
        left_count = count_images(left_folder)
        right_count = count_images(right_folder)
        
        arena_status = "✅"
        if left_count != original_count or right_count != original_count:
            arena_status = "❌"
            all_good = False
        
        arena_summary.append({
            'arena': arena_num,
            'left': left_count,
            'right': right_count,
            'status': arena_status
        })
        
        print(f"{arena_status} Arena {arena_num}: Left={left_count}, Right={right_count}")
    
    # Summary
    print(f"\n{'─'*40}")
    print("SUMMARY:")
    print(f"Expected images per subfolder: {original_count}")
    
    if all_good:
        print("✅ ALL ARENAS PROCESSED CORRECTLY!")
        return True
    else:
        print("❌ SOME ISSUES FOUND!")
        
        # Detailed error report
        print("\nDetailed Issues:")
        for arena in arena_summary:
            if arena['status'] == "❌":
                expected = original_count
                left_diff = arena['left'] - expected
                right_diff = arena['right'] - expected
                print(f"  Arena {arena['arena']}: Left={arena['left']} ({left_diff:+d}), Right={arena['right']} ({right_diff:+d})")
        
        return False

def find_folder_pairs(data_folder):
    """Find all _Recorded and corresponding _Cropped folder pairs."""
    data_folder = Path(data_folder)
    pairs = []
    
    for folder in data_folder.iterdir():
        if folder.is_dir() and folder.name.endswith("_Recorded"):
            cropped_name = folder.name.replace("_Recorded", "_Cropped")
            cropped_folder = data_folder / cropped_name
            pairs.append((folder, cropped_folder))
    
    return pairs

def verify_all_folders(data_folder):
    """Verify all processed folders in the data directory."""
    data_folder = Path(data_folder)
    
    if not data_folder.exists():
        print(f"❌ ERROR: Data folder does not exist: {data_folder}")
        return False
    
    print(f"Checking folders in: {data_folder}")
    
    folder_pairs = find_folder_pairs(data_folder)
    
    if not folder_pairs:
        print("❌ No _Recorded folders found!")
        return False
    
    print(f"Found {len(folder_pairs)} folders to verify")
    
    all_verified = True
    summary_stats = defaultdict(int)
    
    for recorded_folder, cropped_folder in folder_pairs:
        is_good = verify_single_folder(recorded_folder, cropped_folder)
        if is_good:
            summary_stats['success'] += 1
        else:
            summary_stats['failed'] += 1
            all_verified = False
    
    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    print(f"Total folders checked: {len(folder_pairs)}")
    print(f"✅ Successfully verified: {summary_stats['success']}")
    print(f"❌ Failed verification: {summary_stats['failed']}")
    
    if all_verified:
        print("\n🎉 ALL FOLDERS VERIFIED SUCCESSFULLY!")
    else:
        print("\n⚠️  SOME FOLDERS HAVE ISSUES - CHECK DETAILS ABOVE")
    
    return all_verified

def main():
    parser = argparse.ArgumentParser(description="Verify Array2F1tracks cropping results")
    parser.add_argument("--folder", "-f", type=str, help="Specific folder to verify (path to _Recorded folder)")
    parser.add_argument("--data-dir", "-d", type=str, default="/home/matthias/Videos/", 
                       help="Data directory containing folders (default: /home/matthias/Videos/)")
    
    args = parser.parse_args()
    
    if args.folder:
        # Verify specific folder
        recorded_folder = Path(args.folder)
        if not recorded_folder.name.endswith("_Recorded"):
            print("❌ ERROR: Folder should end with '_Recorded'")
            sys.exit(1)
        
        cropped_name = recorded_folder.name.replace("_Recorded", "_Cropped")
        cropped_folder = recorded_folder.parent / cropped_name
        
        success = verify_single_folder(recorded_folder, cropped_folder)
        sys.exit(0 if success else 1)
    else:
        # Verify all folders
        success = verify_all_folders(args.data_dir)
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()