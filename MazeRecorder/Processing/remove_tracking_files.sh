#!/bin/bash

# Script to remove .slp and .h5 tracking files from directories
# Usage: ./remove_tracking_files.sh [OPTIONS] [DIRECTORY_PATTERNS...]

# Check for flags
DRY_RUN=false
INTERACTIVE=false
RECURSIVE=true
REMOVE_SLP=true
REMOVE_H5=true

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] [DIRECTORY_PATTERNS...]"
    echo ""
    echo "Remove .slp and .h5 tracking files from directories"
    echo ""
    echo "OPTIONS:"
    echo "  --dry-run, -n           Show what would be deleted without actually deleting"
    echo "  --interactive, -i       Ask for confirmation before deleting each file"
    echo "  --no-recursive         Only search in specified directories, not subdirectories"
    echo "  --slp-only             Only remove .slp files"
    echo "  --h5-only              Only remove .h5 files"
    echo "  --help, -h             Show this help message"
    echo ""
    echo "DIRECTORY_PATTERNS:"
    echo "  Optional patterns to filter directories. If not provided, will process all"
    echo "  directories under the default data folder."
    echo ""
    echo "Examples:"
    echo "  $0 --dry-run                    # Show what would be deleted"
    echo "  $0 --interactive F1_New         # Interactive mode for directories matching 'F1_New'"
    echo "  $0 --slp-only 241008            # Only remove .slp files from directories matching '241008'"
    echo "  $0 --no-recursive /path/to/dir  # Only process specified directory, no subdirs"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run|-n)
            DRY_RUN=true
            print_info "DRY RUN MODE - No files will be actually deleted"
            shift
            ;;
        --interactive|-i)
            INTERACTIVE=true
            print_info "INTERACTIVE MODE - Will ask for confirmation before deleting each file"
            shift
            ;;
        --no-recursive)
            RECURSIVE=false
            print_info "NON-RECURSIVE MODE - Will not search subdirectories"
            shift
            ;;
        --slp-only)
            REMOVE_H5=false
            print_info "SLP ONLY MODE - Will only remove .slp files"
            shift
            ;;
        --h5-only)
            REMOVE_SLP=false
            print_info "H5 ONLY MODE - Will only remove .h5 files"
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            break  # End of flags, start of directory arguments
            ;;
    esac
done

# Set default data folder (similar to your tracking script)
default_datafolder="/mnt/upramdya_data/MD/F1_Tracks/Videos"

# If no arguments provided, use default folder
if [ $# -eq 0 ]; then
    print_info "No directory patterns specified, using default data folder: $default_datafolder"
    if [ ! -d "$default_datafolder" ]; then
        print_error "Default data folder does not exist: $default_datafolder"
        print_info "Please specify a valid directory pattern or check the default path"
        exit 1
    fi
    search_paths=("$default_datafolder")
else
    # Use provided arguments as search patterns
    search_paths=("$@")
fi

print_info "Search paths/patterns: ${search_paths[*]}"

# Collect directories to process
directories_to_process=()

for search_path in "${search_paths[@]}"; do
    if [ -d "$search_path" ]; then
        # It's a direct directory path
        directories_to_process+=("$search_path")
        if [ "$RECURSIVE" = true ]; then
            # Add all subdirectories
            while IFS= read -r -d '' dir; do
                directories_to_process+=("$dir")
            done < <(find "$search_path" -type d -print0)
        fi
    else
        # It's a pattern, search in default folder
        if [ -d "$default_datafolder" ]; then
            if [ "$RECURSIVE" = true ]; then
                # Find directories matching the pattern recursively
                while IFS= read -r -d '' dir; do
                    if [[ "$dir" == *"$search_path"* ]]; then
                        directories_to_process+=("$dir")
                    fi
                done < <(find "$default_datafolder" -type d -print0)
            else
                # Find directories matching the pattern in default folder only
                for dir in "$default_datafolder"/*; do
                    if [ -d "$dir" ] && [[ "$dir" == *"$search_path"* ]]; then
                        directories_to_process+=("$dir")
                    fi
                done
            fi
        else
            print_warning "Default data folder does not exist, skipping pattern: $search_path"
        fi
    fi
done

# Remove duplicates and sort
mapfile -t directories_to_process < <(printf '%s\n' "${directories_to_process[@]}" | sort -u)

print_info "Found ${#directories_to_process[@]} directories to process"

if [ ${#directories_to_process[@]} -eq 0 ]; then
    print_warning "No directories found to process"
    exit 0
fi

# Collect files to remove
files_to_remove=()
total_slp_files=0
total_h5_files=0
total_size=0

print_info "Scanning for tracking files..."

for dir in "${directories_to_process[@]}"; do
    if [ ! -d "$dir" ]; then
        continue
    fi
    
    dir_slp_count=0
    dir_h5_count=0
    dir_size=0
    
    # Find .slp files
    if [ "$REMOVE_SLP" = true ]; then
        while IFS= read -r -d '' file; do
            if [ -f "$file" ]; then
                files_to_remove+=("$file")
                ((dir_slp_count++))
                ((total_slp_files++))
                file_size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo 0)
                ((dir_size += file_size))
                ((total_size += file_size))
            fi
        done < <(find "$dir" -maxdepth 1 -name "*.slp" -print0 2>/dev/null)
    fi
    
    # Find .h5 files (including .analysis.h5)
    if [ "$REMOVE_H5" = true ]; then
        while IFS= read -r -d '' file; do
            if [ -f "$file" ]; then
                files_to_remove+=("$file")
                ((dir_h5_count++))
                ((total_h5_files++))
                file_size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo 0)
                ((dir_size += file_size))
                ((total_size += file_size))
            fi
        done < <(find "$dir" -maxdepth 1 -name "*.h5" -print0 2>/dev/null)
    fi
    
    # Report directory summary if files found
    if [ $((dir_slp_count + dir_h5_count)) -gt 0 ]; then
        dir_size_mb=$((dir_size / 1024 / 1024))
        print_info "Directory: $(basename "$dir") - .slp: $dir_slp_count, .h5: $dir_h5_count, size: ${dir_size_mb}MB"
    fi
done

# Convert total size to human readable
total_size_mb=$((total_size / 1024 / 1024))
total_size_gb=$((total_size_mb / 1024))

echo ""
print_info "=== SUMMARY ==="
print_info "Total .slp files found: $total_slp_files"
print_info "Total .h5 files found: $total_h5_files"
print_info "Total files to remove: ${#files_to_remove[@]}"
if [ $total_size_gb -gt 0 ]; then
    print_info "Total size: ${total_size_gb}GB (${total_size_mb}MB)"
else
    print_info "Total size: ${total_size_mb}MB"
fi

if [ ${#files_to_remove[@]} -eq 0 ]; then
    print_success "No tracking files found to remove"
    exit 0
fi

# Show files to be removed in dry run mode
if [ "$DRY_RUN" = true ]; then
    echo ""
    print_warning "DRY RUN - Files that would be deleted:"
    for file in "${files_to_remove[@]}"; do
        file_size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo 0)
        file_size_mb=$((file_size / 1024 / 1024))
        echo "  - $file (${file_size_mb}MB)"
    done
    echo ""
    print_info "=== END DRY RUN ==="
    exit 0
fi

# Ask for confirmation unless in interactive mode (which asks per file)
if [ "$INTERACTIVE" = false ]; then
    echo ""
    print_warning "This will permanently delete ${#files_to_remove[@]} files totaling ${total_size_mb}MB"
    read -p "Are you sure you want to proceed? (y/N): " confirm
    if [[ ! $confirm =~ ^[Yy]$ ]]; then
        print_info "Operation cancelled"
        exit 0
    fi
fi

# Remove files
removed_count=0
failed_count=0

print_info "Starting file removal..."

for file in "${files_to_remove[@]}"; do
    file_name=$(basename "$file")
    dir_name=$(basename "$(dirname "$file")")
    
    # Interactive confirmation
    if [ "$INTERACTIVE" = true ]; then
        file_size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo 0)
        file_size_mb=$((file_size / 1024 / 1024))
        echo ""
        print_warning "Delete $dir_name/$file_name (${file_size_mb}MB)?"
        read -p "Delete this file? (y/N/q to quit): " confirm
        if [[ $confirm =~ ^[Qq]$ ]]; then
            print_info "Operation cancelled by user"
            break
        elif [[ ! $confirm =~ ^[Yy]$ ]]; then
            print_info "Skipping $file_name"
            continue
        fi
    fi
    
    # Attempt to remove the file
    if rm "$file" 2>/dev/null; then
        ((removed_count++))
        if [ "$INTERACTIVE" = true ]; then
            print_success "Deleted $file_name"
        elif [ $((removed_count % 10)) -eq 0 ]; then
            print_info "Removed $removed_count files..."
        fi
    else
        ((failed_count++))
        print_error "Failed to delete: $file"
    fi
done

echo ""
print_info "=== REMOVAL COMPLETE ==="
print_success "Successfully removed: $removed_count files"
if [ $failed_count -gt 0 ]; then
    print_error "Failed to remove: $failed_count files"
fi

if [ $removed_count -gt 0 ]; then
    print_success "File removal completed successfully"
else
    print_warning "No files were removed"
fi