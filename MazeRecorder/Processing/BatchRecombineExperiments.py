#!/usr/bin/env python3
"""
Batch recombine all experiments from a YAML file.
This script reads experiment directories from a YAML file and runs recombination on each.
"""

import yaml
from pathlib import Path
import subprocess
import sys
from datetime import datetime
import argparse


def recombine_experiments_from_yaml(yaml_file, pixels_to_move=15, workers=4, use_temp=False, use_cuda=True, show_progress=True):
    """
    Recombine all experiments listed in a YAML file.
    
    Args:
        yaml_file: Path to YAML file containing experiment directories
        pixels_to_move: Number of pixels to transfer from Right to Left
        workers: Number of parallel workers
        use_temp: Copy videos to temp storage before processing
        use_cuda: Use CUDA hardware acceleration
        show_progress: Show progress bars
    """
    # Read YAML file
    yaml_path = Path(yaml_file)
    if not yaml_path.exists():
        print(f"Error: YAML file not found: {yaml_file}")
        return False
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    if 'directories' not in data:
        print("Error: YAML file must contain 'directories' key")
        return False
    
    directories = data['directories']
    print(f"Found {len(directories)} experiments to recombine")
    print(f"Parameters:")
    print(f"  Pixels to move: {pixels_to_move}")
    print(f"  Workers: {workers}")
    print(f"  Use temp storage: {use_temp}")
    print(f"  Use CUDA: {use_cuda}")
    print(f"  Show progress: {show_progress}")
    print()
    
    # Track results
    results = {
        'successful': [],
        'failed': [],
        'missing': []
    }
    
    # Get the RecombineVideos.py script path
    script_dir = Path(__file__).parent
    recombine_script = script_dir / "RecombineVideos.py"
    
    if not recombine_script.exists():
        print(f"Error: RecombineVideos.py not found at {recombine_script}")
        return False
    
    # Process each experiment
    start_time = datetime.now()
    
    for i, exp_dir in enumerate(directories, 1):
        exp_path = Path(exp_dir)
        
        print(f"\n{'='*80}")
        print(f"[{i}/{len(directories)}] {exp_path.name}")
        print(f"{'='*80}")
        
        # Check if experiment exists
        if not exp_path.exists():
            print(f"❌ Experiment directory not found: {exp_path}")
            results['missing'].append(str(exp_path))
            continue
        
        # Build output directory name
        output_dir = exp_path.parent / f"{exp_path.name}_Recombined"
        
        # Build command
        cmd = [
            "python",
            str(recombine_script),
            "--experiment", str(exp_path),
            "--pixels", str(pixels_to_move),
            "--output", str(output_dir),
            "--workers", str(workers)
        ]
        
        if use_temp:
            cmd.append("--use-temp")
        if not use_cuda:
            cmd.append("--no-cuda")
        if not show_progress:
            cmd.append("--no-progress")
        
        print(f"Command: {' '.join(cmd)}")
        print()
        
        try:
            # Run recombination
            result = subprocess.run(cmd, check=True)
            
            if result.returncode == 0:
                print(f"\n✅ Successfully recombined: {exp_path.name}")
                results['successful'].append(str(exp_path))
            else:
                print(f"\n❌ Failed to recombine: {exp_path.name}")
                results['failed'].append(str(exp_path))
                
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Error recombining {exp_path.name}: {e}")
            results['failed'].append(str(exp_path))
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            break
    
    # Print summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'='*80}")
    print("BATCH RECOMBINATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total experiments: {len(directories)}")
    print(f"Successful: {len(results['successful'])}")
    print(f"Failed: {len(results['failed'])}")
    print(f"Missing: {len(results['missing'])}")
    print(f"Duration: {duration}")
    print()
    
    if results['successful']:
        print("✅ Successfully recombined:")
        for exp in results['successful']:
            print(f"  - {Path(exp).name}")
        print()
    
    if results['failed']:
        print("❌ Failed:")
        for exp in results['failed']:
            print(f"  - {Path(exp).name}")
        print()
    
    if results['missing']:
        print("⚠️  Missing:")
        for exp in results['missing']:
            print(f"  - {Path(exp).name}")
        print()
    
    return len(results['failed']) == 0 and len(results['missing']) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Batch recombine experiments from a YAML file"
    )
    parser.add_argument(
        "yaml_file",
        type=str,
        help="Path to YAML file containing experiment directories"
    )
    parser.add_argument(
        "--pixels",
        type=int,
        default=15,
        help="Number of pixels to move from Right to Left (default: 15)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--use-temp",
        action="store_true",
        help="Copy videos to temp storage before processing (faster for network storage)"
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Disable CUDA hardware acceleration"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars (more reliable, less output)"
    )
    
    args = parser.parse_args()
    
    success = recombine_experiments_from_yaml(
        args.yaml_file,
        pixels_to_move=args.pixels,
        workers=args.workers,
        use_temp=args.use_temp,
        use_cuda=not args.no_cuda,
        show_progress=not args.no_progress
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
