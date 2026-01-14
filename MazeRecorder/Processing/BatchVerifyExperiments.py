#!/usr/bin/env python3
"""
Batch verify all experiments from a YAML file.
This script reads experiment directories from a YAML file and runs verification on each.
"""

import yaml
from pathlib import Path
import subprocess
import sys
from datetime import datetime

def verify_experiments_from_yaml(yaml_file, verification_script):
    """
    Verify all experiments listed in a YAML file.
    
    Args:
        yaml_file: Path to YAML file containing experiment directories
        verification_script: Path to VerifyProcessedExperiments.py script
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
    print(f"Found {len(directories)} experiments to verify\n")
    
    # Track results
    results = {
        'successful': [],
        'failed': [],
        'missing': []
    }
    
    # Verify each experiment
    for i, exp_dir in enumerate(directories, 1):
        exp_path = Path(exp_dir)
        
        print(f"[{i}/{len(directories)}] {exp_path.name}")
        print("="*60)
        
        if not exp_path.exists():
            print(f"  ⚠ Directory does not exist: {exp_path}")
            results['missing'].append(str(exp_path))
            print()
            continue
        
        # Run verification
        try:
            result = subprocess.run([
                sys.executable,
                str(verification_script),
                '--folder', str(exp_path)
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"  ✓ Verification complete")
                results['successful'].append(str(exp_path))
            else:
                print(f"  ✗ Verification failed")
                print(f"  Error: {result.stderr}")
                results['failed'].append(str(exp_path))
        
        except subprocess.TimeoutExpired:
            print(f"  ✗ Verification timed out")
            results['failed'].append(str(exp_path))
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results['failed'].append(str(exp_path))
        
        print()
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    print(f"Total experiments: {len(directories)}")
    print(f"Successful: {len(results['successful'])}")
    print(f"Failed: {len(results['failed'])}")
    print(f"Missing: {len(results['missing'])}")
    
    if results['failed']:
        print("\nFailed experiments:")
        for exp in results['failed']:
            print(f"  - {exp}")
    
    if results['missing']:
        print("\nMissing experiments:")
        for exp in results['missing']:
            print(f"  - {exp}")
    
    # Save results to a log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(f"batch_verification_{timestamp}.log")
    
    with open(log_file, 'w') as f:
        f.write("Batch Verification Results\n")
        f.write("="*60 + "\n\n")
        f.write(f"YAML file: {yaml_file}\n")
        f.write(f"Total experiments: {len(directories)}\n")
        f.write(f"Successful: {len(results['successful'])}\n")
        f.write(f"Failed: {len(results['failed'])}\n")
        f.write(f"Missing: {len(results['missing'])}\n\n")
        
        if results['successful']:
            f.write("Successful:\n")
            for exp in results['successful']:
                f.write(f"  {exp}\n")
            f.write("\n")
        
        if results['failed']:
            f.write("Failed:\n")
            for exp in results['failed']:
                f.write(f"  {exp}\n")
            f.write("\n")
        
        if results['missing']:
            f.write("Missing:\n")
            for exp in results['missing']:
                f.write(f"  {exp}\n")
    
    print(f"\nResults saved to: {log_file}")
    
    return len(results['failed']) == 0 and len(results['missing']) == 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Batch verify all experiments from a YAML file"
    )
    parser.add_argument(
        "yaml_file",
        type=str,
        help="Path to YAML file containing experiment directories"
    )
    parser.add_argument(
        "--verification-script",
        type=str,
        default=str(Path(__file__).parent / "VerifyProcessedExperiments.py"),
        help="Path to verification script (default: VerifyProcessedExperiments.py in same directory)"
    )
    
    args = parser.parse_args()
    
    success = verify_experiments_from_yaml(args.yaml_file, args.verification_script)
    sys.exit(0 if success else 1)
