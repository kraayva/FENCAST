#!/usr/bin/env python3
"""
Quick demo script to test the K-Fold Cross Validation functionality.
This runs a minimal example with 2 folds and fewer epochs for testing.
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Run a quick cross validation demo."""
    
    # Path to the cross validation script
    cv_script = Path(__file__).parent / "run_cross_validation.py"
    
    print("="*60)
    print("FENCAST K-Fold Cross Validation Demo")
    print("="*60)
    print(f"Testing with 2 folds and CNN model...")
    print(f"Using latest study for hyperparameters")
    print()
    
    # Command to run
    cmd = [
        sys.executable, str(cv_script),
        "--model-type", "cnn",
        "--k-folds", "2",
        "--study-name", "latest"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print()
    
    try:
        # Run the cross validation
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        print("\n" + "="*60)
        print("Cross validation demo completed successfully!")
        print("="*60)
        
    except subprocess.CalledProcessError as e:
        print("ERROR: Cross validation failed!")
        print(f"Return code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        sys.exit(1)
    
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()