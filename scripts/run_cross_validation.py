#!/usr/bin/env python3
"""
K-Fold Cross Validation Script for FENCAST Models

This script performs K-fold cross validation on trained models to get robust performance estimates.
It uses the refactored training components for consistent behavior across different training modes.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

from fencast.utils.paths import load_config, PROJECT_ROOT
from fencast.utils.tools import setup_logger
from fencast.utils.experiment_management import load_best_params_from_study
from fencast.training import KFoldCrossValidator
from fencast.utils.parser import get_parser


def main():
    parser = get_parser(['config', 'study_name', 'k_folds', 'results_dir'],
                        description="Run K-Fold Cross Validation for FENCAST models")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("cross_validation")
    logger.info(f"Starting {args.k_folds}-fold cross validation for CNN model")
    
    # Load configuration
    config = load_config(args.config)
    setup_name = config.get('setup_name', 'default_setup')
    results_parent_dir = PROJECT_ROOT / "results" / setup_name
    
    # Load hyperparameters from study
    params, study_dir = load_best_params_from_study(
        results_parent_dir=results_parent_dir,
        study_name=args.study_name
    )
    
    # Determine results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = study_dir / f"cross_validation_{timestamp}"
    
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to: {results_dir}")
    
    # Create cross validator and run
    cv = KFoldCrossValidator(config, params, args.k_folds, logger)
    cv_results = cv.run_cross_validation(results_dir)
    
    # Print final summary
    if 'cv_mean_loss' in cv_results:
        print(f"\\n{'='*50}")
        print(f"K-FOLD CROSS VALIDATION RESULTS")
        print(f"{'='*50}")
        print(f"Model: CNN")
        print(f"Folds: {args.k_folds}")
        print(f"Mean ± Std: {cv_results['cv_mean_loss']:.6f} ± {cv_results['cv_std_loss']:.6f}")
        print(f"Min / Max: {cv_results['cv_min_loss']:.6f} / {cv_results['cv_max_loss']:.6f}")
        print(f"Success rate: {cv_results['successful_folds']}/{args.k_folds}")
        print(f"Results saved to: {results_dir}")
    else:
        print("Cross validation failed. Check logs for details.")


if __name__ == "__main__":
    main()