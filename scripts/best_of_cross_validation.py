#!/usr/bin/env python3
"""
Best-of Cross Validation Script for FENCAST Models

This script performs K-fold cross validation on the top N trials from an Optuna study
to find which hyperparameter set generalizes best across folds.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

from fencast.utils.paths import load_config, PROJECT_ROOT
from fencast.utils.tools import setup_logger
from fencast.utils.experiment_management import load_top_n_trials_from_study
from fencast.training import KFoldCrossValidator, validate_training_parameters
from fencast.utils.parser import get_parser


def main():
    parser = get_parser(['config', 'study_name', 'k_folds', 'top_n'],
                        description="Run K-Fold Cross Validation on top N trials from an Optuna study")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("best_of_cv")
    logger.info(f"Starting best-of cross validation: {args.k_folds}-fold CV on top {args.top_n} trials")
    
    # Load configuration
    config = load_config(args.config)
    setup_name = config.get('setup_name', 'default_setup')
    results_parent_dir = PROJECT_ROOT / "results" / setup_name
    
    # Merge fixed parameters from config (same as run_training.py)
    tuning_config = config.get('tuning', {})
    model_tuning_config = config.get('cnn_tuning', {})
    combined_config = {**tuning_config, **model_tuning_config}
    
    # Load top N trials from the study
    top_trials = load_top_n_trials_from_study(
        results_parent_dir=results_parent_dir,
        n=args.top_n,
        study_name=args.study_name
    )
    
    if not top_trials:
        logger.error("No trials found in study.")
        return
    
    # Get study_dir from the first trial (all trials share the same study_dir)
    study_dir = top_trials[0][3]
    
    # Store results for comparison
    all_cv_results = []
    
    # Run cross validation for each trial
    for trial_number, params, trial_value, _ in top_trials:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running CV for Trial {trial_number} (tuning loss: {trial_value:.6f})")
        logger.info(f"{'='*60}")
        
        # Add fixed parameters from config that are not in the study params
        for param_name, param_config in combined_config.items():
            if param_name in ['trials', 'epochs', 'early_stopping_patience']:
                continue
            if isinstance(param_config, (int, float, str)) and param_name not in params:
                params[param_name] = param_config
                logger.info(f"Added fixed parameter from config: {param_name}={param_config}")
        
        # Validate and clean parameters
        final_params = validate_training_parameters(params)
        
        logger.info(f"Parameters:\n{json.dumps(final_params, indent=4)}")
        
        # Set results directory based on trial number
        results_dir = study_dir / f"cross_validation_trial_{trial_number}"
        results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Results will be saved to: {results_dir}")
        
        # Create cross validator and run
        cv = KFoldCrossValidator(config, final_params, args.k_folds, logger)
        cv_results = cv.run_cross_validation(results_dir)
        
        # Store results with trial info
        cv_results['trial_number'] = trial_number
        cv_results['tuning_loss'] = trial_value
        cv_results['params'] = final_params
        all_cv_results.append(cv_results)
        
        if 'cv_mean_loss' in cv_results:
            logger.info(f"Trial {trial_number} CV Result: {cv_results['cv_mean_loss']:.6f} ± {cv_results['cv_std_loss']:.6f}")
    
    # Print final comparison summary
    print(f"\n{'='*70}")
    print(f"BEST-OF CROSS VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"Config: {args.config}")
    print(f"K-Folds: {args.k_folds}")
    print(f"Top N Trials: {args.top_n}")
    print(f"{'='*70}")
    print(f"{'Trial':<10} {'Tuning Loss':<15} {'CV Mean':<15} {'CV Std':<15}")
    print(f"{'-'*70}")
    
    # Sort by CV mean loss
    valid_results = [r for r in all_cv_results if 'cv_mean_loss' in r]
    sorted_results = sorted(valid_results, key=lambda r: r['cv_mean_loss'])
    
    for result in sorted_results:
        print(f"{result['trial_number']:<10} {result['tuning_loss']:<15.6f} {result['cv_mean_loss']:<15.6f} {result['cv_std_loss']:<15.6f}")
    
    print(f"{'='*70}")
    
    if sorted_results:
        best = sorted_results[0]
        print(f"\nBest CV Trial: {best['trial_number']}")
        print(f"  CV Mean Loss: {best['cv_mean_loss']:.6f} ± {best['cv_std_loss']:.6f}")
        print(f"  Tuning Loss:  {best['tuning_loss']:.6f}")
        
        # Save overall summary
        summary_path = study_dir / "best_of_cv_summary.json"
        summary = {
            'config': args.config,
            'k_folds': args.k_folds,
            'top_n': args.top_n,
            'best_trial': best['trial_number'],
            'best_cv_mean_loss': best['cv_mean_loss'],
            'best_cv_std_loss': best['cv_std_loss'],
            'all_results': [
                {
                    'trial_number': r['trial_number'],
                    'tuning_loss': r['tuning_loss'],
                    'cv_mean_loss': r['cv_mean_loss'],
                    'cv_std_loss': r['cv_std_loss']
                }
                for r in sorted_results
            ]
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"\nSummary saved to: {summary_path}")
    else:
        print("All cross validation runs failed. Check logs for details.")


if __name__ == "__main__":
    main()
