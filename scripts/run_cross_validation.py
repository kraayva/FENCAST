#!/usr/bin/env python3
"""
K-Fold Cross Validation Script for FENCAST Models

This script performs K-fold cross validation on trained models to get robust performance estimates.
It uses the refactored training components for consistent behavior across different training modes.
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import optuna

from fencast.utils.paths import load_config, PROJECT_ROOT
from fencast.utils.tools import setup_logger, get_latest_study_dir
from fencast.training import ModelTrainer, validate_training_parameters
from fencast.dataset import FencastDataset


class KFoldCrossValidator:
    """
    K-Fold Cross Validation for FENCAST models.
    """
    
    def __init__(self, config: Dict[str, Any], model_type: str, params: Dict[str, Any], 
                 k_folds: int = 5, logger=None):
        """
        Initialize the K-Fold cross validator.
        
        Args:
            config: Project configuration dictionary
            model_type: Model architecture ('ffnn' or 'cnn')
            params: Training hyperparameters
            k_folds: Number of folds for cross validation
            logger: Logger instance
        """
        self.config = config
        self.model_type = model_type
        self.params = validate_training_parameters(params)
        self.k_folds = k_folds
        self.logger = logger
        self.results = []
        
        if self.logger:
            self.logger.info(f"Initialized {k_folds}-fold cross validator for {model_type} model")
    
    def create_fold_splits(self) -> List[Tuple[List[int], List[int]]]:
        """
        Create K-fold splits based on years to ensure proper temporal separation.
        
        Returns:
            List of (train_years, val_years) tuples for each fold
        """
        # Get available training years (exclude test years)
        test_years = set(self.config['split_years']['test'])
        
        all_available_years = [y for y in np.arange(int(self.config['time_start'][:4]), 
                                                    int(self.config['time_end'][:4]) + 1)
                                                    if y not in test_years]
        all_available_years.sort()
        
        if self.logger:
            self.logger.info(f"Available years for cross validation: {len(all_available_years)} years ({min(all_available_years)}-{max(all_available_years)})")
            self.logger.info(f"Excluded test years: {sorted(test_years)}")
        
        # Create K-fold splits
        fold_splits = []
        remainder_years = len(all_available_years) % self.k_folds
        # Remove remainder years from the end to ensure equal fold sizes
        usable_years = all_available_years[:-remainder_years] if remainder_years > 0 else all_available_years
        years_per_fold = len(usable_years) // self.k_folds
        
        if self.logger and remainder_years > 0:
            excluded_years = all_available_years[-remainder_years:]
            self.logger.info(f"Excluding {remainder_years} years for equal fold sizes: {excluded_years}")
        
        # Generate folds
        for fold in range(self.k_folds):
            # Calculate validation years for this fold
            val_start_idx = fold * years_per_fold
            val_end_idx = (fold + 1) * years_per_fold
            
            val_years = usable_years[val_start_idx:val_end_idx]
            train_years = [year for year in usable_years if year not in val_years]
            
            fold_splits.append((train_years, val_years))
            
            if self.logger:
                self.logger.info(f"Fold {fold + 1}: Train years={len(train_years)}, Val years={len(val_years)}")
        
        return fold_splits
    
    def create_fold_config(self, fold_idx: int, train_years: List[int], val_years: List[int]) -> Dict[str, Any]:
        """
        Create a modified config for a specific fold.
        
        Args:
            fold_idx: Fold index
            train_years: Training years for this fold
            val_years: Validation years for this fold
            
        Returns:
            Modified config dictionary
        """
        fold_config = self.config.copy()
        fold_config['split_years'] = {
            'train': train_years,  # This will be handled by custom dataset creation
            'validation': val_years,
            'test': self.config['split_years']['test']  # Keep original test years
        }
        
        # Add fold information
        fold_config['current_fold'] = fold_idx
        fold_config['total_folds'] = self.k_folds
        
        return fold_config
    
    def train_fold(self, fold_idx: int, train_years: List[int], val_years: List[int], 
                   save_dir: Path) -> Dict[str, Any]:
        """
        Train a model for one fold.
        
        Args:
            fold_idx: Fold index (0-based)
            train_years: Training years for this fold
            val_years: Validation years for this fold
            save_dir: Directory to save fold results
            
        Returns:
            Fold training results
        """
        if self.logger:
            self.logger.info(f"\\n=== Training Fold {fold_idx + 1}/{self.k_folds} ===")
            self.logger.info(f"Train years: {sorted(train_years)}")
            self.logger.info(f"Val years: {sorted(val_years)}")
        
        # Create fold-specific config
        fold_config = self.create_fold_config(fold_idx, train_years, val_years)
        
        # Create trainer for this fold
        trainer = ModelTrainer(fold_config, self.model_type, self.params, self.logger)
        
        # Create custom data loaders with specific years for this fold
        train_loader, val_loader = trainer.create_custom_data_loaders(train_years, val_years)
        
        # Training configuration
        epochs = self.config.get('cross_validation', {}).get('epochs', 
                                self.config.get('tuning', {}).get('epochs', 30))
        
        # Create save path for this fold
        fold_save_path = save_dir / f"fold_{fold_idx + 1}_best_model.pth"
        
        # Train the model
        fold_results = trainer.train_model(train_loader, val_loader, epochs, fold_save_path)
        
        # Add fold-specific information
        fold_results.update({
            'fold_idx': fold_idx,
            'train_years': train_years,
            'val_years': val_years,
            'epochs_trained': epochs
        })
        
        if self.logger:
            self.logger.info(f"Fold {fold_idx + 1} completed. Best val loss: {fold_results['best_val_loss']:.6f}")
        
        return fold_results
    
    def run_cross_validation(self, results_dir: Path) -> Dict[str, Any]:
        """
        Run complete K-fold cross validation.

        Args:
            results_dir: Directory to save results
            
        Returns:
            Cross validation summary results
        """
        if self.logger:
            self.logger.info(f"Starting {self.k_folds}-fold cross validation")
        
        # Create fold splits
        fold_splits = self.create_fold_splits()
        
        # Create results directory
        cv_dir = results_dir / "cross_validation"
        cv_dir.mkdir(parents=True, exist_ok=True)
        
        # Train each fold
        fold_results = []
        for fold_idx, (train_years, val_years) in enumerate(fold_splits):
            try:
                fold_result = self.train_fold(fold_idx, train_years, val_years, cv_dir)
                fold_results.append(fold_result)
                
                # Save individual fold results
                fold_result_path = cv_dir / f"fold_{fold_idx + 1}_results.json"
                with open(fold_result_path, 'w') as f:
                    # Convert non-serializable items
                    serializable_result = {k: v for k, v in fold_result.items() 
                                         if k not in ['model', 'model_args']}
                    json.dump(serializable_result, f, indent=4)
                    
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error in fold {fold_idx + 1}: {e}")
                fold_results.append({
                    'fold_idx': fold_idx,
                    'error': str(e),
                    'best_val_loss': float('inf')
                })
        
        # Calculate cross validation statistics
        val_losses = [result['best_val_loss'] for result in fold_results 
                     if 'best_val_loss' in result and result['best_val_loss'] != float('inf')]
        
        if val_losses:
            cv_summary = {
                'k_folds': self.k_folds,
                'model_type': self.model_type,
                'params': self.params,
                'fold_results': fold_results,
                'cv_mean_loss': np.mean(val_losses),
                'cv_std_loss': np.std(val_losses),
                'cv_min_loss': np.min(val_losses),
                'cv_max_loss': np.max(val_losses),
                'successful_folds': len(val_losses),
                'failed_folds': self.k_folds - len(val_losses)
            }
            
            if self.logger:
                self.logger.info(f"\\n=== Cross Validation Summary ===")
                self.logger.info(f"Mean validation loss: {cv_summary['cv_mean_loss']:.6f} ± {cv_summary['cv_std_loss']:.6f}")
                self.logger.info(f"Min/Max validation loss: {cv_summary['cv_min_loss']:.6f} / {cv_summary['cv_max_loss']:.6f}")
                self.logger.info(f"Successful folds: {cv_summary['successful_folds']}/{self.k_folds}")
        else:
            cv_summary = {
                'error': 'All folds failed',
                'fold_results': fold_results
            }
            if self.logger:
                self.logger.error("All cross validation folds failed!")
        
        # Save summary results
        summary_path = cv_dir / "cv_summary.json"
        with open(summary_path, 'w') as f:
            # Remove non-serializable items for JSON
            serializable_summary = {k: v for k, v in cv_summary.items() 
                                  if k not in ['fold_results']}
            serializable_summary['fold_summaries'] = [
                {k: v for k, v in result.items() if k not in ['model', 'model_args', 'train_losses', 'val_losses']}
                for result in fold_results
            ]
            json.dump(serializable_summary, f, indent=4)
        
        if self.logger:
            self.logger.info(f"Cross validation results saved to: {cv_dir}")
        
        return cv_summary


def main():
    parser = argparse.ArgumentParser(
        description="Run K-Fold Cross Validation for FENCAST models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--config', '-c', default='datapp_de',
                       help='Configuration file name')
    parser.add_argument('--model-type', '-m', required=True, choices=['ffnn', 'cnn'],
                       help='Model architecture to validate')
    parser.add_argument('--study-name', '-s', default='latest',
                       help='Study name to load hyperparameters from')
    parser.add_argument('--k-folds', '-k', type=int, default=5,
                       help='Number of folds for cross validation')
    parser.add_argument('--results-dir', '-r', 
                       help='Custom results directory (default: creates CV subdirectory in study dir)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("cross_validation")
    logger.info(f"Starting {args.k_folds}-fold cross validation for {args.model_type} model")
    
    # Load configuration
    config = load_config(args.config)
    setup_name = config.get('setup_name', 'default_setup')
    results_parent_dir = PROJECT_ROOT / "results" / setup_name
    
    # Load hyperparameters from study
    try:
        if args.study_name == 'latest':
            study_dir = get_latest_study_dir(results_parent_dir, args.model_type)
        else:
            study_dir = results_parent_dir / args.study_name
            
        study_name = study_dir.name
        storage_name = f"sqlite:///{study_dir / study_name}.db"
        
        study = optuna.load_study(study_name=study_name, storage=storage_name)
        params = study.best_trial.params
        
        logger.info(f"Loaded best parameters from study: '{study_name}'")
        logger.info(f"Best trial value: {study.best_trial.value:.6f}")
        
    except Exception as e:
        logger.error(f"Failed to load study: {e}")
        raise
    
    # Determine results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = study_dir / f"cross_validation_{timestamp}"
    
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to: {results_dir}")
    
    # Create cross validator and run
    cv = KFoldCrossValidator(config, args.model_type, params, args.k_folds, logger)
    cv_results = cv.run_cross_validation(results_dir)
    
    # Print final summary
    if 'cv_mean_loss' in cv_results:
        print(f"\\n{'='*50}")
        print(f"K-FOLD CROSS VALIDATION RESULTS")
        print(f"{'='*50}")
        print(f"Model: {args.model_type}")
        print(f"Folds: {args.k_folds}")
        print(f"Mean ± Std: {cv_results['cv_mean_loss']:.6f} ± {cv_results['cv_std_loss']:.6f}")
        print(f"Min / Max: {cv_results['cv_min_loss']:.6f} / {cv_results['cv_max_loss']:.6f}")
        print(f"Success rate: {cv_results['successful_folds']}/{args.k_folds}")
        print(f"Results saved to: {results_dir}")
    else:
        print("Cross validation failed. Check logs for details.")


if __name__ == "__main__":
    main()