# src/fencast/training.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from fencast.dataset import FencastDataset
from fencast.models import DynamicCNN


class ModelTrainer:
    """
    A flexible model trainer that can be used for regular training, cross-validation, and hyperparameter tuning.
    """
    
    def __init__(self, config: Dict[str, Any], params: Dict[str, Any], logger=None):
        """
        Initialize the trainer.
        
        Args:
            config: Project configuration dictionary
            params: Training hyperparameters
            logger: Logger instance (optional)
        """
        self.config = config
        self.params = params
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Validate parameters
        self._validate_parameters()
        
    def _validate_parameters(self):
        """Validate essential training parameters."""
        required_params = ['lr', 'activation_name', 'dropout_rate']
        missing_params = [p for p in required_params if p not in self.params]
        if missing_params:
            error_msg = f"Missing required parameters: {missing_params}"
            if self.logger:
                self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        if self.logger:
            relevant_params = {k: v for k, v in self.params.items() 
                             if k in required_params + ['optimizer_name', 'weight_decay', 'scheduler_name']}
            self.logger.info(f"Training parameters validated: {json.dumps(relevant_params, indent=2)}")
    
    def create_model(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Create and return the model along with its arguments.
        
        Returns:
            Tuple of (model, model_args)
        """
        model_args = {
            'config': self.config,
            'params': self.params
        }
        model = DynamicCNN(**model_args).to(self.device)
            
        return model, model_args
    
    def create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """
        Create and return the optimizer.
        
        Args:
            model: The model to optimize
            
        Returns:
            Optimizer instance
        """
        optimizer_name = self.params.get('optimizer_name', 'Adam')
        
        if optimizer_name == 'AdamW':
            weight_decay = self.params.get('weight_decay', 0.0)
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.params['lr'], weight_decay=weight_decay)
            if self.logger:
                self.logger.info(f"Using optimizer: {optimizer_name} with weight_decay={weight_decay:.2e}")
        else:  # Default to Adam
            optimizer = torch.optim.Adam(model.parameters(), lr=self.params['lr'])
            if self.logger:
                self.logger.info(f"Using optimizer: {optimizer_name}")
                
        return optimizer
    
    def create_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Create and return the learning rate scheduler.
        
        Args:
            optimizer: The optimizer to schedule
            
        Returns:
            Scheduler instance or None
        """
        scheduler = None
        if self.params.get('scheduler_name') == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.params.get('scheduler_factor', 0.2),
                patience=self.params.get('scheduler_patience', 3)
            )
            if self.logger:
                self.logger.info(f"Using ReduceLROnPlateau scheduler with factor={self.params.get('scheduler_factor', 0.2)}, patience={self.params.get('scheduler_patience', 3)}")
        
        return scheduler
    
    def create_data_loaders(self, train_mode: str = 'train', val_mode: str = 'validation', 
                           batch_size: Optional[int] = None) -> Tuple[DataLoader, DataLoader]:
        """
        Create and return train and validation data loaders.
        
        Args:
            train_mode: Training dataset mode
            val_mode: Validation dataset mode
            batch_size: Batch size (uses config default if None)
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        if batch_size is None:
            batch_size = self.config.get('model', {}).get('batch_sizes', {}).get('training', 64)
            
        train_dataset = FencastDataset(config=self.config, mode=train_mode)
        val_dataset = FencastDataset(config=self.config, mode=val_mode)
        
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def create_custom_data_loaders(self, train_years: List[int], val_years: List[int], 
                                batch_size: Optional[int] = None) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Create data loaders with custom year filtering. If val_years is empty, the validation loader will be None.

        Args:
            train_years: List of years for training data
            val_years: List of years for validation data
            batch_size: Batch size (uses params default if None)

        Returns:
            Tuple of (train_loader, val_loader or None)
        """
        if batch_size is None:
            batch_size = self.params.get('batch_size', 64) # Use params for batch size
            
        # 1. Create the training dataset and loader
        train_dataset = FencastDataset(config=self.config, mode='train', 
                                    apply_normalization=False, custom_years=train_years)
        
        val_loader = None
        
        # 2. Conditionally create the validation dataset and loader
        if val_years: # This is True only if the list is not empty
            val_dataset = FencastDataset(config=self.config, mode='validation', 
                                        apply_normalization=False, custom_years=val_years)
            
            # Apply normalization to both
            self._normalize_cnn_datasets(train_dataset, val_dataset)
                
            val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
        else:
            # If no validation set, just normalize the training set
            self._normalize_cnn_datasets(train_dataset) # Pass only one argument

        # 3. Create the training loader after normalization has been applied
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        
        return train_loader, val_loader
    
    def _normalize_cnn_datasets(self, train_dataset, val_dataset = None):
        """Normalize CNN datasets using training data statistics."""
        
        # Calculate statistics from training data
        mean = np.mean(train_dataset.X, axis=(0, 2, 3), keepdims=True)
        std = np.std(train_dataset.X, axis=(0, 2, 3), keepdims=True)
        std[std == 0] = 1e-7
        
        # Apply normalization
        train_dataset.X = (train_dataset.X - mean) / std
        if val_dataset:
            val_dataset.X = (val_dataset.X - mean) / std
    
    def forward_pass(self, model: nn.Module, batch) -> torch.Tensor:
        """
        Perform a forward pass through the model.
        
        Args:
            model: The model to use
            batch: The input batch
            
        Returns:
            Model outputs
        """
        spatial_features, temporal_features, labels = batch
        spatial_features = spatial_features.to(self.device)
        temporal_features = temporal_features.to(self.device)
        outputs = model(spatial_features, temporal_features)
            
        return outputs
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
        """
        Train the model for one epoch.
        
        Args:
            model: The model to train
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            
        Returns:
            Average training loss for the epoch
        """
        model.train()
        train_losses = []
        
        for batch in train_loader:
            # Get labels for loss calculation
            labels = batch[2].to(self.device)  # spatial, temporal, labels
            
            # Forward pass
            outputs = self.forward_pass(model, batch)
            loss = criterion(outputs, labels)
            train_losses.append(loss.item())
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            max_norm = self.config.get('training', {}).get('gradient_clip_max_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            
            optimizer.step()
            
        return np.mean(train_losses)
    
    def validate_epoch(self, model: nn.Module, val_loader: DataLoader, criterion: nn.Module) -> float:
        """
        Validate the model for one epoch.
        
        Args:
            model: The model to validate
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Average validation loss for the epoch
        """
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Get labels for loss calculation
                labels = batch[2].to(self.device)  # spatial, temporal, labels
                
                # Forward pass
                outputs = self.forward_pass(model, batch)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
                
        return np.mean(val_losses)
    
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int,
                   save_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            save_path: Path to save the best model (optional)
            
        Returns:
            Training results dictionary
        """
        # Create model, optimizer, scheduler, and criterion
        model, model_args = self.create_model()
        optimizer = self.create_optimizer(model)
        scheduler = self.create_scheduler(optimizer)
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        if self.logger:
            self.logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            # Always run the training step for the epoch
            avg_train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
            train_losses.append(avg_train_loss)
            
            # Check if the validation dataloader has any data before using it
            if val_loader and len(val_loader.dataset) > 0:
                # --- SCENARIO 1: Standard run with validation ---
                avg_val_loss = self.validate_epoch(model, val_loader, criterion)
                val_losses.append(avg_val_loss)
                
                # Update scheduler if it exists
                if scheduler:
                    scheduler.step(avg_val_loss)
                
                # Save the best model based on validation loss
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    if save_path:
                        # Your existing dictionary for saving the model checkpoint
                        checkpoint = {
                            'model_args': model_args,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch,
                            'train_loss': avg_train_loss,
                            'val_loss': avg_val_loss,
                            'config': self.config,
                            'params': self.params
                        }
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(checkpoint, save_path)
                    
                    if self.logger:
                        self.logger.info(f"Epoch [{epoch+1}/{epochs}], Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f} (saved)")
                else:
                    if self.logger:
                        self.logger.info(f"Epoch [{epoch+1}/{epochs}], Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}")
            
            else:
                # --- SCENARIO 2: Final run with no validation ---
                best_val_loss = avg_train_loss  # Track the training loss as the main metric
                if self.logger:
                    self.logger.info(f"Epoch [{epoch+1}/{epochs}], Train: {avg_train_loss:.6f} (No validation)")

        # If it was a final run, save the model from the last epoch
        if not (val_loader and len(val_loader.dataset) > 0) and save_path:
            if self.logger:
                self.logger.info(f"No validation set. Saving final model from last epoch to {save_path}")
            
            # Your existing dictionary for saving the model checkpoint
            final_checkpoint = {
                'model_args': model_args,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epochs - 1, # The last completed epoch
                'train_loss': avg_train_loss, # Final training loss
                'val_loss': float('nan'), # No validation loss available
                'config': self.config,
                'params': self.params
            }
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(final_checkpoint, save_path)
        
        results = {
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'model': model,
            'model_args': model_args
        }
        
        if self.logger:
            self.logger.info(f"Training completed. Best validation loss: {best_val_loss:.6f}")
        
        return results


def validate_training_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean training parameters with defaults and legacy support.
    
    Args:
        params: Raw parameters from study or config
        
    Returns:
        Cleaned and validated parameters
    """
    final_params = dict(params)  # Create a copy
    
    # Handle missing required parameters with defaults
    if 'lr' not in final_params:
        final_params['lr'] = 1e-3
    if 'activation_name' not in final_params:
        final_params['activation_name'] = 'ELU'
    
    # Handle legacy parameter names
    if 'learning_rate' in final_params and 'lr' not in final_params:
        final_params['lr'] = final_params['learning_rate']
    if 'activation' in final_params and 'activation_name' not in final_params:
        final_params['activation_name'] = final_params['activation']
    
    # Handle CNN-specific parameter mapping
    if 'filters' in final_params and 'out_channels' not in final_params:
        # Convert single filter value to list for CNN architecture
        n_conv_layers = final_params.get('n_conv_layers', 3)
        base_filters = final_params['filters']
        # Create increasing filter sizes: [base, base*2, base*4, ...]
        final_params['out_channels'] = [base_filters * (2**i) for i in range(n_conv_layers)]
    
    # Ensure CNN has required parameters
    if 'out_channels' in final_params or 'filters' in final_params:
        if 'kernel_size' not in final_params:
            final_params['kernel_size'] = 3
        if 'n_conv_layers' not in final_params:
            final_params['n_conv_layers'] = 3
    
    return final_params


class KFoldCrossValidator:
    """
    K-Fold Cross Validation for FENCAST models.
    """
    
    def __init__(self, config: Dict[str, Any], params: Dict[str, Any], 
                 k_folds: int = 5, logger=None):
        """
        Initialize the K-Fold cross validator.
        
        Args:
            config: Project configuration dictionary
            params: Training hyperparameters
            k_folds: Number of folds for cross validation
            logger: Logger instance
        """
        self.config = config
        self.params = validate_training_parameters(params)
        self.k_folds = k_folds
        self.logger = logger
        self.results = []
        
        if self.logger:
            self.logger.info(f"Initialized {k_folds}-fold cross validator for CNN model")
    
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
        trainer = ModelTrainer(fold_config, self.params, self.logger)
        
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