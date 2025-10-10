# src/fencast/training.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from fencast.dataset import FencastDataset
from fencast.models import DynamicFFNN, DynamicCNN


class ModelTrainer:
    """
    A flexible model trainer that can be used for regular training, cross-validation, and hyperparameter tuning.
    """
    
    def __init__(self, config: Dict[str, Any], model_type: str, params: Dict[str, Any], logger=None):
        """
        Initialize the trainer.
        
        Args:
            config: Project configuration dictionary
            model_type: Model architecture ('ffnn' or 'cnn')
            params: Training hyperparameters
            logger: Logger instance (optional)
        """
        self.config = config
        self.model_type = model_type
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
        activation_fn = getattr(nn, self.params['activation_name'])()
        
        if self.model_type == 'ffnn':
            model_args = {
                'input_size': self.config['input_size_flat'],
                'output_size': self.config['target_size'],
                'hidden_layers': self.params['hidden_layers'],
                'dropout_rate': self.params['dropout_rate'],
                'activation_fn': activation_fn
            }
            model = DynamicFFNN(**model_args).to(self.device)
        elif self.model_type == 'cnn':
            model_args = {
                'config': self.config,
                'params': self.params
            }
            model = DynamicCNN(**model_args).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
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
            
        train_dataset = FencastDataset(config=self.config, mode=train_mode, model_type=self.model_type)
        val_dataset = FencastDataset(config=self.config, mode=val_mode, model_type=self.model_type)
        
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def create_custom_data_loaders(self, train_years: List[int], val_years: List[int], 
                                 batch_size: Optional[int] = None) -> Tuple[DataLoader, DataLoader]:
        """
        Create data loaders with custom year filtering for cross validation.
        
        Args:
            train_years: Years to include in training dataset
            val_years: Years to include in validation dataset
            batch_size: Batch size (uses config default if None)
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        if batch_size is None:
            batch_size = self.config.get('model', {}).get('batch_sizes', {}).get('training', 64)
            
        train_dataset = FencastDataset(config=self.config, mode='train', model_type=self.model_type, 
                                     apply_normalization=False, custom_years=train_years)
        val_dataset = FencastDataset(config=self.config, mode='validation', model_type=self.model_type, 
                                   apply_normalization=False, custom_years=val_years)
        
        # Apply normalization based on training data
        if self.model_type == 'ffnn':
            self._normalize_ffnn_datasets(train_dataset, val_dataset)
        elif self.model_type == 'cnn':
            self._normalize_cnn_datasets(train_dataset, val_dataset)
        
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def _normalize_ffnn_datasets(self, train_dataset, val_dataset):
        """Normalize FFNN datasets using training data statistics."""
        from sklearn.preprocessing import StandardScaler
        
        exclude_patterns = self.config.get('features', {}).get('normalization', {}).get('exclude_patterns', [])
        keep_values_columns = [col for col in train_dataset.X.columns for pattern in exclude_patterns if pattern in col]
        normalize_columns = [col for col in train_dataset.X.columns if col not in keep_values_columns]
        
        # Fit scaler on training data
        scaler = StandardScaler()
        train_dataset.X[normalize_columns] = scaler.fit_transform(train_dataset.X[normalize_columns])
        val_dataset.X[normalize_columns] = scaler.transform(val_dataset.X[normalize_columns])
    
    def _normalize_cnn_datasets(self, train_dataset, val_dataset):
        """Normalize CNN datasets using training data statistics."""
        import numpy as np
        
        # Calculate statistics from training data
        mean = np.mean(train_dataset.X, axis=(0, 2, 3), keepdims=True)
        std = np.std(train_dataset.X, axis=(0, 2, 3), keepdims=True)
        std[std == 0] = 1e-7
        
        # Apply normalization
        train_dataset.X = (train_dataset.X - mean) / std
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
        if self.model_type == 'cnn':
            spatial_features, temporal_features, labels = batch
            spatial_features = spatial_features.to(self.device)
            temporal_features = temporal_features.to(self.device)
            outputs = model(spatial_features, temporal_features)
        else:  # FFNN
            features, labels = batch
            features = features.to(self.device)
            outputs = model(features)
            
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
            if self.model_type == 'cnn':
                labels = batch[2].to(self.device)  # spatial, temporal, labels
            else:
                labels = batch[1].to(self.device)  # features, labels
            
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
                if self.model_type == 'cnn':
                    labels = batch[2].to(self.device)  # spatial, temporal, labels
                else:
                    labels = batch[1].to(self.device)  # features, labels
                
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
            # Train and validate
            avg_train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
            avg_val_loss = self.validate_epoch(model, val_loader, criterion)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            # Update scheduler
            if scheduler:
                scheduler.step(avg_val_loss)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if save_path:
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        'model_type': self.model_type,
                        'model_args': model_args,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'train_loss': avg_train_loss,
                        'val_loss': avg_val_loss,
                        'config': self.config,
                        'params': self.params
                    }, save_path)
                
                if self.logger:
                    self.logger.info(f"Epoch [{epoch+1}/{epochs}], Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f} (saved)")
            else:
                if self.logger:
                    self.logger.info(f"Epoch [{epoch+1}/{epochs}], Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}")
        
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