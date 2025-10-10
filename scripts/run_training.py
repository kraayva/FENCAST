# scripts/run_training.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import optuna
import argparse
import json
from pathlib import Path

# Import our custom modules
from fencast.utils.paths import load_config, PROJECT_ROOT
from fencast.dataset import FencastDataset
from fencast.models import DynamicFFNN, DynamicCNN
from fencast.utils.tools import setup_logger, get_latest_study_dir

logger = setup_logger("final_training")

def run_training(config: dict, model_type: str, params: dict, study_dir: Path):
    """
    Main function to run the final model training and validation process using a given set of hyperparameters.
    
    Args:
        config (dict): The project's configuration dictionary.
        model_type (str): The model architecture to train ('ffnn' or 'cnn').
        params (dict): A dictionary containing all necessary hyperparameters for the model.
        study_dir (Path): Directory where the study results and model checkpoints will be saved.
    """
    # 1. SETUP & VALIDATION
    # =================================================================================
    logger.info("--- 1. Setting up experiment and validating parameters ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Validate essential parameters
    required_params = ['lr', 'activation_name', 'dropout_rate']
    missing_params = [p for p in required_params if p not in params]
    if missing_params:
        logger.error(f"Missing required parameters: {missing_params}")
        raise ValueError(f"Missing required parameters: {missing_params}")
    
    logger.info(f"Training parameters validated: {json.dumps({k: v for k, v in params.items() if k in required_params + ['optimizer_name', 'weight_decay', 'scheduler_name']}, indent=2)}")

    # 2. DATA LOADING
    # =================================================================================
    logger.info(f"--- 2. Loading data for '{model_type}' model ---")
    # Pass model_type to the dataset to ensure correct data format is loaded
    train_dataset = FencastDataset(config=config, mode='train', model_type=model_type)
    validation_dataset = FencastDataset(config=config, mode='validation', model_type=model_type)
    
    batch_size = config.get('model', {}).get('batch_sizes', {}).get('training', 64)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)

    # 3. MODEL, LOSS, and OPTIMIZER
    # =================================================================================
    logger.info("--- 3. Initializing model, loss, and optimizer ---")

    model_args = {}
    model = None
    activation_fn = getattr(nn, params['activation_name'])()
    
    # Create the model based on its type and the provided hyperparameters
    if model_type == 'ffnn':
        model_args = {
            'input_size': config['input_size_flat'],
            'output_size': config['target_size'],
            'hidden_layers': params['hidden_layers'],
            'dropout_rate': params['dropout_rate'],
            'activation_fn': activation_fn
        }
        model = DynamicFFNN(**model_args).to(device)
    elif model_type == 'cnn':
        model_args = {
            'config': config,
            'params': params
        }
        model = DynamicCNN(**model_args).to(device)
    
    criterion = nn.MSELoss()
    # Create optimizer based on configuration 
    optimizer_name = params.get('optimizer_name', 'Adam')
    if optimizer_name == 'AdamW':
        weight_decay = params.get('weight_decay', 0.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=weight_decay)
        logger.info(f"Using optimizer: {optimizer_name} with weight_decay={weight_decay:.2e}")
    else:  # Default to Adam
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
        logger.info(f"Using optimizer: {optimizer_name}")

    # Create scheduler based on configuration
    scheduler = None
    if params.get('scheduler_name') == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=params.get('scheduler_factor', 0.2),
            patience=params.get('scheduler_patience', 3)
        )
        logger.info(f"Using ReduceLROnPlateau scheduler with factor={params.get('scheduler_factor', 0.2)}, patience={params.get('scheduler_patience', 3)}")

    # 4. TRAINING LOOP
    # =================================================================================
    logger.info("--- 4. Starting training ---")
    # Get epochs from config with fallback hierarchy
    epochs = config.get('tuning', {}).get('epochs', config.get('training', {}).get('final_epochs', 50))
    logger.info(f"Training for {epochs} epochs")
    best_validation_loss = float('inf')
    
    # Make model save path unique to the model type
    model_save_path = study_dir / "best_model.pth"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            if model_type == 'cnn':
                spatial_features, temporal_features, labels = batch
                spatial_features, temporal_features, labels = spatial_features.to(device), temporal_features.to(device), labels.to(device)
                outputs = model(spatial_features, temporal_features)
            else: # FFNN
                features, labels = batch
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
            
            loss = criterion(outputs, labels)
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            max_norm = config.get('training', {}).get('gradient_clip_max_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
        avg_train_loss = np.mean(train_losses)

        model.eval()
        validation_losses = []
        with torch.no_grad():
            for batch in validation_loader:
                if model_type == 'cnn':
                    spatial_features, temporal_features, labels = batch
                    spatial_features = spatial_features.to(device)
                    temporal_features = temporal_features.to(device)
                    labels = labels.to(device)
                    outputs = model(spatial_features, temporal_features)
                else:  # FFNN
                    features, labels = batch
                    features, labels = features.to(device), labels.to(device)
                    outputs = model(features)
                
                loss = criterion(outputs, labels)
                validation_losses.append(loss.item())
        avg_validation_loss = np.mean(validation_losses)

        # Update the learning rate scheduler with the new validation loss
        if scheduler:
            scheduler.step(avg_validation_loss)

        # Save the best model checkpoint
        if avg_validation_loss < best_validation_loss:
            best_validation_loss = avg_validation_loss
            torch.save({
                'model_type': model_type,
                'model_args': model_args,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_save_path)
            logger.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_validation_loss:.6f} (saved)")
        else:
            logger.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_validation_loss:.6f}")

    logger.info("\n--- Training finished ---")
    logger.info(f"Best model validation loss: {best_validation_loss:.6f}")
    logger.info(f"Model saved to: {model_save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run final model training with best hyperparameters from a study.')
    parser.add_argument(
        '--config', '-c', 
        default='datapp_de',
        help='Configuration file name (default: datapp_de)'
    )
    parser.add_argument(
        '--model-type', '-m',
        required=True,
        choices=['ffnn', 'cnn'],
        help='The model architecture to train.'
    )
    parser.add_argument(
        '--study-name', '-s',
        default='latest',
        help='Specify the study name to load params from (default: latest for the given model-type).'
    )
    
    args = parser.parse_args()
    config = load_config(args.config)
    setup_name = config.get('setup_name', 'default_setup')
    results_parent_dir = PROJECT_ROOT / "results" / setup_name

    logger.info("--- Loading best hyperparameters from Optuna study ---")
    try:
        if args.study_name == 'latest':
            study_dir = get_latest_study_dir(results_parent_dir, args.model_type)
            logger.info("Loading best trained model...")
        else:
            study_dir = results_parent_dir / args.study_name

        study_name = study_dir.name
        storage_name = f"sqlite:///{study_dir / study_name}.db"
        
        study = optuna.load_study(study_name=study_name, storage=storage_name)
        params = study.best_trial.params
        logger.info(f"Loaded best parameters from study: '{study_name}'")
    except Exception as e:
        logger.error(f"Failed to load study: {e}")
        raise

    # Use parameters directly from the study
    final_params = dict(params)  # Create a copy
    
    # Handle any missing required parameters with defaults
    if 'lr' not in final_params:
        final_params['lr'] = 1e-3
        logger.warning("No learning rate found, using default 1e-3")
    if 'activation_name' not in final_params:
        final_params['activation_name'] = 'ELU'
        logger.warning("No activation found, using default ELU")
    
    # Handle legacy parameter names if they exist
    if 'learning_rate' in final_params and 'lr' not in final_params:
        final_params['lr'] = final_params['learning_rate']
    if 'activation' in final_params and 'activation_name' not in final_params:
        final_params['activation_name'] = final_params['activation']

    logger.info(f"Final training hyperparameters:\n{json.dumps(final_params, indent=4)}")

    # Run final training
    run_training(
        config=config,
        model_type=args.model_type,
        params=final_params,
        study_dir=study_dir
    )