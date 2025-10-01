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
from fencast.models import DynamicFFNN, DynamicCNN # Import both models
from fencast.utils.tools import setup_logger

logger = setup_logger("final_training")

def run_training(config: dict, model_type: str, params: dict):
    """
    Main function to orchestrate the final model training and validation process
    using a given set of hyperparameters.
    
    Args:
        config (dict): The project's configuration dictionary.
        model_type (str): The model architecture to train ('ffnn' or 'cnn').
        params (dict): A dictionary containing all necessary hyperparameters for the model.
    """
    # 1. SETUP
    # =================================================================================
    logger.info("--- 1. Setting up experiment ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

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
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',      # We want to minimize the validation loss
        factor=0.2,      # Reduce LR by a factor of 0.2 (e.g., 0.001 -> 0.0002)
        patience=3,      # Wait 3 epochs with no improvement before reducing LR
    )

    # 4. TRAINING LOOP
    # =================================================================================
    logger.info("--- 4. Starting training ---")
    epochs = config.get('training', {}).get('final_epochs', 30)
    best_validation_loss = float('inf')
    
    # Make model save path unique to the model type
    setup_name = config.get('setup_name', 'default_setup')
    model_save_path = PROJECT_ROOT / "model" / f"{setup_name}_{model_type}_best_model.pth"
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


def get_latest_study_dir(results_parent_dir: Path, model_type: str) -> Path:
    """Finds the most recent study directory for a given model type."""
    logger.info(f"Searching for latest study for model type '{model_type}' in {results_parent_dir}...")
    
    # Filter directories that match the study prefix for the model type
    prefix = f"study_{model_type}"
    model_studies = [d for d in results_parent_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    
    if not model_studies:
        raise FileNotFoundError(f"No study found for model type '{model_type}' in {results_parent_dir}")
        
    # Sort by modification time and return the latest one
    latest_study_dir = sorted(model_studies, key=lambda f: f.stat().st_mtime, reverse=True)[0]
    return latest_study_dir


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

    # Reconstruct architectural parameters from the Optuna trial format
    final_params = {
        "learning_rate": params["lr"],
        "activation_name": params["activation"],
        "dropout_rate": params["dropout"]
    }

    if args.model_type == 'ffnn':
        final_params["hidden_layers"] = [params[f"n_units_l{i}"] for i in range(params["n_layers"])]
    elif args.model_type == 'cnn':
        final_params["out_channels"] = [params[f"n_filters_l{i}"] for i in range(params["n_conv_layers"])]
        final_params["kernel_size"] = params["kernel_size"]

    logger.info(f"Final training hyperparameters:\n{json.dumps(final_params, indent=4)}")

    # Run final training
    run_training(
        config=config,
        model_type=args.model_type,
        params=final_params
    )