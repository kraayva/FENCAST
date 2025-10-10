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
from fencast.utils.tools import setup_logger, get_latest_study_dir
from fencast.training import ModelTrainer, validate_training_parameters

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
    logger.info("--- Starting final model training ---")
    logger.info(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Create trainer instance
    trainer = ModelTrainer(config, model_type, params, logger)
    
    # Create data loaders
    train_loader, val_loader = trainer.create_data_loaders()
    logger.info(f"Data loaders created for '{model_type}' model")
    
    # Get training configuration
    epochs = config.get('tuning', {}).get('epochs', config.get('training', {}).get('final_epochs', 50))
    model_save_path = study_dir / "best_model.pth"
    
    # Train the model
    results = trainer.train_model(train_loader, val_loader, epochs, model_save_path)
    
    logger.info("--- Training finished ---")
    logger.info(f"Best model validation loss: {results['best_val_loss']:.6f}")
    logger.info(f"Model saved to: {model_save_path}")
    
    return results

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

    # Validate and clean parameters
    final_params = validate_training_parameters(params)

    logger.info(f"Final training hyperparameters:\n{json.dumps(final_params, indent=4)}")

    # Run final training
    run_training(
        config=config,
        model_type=args.model_type,
        params=final_params,
        study_dir=study_dir
    )