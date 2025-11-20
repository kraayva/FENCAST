# scripts/run_training.py

import torch
import argparse
import json
from pathlib import Path

# Import our custom modules
from fencast.utils.paths import load_config, PROJECT_ROOT
from fencast.utils.tools import setup_logger, get_latest_study_dir
from fencast.utils.experiment_management import load_best_params_from_study
from fencast.training import ModelTrainer, validate_training_parameters

def run_training(config: dict, model_type: str, params: dict, study_dir: Path, use_all_data: bool = False) -> dict:
    """
    Main function to run the model training and validation process using a given set of hyperparameters.
    
    Args:
        config (dict): The project's configuration dictionary.
        model_type (str): The model architecture to train ('ffnn' or 'cnn').
        params (dict): A dictionary containing all necessary hyperparameters for the model.
        study_dir (Path): Directory where the study results and model checkpoints will be saved.
        use_all_data (bool): Whether to use all available data (training + validation) for training. Default is False.
    """
    logger.info("--- Starting final model training ---")
    logger.info(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Create trainer instance
    trainer = ModelTrainer(config, model_type, params, logger)
    
    # Create data loaders based on the training mode
    if use_all_data:
        # Mode 1: Final training run on all data
        logger.info("Mode: Training on combined train + validation data.")
        
        # Get the explicitly defined validation and test years and derive training years
        val_years = config['split_years']['validation']
        test_years = config['split_years']['test']
        all_possible_years = list(range(int(config['time_start'][:4]), int(config['time_end'][:4]) + 1))
        train_years = [year for year in all_possible_years if year not in set(val_years + test_years)]
        
        # Now, combine the calculated train years and the validation years for the final training run
        all_training_years = sorted(train_years + val_years)
        logger.info(f"Combining data from {len(all_training_years)} years.")

        # Create loaders: all data for training, empty for validation
        train_loader, val_loader = trainer.create_custom_data_loaders(
            train_years=all_training_years,
            val_years=[]
        )
        
        epochs = config.get('training', {}).get('final_epochs', 50)
        model_save_path = study_dir / "final_model.pth"
        
    else:
        # Mode 2: Standard training run with separate validation
        logger.info("Mode: Training with separate train and validation sets.")
        
        train_loader, val_loader = trainer.create_data_loaders()
        
        epochs = config.get('tuning', {}).get('epochs', 30)
        model_save_path = study_dir / "best_model.pth"

    logger.info(f"Data loaders created. Training for {epochs} epochs.")
    
    # The train_model call is now universal, thanks to the robust ModelTrainer
    results = trainer.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        save_path=model_save_path
    )
    
    logger.info("--- Training finished ---")
    logger.info(f"Best model validation loss: {results['best_val_loss']:.6f}")
    logger.info(f"Model saved to: {model_save_path}")
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run final model training with best hyperparameters from a study.')
    parser.add_argument('--config', '-c', 
        default='datapp_de',
        help='Configuration file name (default: datapp_de)'
    )
    parser.add_argument('--model-type', '-m',
        choices=['ffnn', 'cnn'],
        default='cnn',
        help='The model architecture to train.'
    )
    parser.add_argument(
        '--study-name', '-s',
        default='latest',
        help='Specify the study name to load params from (default: latest for the given model-type).'
    )
    parser.add_argument(
        '--final-run',
        action='store_true',
        help='Flag to train on all data (train + validation) to produce a final model.'
    )
    
    args = parser.parse_args()
    config = load_config(args.config)
    setup_name = config.get('setup_name', 'default_setup')
    results_parent_dir = PROJECT_ROOT / "results" / setup_name

    if args.final_run:
        logger = setup_logger("final_training")
    else:
        logger = setup_logger("training")

    logger.info("--- Loading best hyperparameters from Optuna study ---")
    params, study_dir = load_best_params_from_study(
        results_parent_dir=results_parent_dir,
        model_type=args.model_type,
        study_name=args.study_name
    )

    # Validate and clean parameters
    final_params = validate_training_parameters(params)

    logger.info(f"Final training hyperparameters:\n{json.dumps(final_params, indent=4)}")

    # Run final training
    run_training(
        config=config,
        model_type=args.model_type,
        params=final_params,
        study_dir=study_dir,
        use_all_data=args.final_run
    )