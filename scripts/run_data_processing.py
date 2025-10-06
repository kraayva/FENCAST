# scripts/run_data_processing.py

import argparse
from pathlib import Path
import numpy as np # Import numpy for saving arrays

from fencast.utils.paths import load_config, PROCESSED_DATA_DIR
from fencast.data_processing import load_and_prepare_data
from fencast.utils.tools import setup_logger

def run_data_processing(config_name: str, model_target: str, force_save: bool, features_prefix: str = "era5_de"):
    """
    Process raw data according to configuration and save processed files.
    
    Args:
        config_name (str): Name of the configuration file to use.
        model_target (str): The target model architecture ('ffnn' or 'cnn').
        force_save (bool): If True, save without prompting.
        features_prefix (str): Prefix for feature data files. Default is "era5_de".
    """
    logger = setup_logger("data_processing")
    
    try:
        # 1. Load configuration
        logger.info(f"Loading configuration '{config_name}'...")
        config = load_config(config_name)
        
        # 2. Process the data based on the model target
        logger.info(f"Starting data processing for target: '{model_target}'...")
        X_processed, y_processed = load_and_prepare_data(config=config, model_target=model_target, feature_prefix=features_prefix)
        
        # 3. Define paths and save the results
        setup_name = config.get('setup_name', 'default_setup')
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        if model_target == 'ffnn':
            features_path = PROCESSED_DATA_DIR / f"{setup_name}_features_ffnn.parquet"
            labels_path = PROCESSED_DATA_DIR / f"{setup_name}_labels_ffnn.parquet"
        elif model_target == 'cnn':
            features_path = PROCESSED_DATA_DIR / f"{setup_name}_features_cnn.npz"
            labels_path = PROCESSED_DATA_DIR / f"{setup_name}_labels_cnn.parquet" # Labels are likely still 2D

        # Check for existing files and prompt for overwrite if necessary
        should_save = True
        if features_path.exists() and not force_save:
            logger.warning(f"Processed file already exists: {features_path}")
            response = input("Overwrite existing file? (y/n): ").lower()
            should_save = (response == 'y')
            
        if should_save:
            logger.info(f"Saving processed data to {PROCESSED_DATA_DIR}...")
            
            # Saving logic depends on the model target
            if model_target == 'ffnn':
                X_processed.to_parquet(features_path)
                y_processed.to_parquet(labels_path)
                logger.info(f"  Shape:    {X_processed.shape[0]} samples, {X_processed.shape[1]} features")
            elif model_target == 'cnn':
                # Save features as a compressed NumPy array and labels as Parquet
                np.savez_compressed(features_path, features=X_processed)
                y_processed.to_parquet(labels_path)
                logger.info(f"  Features Shape: {X_processed.shape}")

            logger.info(f"Data processing complete!")
            logger.info(f"   Features: {features_path}")
            logger.info(f"   Labels:   {labels_path}")
        else:
            logger.info("Data processing was cancelled. No files were saved.")
            
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process raw data for model training.')
    parser.add_argument(
        '--config', '-c', 
        default='datapp_de',
        help='Configuration file name (default: datapp_de)'
    )
    parser.add_argument(
        '--model-target', '-m',
        required=True,
        choices=['ffnn', 'cnn'],
        help='The target model architecture to process data for.'
    )
    parser.add_argument(
        '--force-save', '-f',
        action='store_true',
        help='Save without prompting (overwrite existing files)'
    )
    parser.add_argument(
        '--feature-prefix', '-p',
        default='era5_de',
        help='Prefix for feature data files (default: era5_de)'
    )
    
    args = parser.parse_args()
    
    run_data_processing(
        config_name=args.config, 
        model_target=args.model_target,
        force_save=args.force_save,
        features_prefix=args.feature_prefix
    )