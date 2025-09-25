# scripts/run_data_processing.py

import argparse
from pathlib import Path

from fencast.utils.paths import load_config, PROCESSED_DATA_DIR, PROJECT_ROOT
from fencast.data_processing import load_and_prepare_data
from fencast.utils.tools import setup_logger

def run_data_processing(config_name: str = "datapp_de", force_save: bool = True):
    """
    Process raw data according to configuration and save processed files.
    
    Args:
        config_name (str): Name of the configuration file to use
        force_save (bool): If True, save without prompting
    """
    logger = setup_logger("data_processing")
    
    try:
        # 1. Load configuration
        logger.info(f"Loading configuration '{config_name}'...")
        config = load_config(config_name)
        
        # 2. Process the data
        logger.info("Starting data processing...")
        X_processed, y_processed = load_and_prepare_data(config=config)
        
        # 3. Save the results
        setup_name = config.get('setup_name', 'default_setup')
        features_path = PROCESSED_DATA_DIR / f"{setup_name}_features.parquet"
        labels_path = PROCESSED_DATA_DIR / f"{setup_name}_labels.parquet"
        
        # Check if files already exist
        files_exist = features_path.exists() and labels_path.exists()
        
        if files_exist and not force_save:
            logger.warning(f"Processed files already exist at {PROCESSED_DATA_DIR}")
            logger.warning(f"  - {features_path.name}")
            logger.warning(f"  - {labels_path.name}")
            response = input("\nOverwrite existing files? (y/n): ").lower()
            should_save = response == 'y'
        else:
            should_save = True
            
        if should_save:
            # Convert data types
            data_type = config.get('data_processing', {}).get('data_type', 'float32')
            logger.info(f"Converting data to {data_type}...")
            X_processed = X_processed.astype(data_type)
            y_processed = y_processed.astype(data_type)
            
            # Create directories and save
            PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Saving processed data...")
            X_processed.to_parquet(features_path)
            y_processed.to_parquet(labels_path)
            
            logger.info(f"✅ Data processing complete!")
            logger.info(f"   Features: {features_path}")
            logger.info(f"   Labels:   {labels_path}")
            logger.info(f"   Shape:    {X_processed.shape[0]} samples, {X_processed.shape[1]} features")
        else:
            logger.info("Data processing completed but files were not saved.")
            
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process raw data for FENCAST training')
    parser.add_argument('--config', '-c', 
                       default='datapp_de',
                       help='Configuration file name (default: datapp_de)')
    parser.add_argument('--force-save', '-f',
                       action='store_true',
                       help='Save without prompting (overwrite existing files)')
    parser.add_argument('--quiet', '-q',
                       action='store_true', 
                       help='Skip confirmation prompts')
    
    args = parser.parse_args()
    
    # If quiet mode, force save without prompting
    force_save = args.force_save or args.quiet
    
    run_data_processing(config_name=args.config, force_save=force_save)