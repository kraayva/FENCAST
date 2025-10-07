# scripts/mlwp_evaluation.py

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import xarray as xr
import argparse
import joblib
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch.nn as nn

# Import our custom modules
from fencast.utils.paths import load_config, PROJECT_ROOT, PROCESSED_DATA_DIR, RAW_DATA_DIR
from fencast.models import DynamicCNN
from fencast.utils.tools import setup_logger, get_latest_study_dir

logger = setup_logger("mlwp_evaluation")

def load_mlwp_data(config: dict, mlwp_name: str, timedelta: str) -> xr.Dataset:
    """Loads and merges a specific MLWP forecast dataset."""
    feature_prefix = f"{mlwp_name}_{timedelta}_de"
    feature_var_names = config['feature_var_names']
    
    feature_files = [RAW_DATA_DIR / f'{feature_prefix}_{var}.nc' for var in feature_var_names.keys()]
    logger.info(f"Loading files with prefix '{feature_prefix}'...")
    
    for f in feature_files:
        if not f.exists():
            raise FileNotFoundError(f"Required data file not found: {f}")
            
    datasets = [xr.open_dataset(f) for f in feature_files]
    weather_data = xr.merge(datasets, compat='override', join='inner')
    weather_data = weather_data.rename(config['feature_var_names'])
    return weather_data

def predict_and_evaluate_run(config: dict, model: nn.Module, setup_name: str, study_dir: Path, mlwp_name: str, timedelta_str: str):
    """
    Performs a single prediction and evaluation run for one MLWP and timedelta.
    """
    run_name = f"{mlwp_name}_{timedelta_str}"
    logger.info(f"\n--- Starting run for: {run_name} ---")

    # 1. LOAD AND PROCESS THE MLWP FEATURE DATA
    # ============================================================================
    try:
        mlwp_data_xr = load_mlwp_data(config, mlwp_name, timedelta_str)
    except FileNotFoundError as e:
        logger.error(f"Could not run '{run_name}'. Reason: {e}")
        return

    # Process data into the 4D tensor format required by the CNN
    var_names = list(config['feature_var_names'].values())
    da_weather = mlwp_data_xr[var_names].to_array(dim='variable')
    da_weather = da_weather.transpose('time', 'variable', 'level', 'latitude', 'longitude')
    X_np = da_weather.values
    n_samples, n_vars, n_levels, n_lat, n_lon = X_np.shape
    X_spatial = X_np.reshape(n_samples, n_vars * n_levels, n_lat, n_lon)
    
    # Generate temporal features
    timestamps = pd.to_datetime(mlwp_data_xr.time.values)
    day_of_year = timestamps.dayofyear
    norm_denom = 365.0
    day_of_year_rad = ((day_of_year - 1) / norm_denom) * 2 * np.pi
    X_temporal = np.stack([np.sin(day_of_year_rad), np.cos(day_of_year_rad)], axis=1)
    
    # Load and apply the original training scaler
    logger.info("Loading and applying the original training scaler...")
    scaler_path = PROCESSED_DATA_DIR / f"{setup_name}_cnn_scaler.npz"
    with np.load(scaler_path) as data:
        mean, std = data['mean'], data['std']
    X_spatial_scaled = (X_spatial - mean) / std
    
    dataset = TensorDataset(
        torch.tensor(X_spatial_scaled, dtype=torch.float32),
        torch.tensor(X_temporal, dtype=torch.float32)
    )

    # 2. GENERATE PREDICTIONS
    # ============================================================================
    logger.info("Generating predictions...")
    device = next(model.parameters()).device
    data_loader = DataLoader(dataset, batch_size=256, shuffle=False)
    all_predictions = []
    with torch.no_grad():
        for batch in data_loader:
            outputs = model(batch[0].to(device), batch[1].to(device))
            all_predictions.append(outputs.cpu().numpy())
    predictions_np = np.concatenate(all_predictions, axis=0)

    # 3. ALIGN WITH GROUND TRUTH AND EVALUATE
    # ============================================================================
    logger.info("Aligning predictions with ground truth and evaluating...")
    
    # Load the full ground truth dataset
    gt_file = PROJECT_ROOT / config['target_data_raw']
    gt_df = pd.read_csv(gt_file, index_col='Date', parse_dates=True)
    gt_df.index = gt_df.index + pd.Timedelta(hours=12) # Apply same noon-shift as in processing
    
    # Create predictions DataFrame
    preds_df = pd.DataFrame(predictions_np, index=timestamps, columns=gt_df.columns)
    
    # Find common timestamps and align
    common_index = gt_df.index.intersection(preds_df.index)
    aligned_preds = preds_df.loc[common_index]
    aligned_gt = gt_df.loc[common_index]
    
    if aligned_gt.empty:
        logger.warning("No overlapping timestamps found between predictions and ground truth. Cannot calculate metrics.")
        metrics = {'rmse': None, 'mae': None, 'sample_count': 0}
    else:
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(aligned_gt, aligned_preds))
        mae = mean_absolute_error(aligned_gt, aligned_preds)
        metrics = {'rmse': rmse, 'mae': mae, 'sample_count': len(aligned_gt)}
        logger.info(f"Evaluation complete for {run_name}: RMSE = {rmse:.4f}, MAE = {mae:.4f}")

    # 4. SAVE RESULTS
    # ============================================================================
    output_dir = study_dir / "mlwp_evaluation" / mlwp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions and metrics
    preds_df.to_csv(output_dir / f"predictions_{timedelta_str}.csv")
    with open(output_dir / f"metrics_{timedelta_str}.json", 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Results for {run_name} saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained CNN on MLWP forecast datasets.')
    parser.add_argument('--config', '-c', default='datapp_de', help='Configuration file name.')
    parser.add_argument('--study-name', '-s', default='latest', help='Study name to load the model from.')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    setup_name = config.get('setup_name', 'default_setup')
    
    # Find the study directory and load the trained model once
    results_parent_dir = PROJECT_ROOT / "results" / setup_name
    try:
        study_dir = get_latest_study_dir(results_parent_dir, model_type='cnn') if args.study_name == 'latest' else results_parent_dir / args.study_name
        logger.info(f"Using model from study: {study_dir.name}")
        
        model_path = study_dir / "best_model.pth"
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model = DynamicCNN(**checkpoint['model_args'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
    except FileNotFoundError as e:
        logger.error(f"Failed to load model. Reason: {e}")
        exit()

    # Get experiment parameters from config
    mlwp_names = config.get('mlwp_names', [])
    mlwp_timedeltas = config.get('mlwp_timedelta', [])

    if not mlwp_names or not mlwp_timedeltas:
        logger.error("Config keys 'mlwp_names' or 'mlwp_timedelta' are missing or empty in the config file.")
        exit()

    # Loop through all combinations and run evaluation
    for mlwp in mlwp_names:
        for td in mlwp_timedeltas:
            # Format timedelta string like 'td01', 'td02'
            td_str = f"td{td:02d}"
            predict_and_evaluate_run(config, model, setup_name, study_dir, mlwp, td_str)
            
    logger.info("\nAll evaluation runs complete.")