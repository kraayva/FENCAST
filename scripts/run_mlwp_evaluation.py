#!/usr/bin/env python3
"""
Entry point script for evaluating trained CNN models on MLWP weather forecasts.

This script loads a trained CNN model and evaluates its performance when fed with
MLWP weather model predictions instead of ERA5 data. Results are saved for each
MLWP model and forecast lead time combination.
"""

import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import xarray as xr
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch.nn as nn

from fencast.utils.paths import load_config, PROJECT_ROOT, PROCESSED_DATA_DIR
from fencast.utils.tools import setup_logger, get_latest_study_dir, load_ground_truth_data, load_mlwp_data
from fencast.utils.experiment_management import load_trained_model
from fencast.utils.parser import get_parser


def load_mlwp_forecast_data(config: dict, mlwp_name: str, timedelta: int) -> xr.Dataset:
    
    feature_var_names = config['feature_var_names']
    logger.info(f"Loading MLWP data for {mlwp_name} time delta {timedelta} days ...")
    
    # Load each variable using the existing load_mlwp_data function
    timedelta_str = f"{timedelta:02d}"
    datasets = []
    for var_name in feature_var_names.keys():
        try:
            ds = load_mlwp_data(mlwp_name, timedelta_str, var_name)
            datasets.append(ds)
        except FileNotFoundError as e:
            logger.error(f"Missing MLWP data file for variable {var_name}: {e}")
            raise FileNotFoundError(f"Required MLWP data files not found for {mlwp_name} at {timedelta}.")
    
    # Merge all variables and rename according to config
    weather_data = xr.merge(datasets, compat='override', join='inner')
    weather_data = weather_data.rename(config['feature_var_names'])

    # Filter to specified pressure levels if configured (to match training data)
    if 'feature_level' in config and 'level' in weather_data.dims:
        feature_levels = config['feature_level']
        logger.info(f"Filtering MLWP data to specified pressure levels: {feature_levels}")
        weather_data = weather_data.sel(level=feature_levels)

    # shift time coordinates by forecast lead time
    time_delta_hours = np.timedelta64(timedelta * 24, 'h')
    weather_data['time'] = weather_data['time'] + time_delta_hours

    return weather_data


def evaluate_model_on_mlwp(config: dict, model: nn.Module, setup_name: str, study_dir: Path, 
                          mlwp_name: str, timedelta: int, final_model = False) -> dict:
    """
    Evaluates a trained CNN model on MLWP forecast data for a specific lead time.
    
    Args:
        config (dict): Configuration dictionary.
        model (nn.Module): Trained CNN model.
        setup_name (str): Name of the experimental setup.
        study_dir (Path): Directory of the study containing the model.
        mlwp_name (str): Name of the MLWP model to evaluate.
        timedelta (int): Forecast lead time in days (e.g., 1).
        final_model (bool): Whether to use the final model trained on all data.
    Returns:
        dict: Evaluation metrics (RMSE, MAE, sample count)
    """
    timedelta_str = f"td{timedelta:02d}"
    run_name = f"{mlwp_name}_{timedelta_str}"
    logger.info(f"\n--- Evaluating model on: {run_name} ---")

    # 1. LOAD AND PROCESS THE MLWP FEATURE DATA
    try:
        mlwp_data_xr = load_mlwp_forecast_data(config, mlwp_name, timedelta)
    except FileNotFoundError as e:
        logger.error(f"Could not evaluate '{run_name}'. Reason: {e}")
        return {'rmse': None, 'mae': None, 'sample_count': 0}

    # Process data into the 4D tensor format required by the CNN
    var_names = list(config['feature_var_names'].values())
    da_weather = mlwp_data_xr[var_names].to_array(dim='variable')
    da_weather = da_weather.transpose('time', 'variable', 'level', 'latitude', 'longitude')
    X_np = da_weather.values
    n_samples, n_vars, n_levels, n_lat, n_lon = X_np.shape
    X_spatial = X_np.reshape(n_samples, n_vars * n_levels, n_lat, n_lon)
    
    # Generate temporal features (day-of-year encoding)
    timestamps = pd.to_datetime(mlwp_data_xr.time.values)
    day_of_year = timestamps.dayofyear
    norm_denom = config.get('data_processing', {}).get('day_of_year_normalize_denominator', 365.0)
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
    logger.info("Generating CNN predictions on MLWP data...")
    device = next(model.parameters()).device
    data_loader = DataLoader(dataset, batch_size=256, shuffle=False)
    all_predictions = []
    with torch.no_grad():
        for batch in data_loader:
            outputs = model(batch[0].to(device), batch[1].to(device))
            all_predictions.append(outputs.cpu().numpy())
    predictions_np = np.concatenate(all_predictions, axis=0)

    # 3. ALIGN WITH GROUND TRUTH AND EVALUATE
    logger.info("Aligning predictions with ground truth and evaluating...")
    
    # Load processed labels (already at correct 12:00 timestamps) if available
    labels_file = PROCESSED_DATA_DIR / f"{setup_name}_labels_cnn.parquet"
    
    if labels_file.exists():
        logger.info(f"Loading processed labels from: {labels_file}")
        gt_df = pd.read_parquet(labels_file)
    else:
        logger.warning("Processed labels not found, falling back to raw data (may have timestamp misalignment)")
        gt_df = load_ground_truth_data(config, list(range(1990, 2024)))
    
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
        # Find any row that has a NaN in either the ground truth or the predictions
        invalid_rows_mask = aligned_gt.isna().any(axis=1) | aligned_preds.isna().any(axis=1)
        
        # Drop these rows from both DataFrames before calculating metrics
        clean_gt = aligned_gt[~invalid_rows_mask]
        clean_preds = aligned_preds[~invalid_rows_mask]
        
        if invalid_rows_mask.any():
            logger.info(f"Removed {invalid_rows_mask.sum()} out of {len(aligned_gt)} rows containing NaN values before calculating metrics.")

        # Calculate metrics
        if clean_gt.empty:
            logger.warning("All aligned rows contained NaN values. Cannot calculate metrics.")
            metrics = {'rmse': None, 'mae': None, 'sample_count': 0}
        else:
            # Calculate overall metrics
            rmse = np.sqrt(mean_squared_error(clean_gt, clean_preds))
            mae = mean_absolute_error(clean_gt, clean_preds)
            
            # Calculate per-region metrics
            region_metrics = {}
            for region in clean_gt.columns:
                if region in clean_preds.columns:
                    region_rmse = np.sqrt(mean_squared_error(clean_gt[region], clean_preds[region]))
                    region_mae = mean_absolute_error(clean_gt[region], clean_preds[region])
                    region_metrics[region] = {
                        'rmse': region_rmse,
                        'mae': region_mae,
                        'sample_count': len(clean_gt[region].dropna())
                    }
                        
            metrics = {
                'rmse': rmse, 
                'mae': mae, 
                'sample_count': len(clean_gt),
                'forecast_lead_time_days': timedelta,
                'forecast_lead_time_hours': timedelta * 24,
                'region_metrics': region_metrics
            }
            logger.info(f"Evaluation complete for {run_name}: RMSE = {rmse:.4f}, MAE = {mae:.4f}, Lead time = {timedelta:.2f} days")

    # 4. SAVE RESULTS
    if final_model:
        output_dir = study_dir / "final_model" / "mlwp_evaluation" / mlwp_name
    else:
        output_dir = study_dir / "best_model" / "mlwp_evaluation" / mlwp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions and metrics
    preds_df.to_csv(output_dir / f"predictions_{timedelta_str}.csv")
    with open(output_dir / f"metrics_{timedelta_str}.json", 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Results for {run_name} saved to {output_dir}")
    
    return metrics


def main():
    parser = get_parser(['config', 'study_name', 'mlwp_models', 'mlwp_timedeltas', 'final_model'],
                        description="Evaluate trained CNN models on MLWP weather forecasts")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("mlwp_evaluation")
    logger.info(f"Starting MLWP evaluation for config: {args.config}")
    
    # Load configuration
    config = load_config(args.config)
    setup_name = config.get('setup_name', 'default_setup')
    
    # Find the study directory and load the trained model
    results_parent_dir = PROJECT_ROOT / "results" / setup_name
    
    study_dir = get_latest_study_dir(results_parent_dir) if args.study_name == 'latest' else results_parent_dir / args.study_name
    model = load_trained_model(
        study_dir=study_dir, 
        use_final_model=args.final_model, 
        device='cpu'
    )

    # Get experiment parameters from config or command line
    mlwp_names = args.mlwp_models if args.mlwp_models else config.get('mlwp_names', [])

    if not mlwp_names:
        logger.error("No MLWP models specified. Check config or use --mlwp-models argument.")
        return

    # Loop through all combinations and run evaluation
    for mlwp in mlwp_names:
        mlwp_timedeltas = args.timedeltas if args.timedeltas else config.get('mlwp_timedelta_days', [])
        logger.info(f"\nEvaluating MLWP model: {mlwp} on {len(mlwp_timedeltas)} lead times")
        for td in mlwp_timedeltas:
            # Format timedelta string like 'td01', 'td02'
            td_str = f"td{td:02d}"
            try:
                metrics = evaluate_model_on_mlwp(config, model, setup_name, study_dir, mlwp, td, final_model=args.final_model)
            except Exception as e:
                logger.error(f"Error evaluating {mlwp} {td_str}: {e}")
                continue
            
    logger.info("\nAll MLWP evaluation runs complete.")


if __name__ == "__main__":
    # Global logger for the module
    logger = setup_logger("mlwp_evaluation")
    main()