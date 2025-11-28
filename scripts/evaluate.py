# scripts/evaluate.py

'''Entry point for evaluating a trained model on the test set (only ERA5 data).'''

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import torch.nn as nn
import argparse
from pathlib import Path

from fencast.utils.paths import load_config, PROJECT_ROOT
from fencast.dataset import FencastDataset
from fencast.models import DynamicCNN
from fencast.utils.tools import setup_logger, get_latest_study_dir
from fencast.utils.parser import get_parser

logger = setup_logger("evaluation")

def get_predictions(model, data_loader, device):
    """Runs the model on the test set and returns predictions and labels."""
    model.eval()
    all_predictions, all_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            spatial_features, temporal_features, labels = batch
            spatial_features, temporal_features, labels = spatial_features.to(device), temporal_features.to(device), labels.to(device)
            outputs = model(spatial_features, temporal_features)

            all_predictions.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_predictions, all_labels

def calculate_persistence_metrics(labels_df: pd.DataFrame):
    """Calculates RMSE and MAE for the persistence (naive) baseline."""
    persistence_preds = labels_df.shift(1)
    # Align by dropping the first row which has no prediction
    valid_labels = labels_df.iloc[1:]
    valid_preds = persistence_preds.iloc[1:]
    rmse = np.sqrt(mean_squared_error(valid_labels, valid_preds))
    mae = mean_absolute_error(valid_labels, valid_preds)
    return rmse, mae

def calculate_climatology_baseline(labels_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Calculates a climatology baseline.
    For each day in the test set, the prediction is the average of that
    day's value over a historical period (1990-1999).
    """
    logger.info("Calculating climatology baseline...")
    
    # 1. Load the full, raw target dataset
    full_cf_path = Path(config['target_data_raw'])
    full_df = pd.read_csv(full_cf_path, index_col='Date', parse_dates=True)
    
    # 2. Filter for the historical reference period (1990-1999)
    historical_data = full_df.loc['1990':'1999']
    
    # 3. Calculate the mean value for each day of the year (1-366)
    daily_climatology = historical_data.groupby([historical_data.index.month, historical_data.index.day]).mean()
    daily_climatology.index.names = ['month', 'day']
    
    # 4. Create predictions for the test set dates
    preds_df = pd.DataFrame(index=labels_df.index)
    preds_df['month'] = preds_df.index.month
    preds_df['day'] = preds_df.index.day
    
    # Merge the daily averages onto the test set dates
    merged = pd.merge(preds_df, daily_climatology, on=['month', 'day'], how='left')
    
    merged = merged.fillna(method='ffill').fillna(method='bfill')
    
    merged.index = labels_df.index
    climatology_preds_df = merged[labels_df.columns]
    
    return climatology_preds_df

def create_plots(labels_df, nn_preds_df, climatology_preds_df, results_dir):
    """Creates and saves time-series and scatter plots."""
    persistence_preds_df = labels_df.shift(1)
    
    # --- Time-Series Plot (with Baselines) ---
    region_to_plot = labels_df.columns[0]
    start_date = labels_df.index.min() + pd.Timedelta(days=240)
    end_date = start_date + pd.Timedelta(days=60)
    plot_slice = slice(start_date, end_date)

    plt.figure(figsize=(15, 7))
    plt.plot(labels_df.loc[plot_slice].index, labels_df.loc[plot_slice, region_to_plot], label='Actual Values', color='black', linewidth=2)
    plt.plot(nn_preds_df.loc[plot_slice].index, nn_preds_df.loc[plot_slice, region_to_plot], label='CNN Predictions', color='blue', linestyle='--')
    plt.plot(persistence_preds_df.loc[plot_slice].index, persistence_preds_df.loc[plot_slice, region_to_plot], label='Persistence Baseline', color='green', linestyle=':')
 
    plt.plot(climatology_preds_df.loc[plot_slice].index, climatology_preds_df.loc[plot_slice, region_to_plot], label='Climatology Baseline', color='red', linestyle='-.')
    
    plt.title(f'CNN Predictions vs Actuals for {region_to_plot} - {start_date.strftime("%B %Y")}')
    plt.xlabel('Date'); plt.ylabel('Capacity Factor')
    plt.legend(); plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(results_dir / 'timeseries_plot_cnn.png')
    logger.info(f"Time-series plot saved to {results_dir}.")
    plt.close()

def evaluate(config_name: str, study_name: str, model_name: str):
    """Main evaluation function."""
    logger.info(f"--- Starting Final Model Evaluation for '{model_name}' ---")
    
    # 1. SETUP & DATA LOADING
    config = load_config(config_name)
    setup_name = config.get('setup_name', 'default_setup')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results_parent_dir = PROJECT_ROOT / "results" / setup_name
    try:
        if study_name == 'latest':
            study_dir = get_latest_study_dir(results_parent_dir)
            study_name = study_dir.name
        else:
            study_dir = results_parent_dir / study_name
        logger.info(f"Using results from study directory: {study_name}")
    except FileNotFoundError as e:
        logger.error(e)
        return
    
    logger.info("Loading test set data...")
    test_dataset = FencastDataset(config=config, mode='test')
    batch_size = config.get('model', {}).get('batch_sizes', {}).get('evaluation', 256)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. LOAD BEST MODEL
    logger.info("Loading best trained model...")
    model_path = study_dir / f"{model_name}.pth"
    if not model_path.exists():
        logger.error(f"{model_name}.pth not found at {model_path}. Please run training first.")
        return
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = DynamicCNN(**checkpoint['model_args']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 3. GET PREDICTIONS
    logger.info("Generating predictions on the test set...")
    predictions_np, labels_np = get_predictions(model, test_loader, device)
    
    labels_df = pd.DataFrame(labels_np, index=test_dataset.y.index, columns=test_dataset.y.columns)
    preds_df = pd.DataFrame(predictions_np, index=test_dataset.y.index, columns=test_dataset.y.columns)
    
    # 4. CALCULATE METRICS
    logger.info("Calculating performance metrics...")
    nn_rmse = np.sqrt(mean_squared_error(labels_df, preds_df))
    nn_mae = mean_absolute_error(labels_df, preds_df)
    
    persistence_rmse, persistence_mae = calculate_persistence_metrics(labels_df)
    
    climatology_preds_df = calculate_climatology_baseline(labels_df, config)
    climatology_rmse = np.sqrt(mean_squared_error(labels_df, climatology_preds_df))
    climatology_mae = mean_absolute_error(labels_df, climatology_preds_df)

    logger.info("\n" + "="*35)
    logger.info("      PERFORMANCE SUMMARY")
    logger.info("="*35)
    logger.info(f"  Persistence Model:")
    logger.info(f"    RMSE: {persistence_rmse:.6f}")
    logger.info(f"    MAE:  {persistence_mae:.6f}")
    logger.info(f"  Climatology Model (1999-2009 Avg):")
    logger.info(f"    RMSE: {climatology_rmse:.6f}")
    logger.info(f"    MAE:  {climatology_mae:.6f}")
    logger.info(f"  CNN Model:")
    logger.info(f"    RMSE: {nn_rmse:.6f}")
    logger.info(f"    MAE:  {nn_mae:.6f}")
    logger.info("="*35)

    # 5. VISUALIZE RESULTS
    logger.info("Creating visualizations...")
    create_plots(labels_df, preds_df, climatology_preds_df, study_dir)

if __name__ == '__main__':
    parser_names = ['config', 'study_name', 'model_name']
    parser = get_parser(parser_names, description="Evaluate a trained model on the test set.")
    args = parser.parse_args()

    if args.config == 'default':
        config = load_config()
        args.config = config.get('default_config')
    
    evaluate(config_name=args.config, study_name=args.study_name, model_name=args.model_name)