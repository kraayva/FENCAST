# scripts/evaluate.py

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import torch.nn as nn
import argparse
from pathlib import Path

# Import our custom modules
from fencast.utils.paths import load_config, PROJECT_ROOT
from fencast.dataset import FencastDataset
from fencast.models import DynamicFFNN, DynamicCNN # Import both models
from fencast.utils.tools import setup_logger

logger = setup_logger("evaluation")

def get_predictions(model, data_loader, device):
    """Runs the model on the test set and returns predictions and labels."""
    model.eval()
    all_predictions, all_labels = [], []
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            outputs = model(features)
            all_predictions.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_predictions, all_labels

def calculate_persistence_rmse(labels_df: pd.DataFrame):
    """Calculates the RMSE for the persistence (naive) baseline."""
    persistence_preds = labels_df.shift(1)
    # Align by dropping the first row which has no prediction
    valid_labels = labels_df.iloc[1:]
    valid_preds = persistence_preds.iloc[1:]
    return np.sqrt(mean_squared_error(valid_labels, valid_preds))


def create_plots(labels_df, nn_preds_df, results_dir, model_type):
    """Creates and saves time-series and scatter plots."""
    persistence_preds_df = labels_df.shift(1)
    
    # --- Time-Series Plot (with Baseline) ---
    region_to_plot = labels_df.columns[0]
    
    # Plot a 30-day slice for readability
    start_date = labels_df.index.min()
    end_date = start_date + pd.Timedelta(days=30)
    plot_slice = slice(start_date, end_date)

    plt.figure(figsize=(15, 7))
    plt.plot(labels_df[plot_slice].index, labels_df[plot_slice][region_to_plot], label='Actual Values', color='black', linewidth=2)
    plt.plot(nn_preds_df[plot_slice].index, nn_preds_df[plot_slice][region_to_plot], label=f'{model_type.upper()} Predictions', color='blue', linestyle='--')
    plt.plot(persistence_preds_df[plot_slice].index, persistence_preds_df[plot_slice][region_to_plot], label='Persistence Baseline', color='green', linestyle=':')
    
    plt.title(f'{model_type.upper()} Predictions vs Actuals for {region_to_plot} - {start_date.strftime("%B %Y")}')
    plt.xlabel('Date'); plt.ylabel('Capacity Factor')
    plt.legend(); plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(results_dir / f'timeseries_plot_{model_type}.png')
    logger.info(f"Time-series plot saved to {results_dir}.")

    # --- Scatter Plot for the trained model ---
    plt.figure(figsize=(8, 8))
    sample_indices = np.random.choice(labels_df.size, size=min(5000, labels_df.size), replace=False)
    plt.scatter(labels_df.values.flatten()[sample_indices], nn_preds_df.values.flatten()[sample_indices], alpha=0.3, s=10)
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Prediction (y=x)')
    plt.title(f'Scatter Plot: {model_type.upper()} Predictions vs Actuals')
    plt.xlabel('Actual Values'); plt.ylabel('Predicted Values')
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.grid(True); plt.legend(); plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(results_dir / f'scatter_plot_{model_type}.png')
    logger.info(f"{model_type.upper()} scatter plot saved to {results_dir}.")

def get_latest_study_dir(results_parent_dir: Path, model_type: str) -> Path:
    """Finds the most recent study directory for a given model type."""
    logger.info(f"Searching for latest study for model type '{model_type}'...")
    prefix = f"study_{model_type}"
    model_studies = [d for d in results_parent_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    if not model_studies:
        raise FileNotFoundError(f"No study found for model type '{model_type}' in {results_parent_dir}")
    return sorted(model_studies, key=lambda f: f.stat().st_mtime, reverse=True)[0]

def evaluate(config_name: str, model_type: str, study_name: str):
    """Main evaluation function."""
    logger.info(f"--- Starting Final Model Evaluation for '{model_type}' ---")
    
    # 1. SETUP
    config = load_config(config_name)
    setup_name = config.get('setup_name', 'default_setup')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results_parent_dir = PROJECT_ROOT / "results" / setup_name
    try:
        if study_name == 'latest':
            study_dir = get_latest_study_dir(results_parent_dir, model_type)
            study_name = study_dir.name
        else:
            study_dir = results_parent_dir / study_name
        logger.info(f"Using results from study directory: {study_name}")
    except FileNotFoundError as e:
        logger.error(e)
        return
    
    # 2. LOAD DATA (TEST SET)
    logger.info("Loading test set data...")
    test_dataset = FencastDataset(config=config, mode='test', model_type=model_type)
    batch_size = config.get('model', {}).get('batch_sizes', {}).get('evaluation', 256)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. LOAD BEST MODEL
    logger.info("Loading best trained model...")
    final_model_path = PROJECT_ROOT / "model" / f"{setup_name}_{model_type}_best_model.pth"
    if not final_model_path.exists():
        logger.error(f"Final model not found at {final_model_path}. Please run training first.")
        return
    
    checkpoint = torch.load(final_model_path, map_location=device)
    # Dynamically instantiate the correct model
    if checkpoint.get('model_type') == 'cnn':
        model = DynamicCNN(**checkpoint['model_args']).to(device)
    else: # Default to FFNN for backward compatibility
        model = DynamicFFNN(**checkpoint['model_args']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 4. GET PREDICTIONS
    logger.info("Generating predictions on the test set...")
    predictions_np, labels_np = get_predictions(model, test_loader, device)
    
    # Reconstruct DataFrames with correct index and columns for analysis
    labels_df = pd.DataFrame(labels_np, index=test_dataset.y.index, columns=test_dataset.y.columns)
    preds_df = pd.DataFrame(predictions_np, index=test_dataset.y.index, columns=test_dataset.y.columns)
    
    # 5. CALCULATE METRICS
    logger.info("Calculating performance metrics...")
    nn_rmse = np.sqrt(mean_squared_error(labels_df, preds_df))
    nn_mae = mean_absolute_error(labels_df, preds_df)
    persistence_rmse = calculate_persistence_rmse(labels_df)

    logger.info("\n" + "="*30)
    logger.info("  PERFORMANCE SUMMARY")
    logger.info("="*30)
    logger.info(f"  Persistence Model RMSE: {persistence_rmse:.6f}")
    logger.info(f"  {model_type.upper()} Model RMSE:          {nn_rmse:.6f}")
    logger.info(f"  {model_type.upper()} Model MAE:           {nn_mae:.6f}")
    logger.info("="*30)

    # 6. VISUALIZE RESULTS
    logger.info("Creating visualizations...")
    create_plots(labels_df, preds_df, study_dir, model_type)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained model on the test set.')
    parser.add_argument(
        '--config', '-c', 
        default='datapp_de',
        help='Configuration file name (default: datapp_de)'
    )
    parser.add_argument(
        '--model-type', '-m',
        required=True,
        choices=['ffnn', 'cnn'],
        help='The model architecture to evaluate.'
    )
    parser.add_argument(
        '--study-name', '-s',
        default='latest',
        help='Study directory to use for saving plots (default: latest for the given model-type).'
    )
    args = parser.parse_args()
    evaluate(
        config_name=args.config, 
        model_type=args.model_type, 
        study_name=args.study_name
    )