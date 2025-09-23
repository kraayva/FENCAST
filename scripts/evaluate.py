# scripts/evaluate.py

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import torch.nn as nn

# Import our custom modules
from fencast.utils.paths import load_config, PROJECT_ROOT
from fencast.dataset import FencastDataset
from fencast.models import DynamicFFNN
from fencast.utils.tools import setup_logger

logger = setup_logger("evaluation")

def get_predictions(model, data_loader, device):
    """Runs the model on the test set and returns predictions and labels."""
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            outputs = model(features)
            all_predictions.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_predictions, all_labels

def calculate_persistence_rmse(labels_df: pd.DataFrame):
    """Calculates the RMSE for the persistence (naive) baseline."""
    # Prediction for today is the value from yesterday
    persistence_preds = labels_df.shift(1)
    
    # Drop the first row
    valid_labels = labels_df.iloc[1:]
    valid_preds = persistence_preds.iloc[1:]
    
    return np.sqrt(mean_squared_error(valid_labels, valid_preds))

def create_plots(labels_df, preds_df, results_dir):
    """Creates and saves time-series and scatter plots."""
    # Time-Series Plot (for one region and a slice of time)
    region_to_plot = labels_df.columns[0]
    time_slice = slice('2023-05-01', '2023-05-31')

    plt.figure(figsize=(15, 6))
    plt.plot(labels_df.index[time_slice], labels_df[region_to_plot][time_slice], label='Actual Values', alpha=0.8)
    plt.plot(preds_df.index[time_slice], preds_df[region_to_plot][time_slice], label='NN Predictions', alpha=0.8, linestyle='--')
    plt.title(f'Predictions vs Actuals for {region_to_plot} - May 2023')
    plt.xlabel('Date')
    plt.ylabel('Capacity Factor')
    plt.legend()
    plt.grid(True)
    plt.savefig(results_dir / 'timeseries_plot.png')
    logger.info("Time-series plot saved.")

    # Scatter Plot
    plt.figure(figsize=(8, 8))
    # Use a sample to avoid overplotting
    sample_indices = np.random.choice(labels_df.size, size=5000, replace=False)
    plt.scatter(labels_df.values.flatten()[sample_indices], preds_df.values.flatten()[sample_indices], alpha=0.3, s=10)
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Prediction (y=x)')
    plt.title('Scatter Plot of Predictions vs Actuals')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(results_dir / 'scatter_plot.png')
    logger.info("Scatter plot saved.")

def evaluate():
    """Main evaluation function."""
    logger.info("--- Starting Final Model Evaluation ---")
    
    # 1. SETUP
    # =================================================================================
    config = load_config("datapp_de")
    setup_name = config.get('setup_name', 'default_setup')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Find the results directory from the latest tuning run
    results_parent_dir = PROJECT_ROOT / "results" / setup_name
    latest_study_dir = sorted(results_parent_dir.iterdir(), key=lambda f: f.stat().st_mtime, reverse=True)[0]
    study_name = latest_study_dir.name
    storage_name = f"sqlite:///{latest_study_dir / study_name}.db"
    model_path = latest_study_dir / f"{study_name}_best_model.pth" # This needs to be saved from training, not tuning
    
    logger.info(f"Loading best trial from study: {study_name}")
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    best_params = study.best_trial.params

    # 2. LOAD DATA (TEST SET)
    # =================================================================================
    logger.info("Loading test set data...")
    test_dataset = FencastDataset(config=config, mode='test')
    test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)
    
    # 3. LOAD BEST MODEL
    # =================================================================================
    logger.info("Loading best trained model...")
    # load best model
    final_model_path = PROJECT_ROOT / "model" / f"{setup_name}_best_model.pth"
    if not final_model_path.exists():
        logger.error(f"Final model not found at {final_model_path}")
        logger.error("Please run a final training with the best hyperparameters first.")
        return 
    # Load the model architecture and weights from the saved .pth file
    checkpoint = torch.load(final_model_path, map_location=device)
    model = DynamicFFNN(**checkpoint['model_args']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 4. GET PREDICTIONS
    # =================================================================================
    logger.info("Generating predictions on the test set...")
    predictions_np, labels_np = get_predictions(model, test_loader, device)

    # Convert to DataFrames for easier handling
    labels_df = pd.DataFrame(labels_np, index=test_dataset.y.index, columns=test_dataset.y.columns)
    preds_df = pd.DataFrame(predictions_np, index=test_dataset.y.index, columns=test_dataset.y.columns)
    
    # 5. CALCULATE METRICS
    # =================================================================================
    logger.info("Calculating performance metrics...")
    nn_rmse = np.sqrt(mean_squared_error(labels_df, preds_df))
    nn_mae = mean_absolute_error(labels_df, preds_df)
    
    persistence_rmse = calculate_persistence_rmse(labels_df)

    logger.info("\n--- Performance Summary ---")
    logger.info(f"  Persistence Model RMSE: {persistence_rmse:.6f}")
    logger.info(f"  Neural Network RMSE:    {nn_rmse:.6f}")
    logger.info(f"  Neural Network MAE:     {nn_mae:.6f}")
    logger.info("---------------------------")

    # 6. VISUALIZE RESULTS
    # =================================================================================
    logger.info("Creating visualizations...")
    create_plots(labels_df, preds_df, latest_study_dir)

if __name__ == '__main__':
    evaluate()