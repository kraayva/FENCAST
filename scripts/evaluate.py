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
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_predictions, all_labels

def calculate_persistence_rmse(labels_df: pd.DataFrame):
    """Calculates the RMSE for the persistence (naive) baseline."""
    persistence_preds = labels_df.shift(1)
    valid_labels = labels_df.iloc[1:]
    valid_preds = persistence_preds.iloc[1:]
    return np.sqrt(mean_squared_error(valid_labels, valid_preds))


def create_plots(labels_df, nn_preds_df, persistence_preds_df, results_dir, time_name='timeseries_plot.png', scatter_name='scatter_plot_nn.png'):
    """Creates and saves time-series and scatter plots for both models."""
    # --- Time-Series Plot (with Baseline) ---
    region_to_plot = labels_df.columns[0]
    
    if not labels_df.empty:
        start_date = labels_df.index.min()
        end_date = start_date + pd.Timedelta(days=30)
        labels_slice = labels_df.loc[start_date:end_date]
        nn_preds_slice = nn_preds_df.loc[start_date:end_date]
        persistence_preds_slice = persistence_preds_df.loc[start_date:end_date]
        plot_title = f'Predictions vs Actuals for {region_to_plot} - {start_date.strftime("%B %Y")}'
    else:
        labels_slice, nn_preds_slice, persistence_preds_slice = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        plot_title = f'Predictions vs Actuals for {region_to_plot} (No data available)'

    plt.figure(figsize=(15, 7))
    if not labels_slice.empty:
        plt.plot(labels_slice.index, labels_slice[region_to_plot], label='Actual Values', color='black', linewidth=2)
        plt.plot(nn_preds_slice.index, nn_preds_slice[region_to_plot], label='NN Predictions', color='blue', linestyle='--')
        plt.plot(persistence_preds_slice.index, persistence_preds_slice[region_to_plot], label='Persistence Baseline', color='green', linestyle=':')
    
    plt.title(plot_title)
    plt.xlabel('Date')
    plt.ylabel('Capacity Factor')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(results_dir / time_name)
    logger.info(f"Time-series plot saved as {time_name}.")

    # --- Scatter Plot for Neural Network ---
    plt.figure(figsize=(8, 8))
    sample_indices = np.random.choice(labels_df.size, size=min(5000, labels_df.size), replace=False)
    plt.scatter(labels_df.values.flatten()[sample_indices], nn_preds_df.values.flatten()[sample_indices], alpha=0.3, s=10)
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Prediction (y=x)')
    plt.title('Scatter Plot: NN Predictions vs Actuals')
    plt.xlabel('Actual Values'); plt.ylabel('Predicted Values')
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.grid(True); plt.legend(); plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(results_dir / scatter_name)
    logger.info(f"NN scatter plot saved as {scatter_name}.")

    # --- NEW: Scatter Plot for Persistence Baseline ---
    plt.figure(figsize=(8, 8))
    valid_labels = labels_df.iloc[1:]
    valid_preds = persistence_preds_df.iloc[1:]
    sample_indices = np.random.choice(valid_labels.size, size=min(5000, valid_labels.size), replace=False)
    plt.scatter(valid_labels.values.flatten()[sample_indices], valid_preds.values.flatten()[sample_indices], alpha=0.3, s=10, color='green')
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Prediction (y=x)')
    plt.title('Scatter Plot: Persistence Baseline vs Actuals')
    plt.xlabel('Actual Values'); plt.ylabel('Predicted Values (Yesterday\'s Actuals)')
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.grid(True); plt.legend(); plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(results_dir / 'scatter_plot_baseline.png') # Using a fixed name for this new plot
    logger.info("Persistence baseline scatter plot saved.")


def evaluate(config_name: str = "datapp_de", study_name: str = 'latest'):
    """Main evaluation function."""
    logger.info("--- Starting Final Model Evaluation ---")
    
    # 1. SETUP
    config = load_config(config_name)
    setup_name = config.get('setup_name', 'default_setup')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results_parent_dir = PROJECT_ROOT / "results" / setup_name
    if study_name == 'latest':
        logger.info("Locating the latest study directory...")
        study_dir = sorted(results_parent_dir.iterdir(), key=lambda f: f.stat().st_mtime, reverse=True)[0]
        study_name = study_dir.name
    else:
        study_dir = results_parent_dir / study_name
    
    logger.info(f"Loading results from study: {study_name}")
    
    # 2. LOAD DATA (TEST SET)
    logger.info("Loading test set data...")
    test_dataset = FencastDataset(config=config, mode='test')
    batch_size = config.get('model', {}).get('batch_sizes', {}).get('evaluation', 256)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. LOAD BEST MODEL
    logger.info("Loading best trained model...")
    final_model_path = PROJECT_ROOT / "model" / f"{setup_name}_best_model.pth"
    if not final_model_path.exists():
        logger.error(f"Final model not found at {final_model_path}. Please run a final training run.")
        return 
    checkpoint = torch.load(final_model_path, map_location=device, weights_only=False)
    model = DynamicFFNN(**checkpoint['model_args']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 4. GET PREDICTIONS
    logger.info("Generating predictions on the test set...")
    predictions_np, labels_np = get_predictions(model, test_loader, device)
    labels_df = pd.DataFrame(labels_np, index=test_dataset.y.index, columns=test_dataset.y.columns)
    preds_df = pd.DataFrame(predictions_np, index=test_dataset.y.index, columns=test_dataset.y.columns)
    
    # Generate persistence predictions
    persistence_preds_df = labels_df.shift(1)
    
    # 5. CALCULATE METRICS
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
    logger.info("Creating visualizations...")
    create_plots(labels_df, preds_df, persistence_preds_df, study_dir)


if __name__ == '__main__':
    evaluate()