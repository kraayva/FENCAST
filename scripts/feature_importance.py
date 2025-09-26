# scripts/feature_importance.py

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import torch.nn as nn
import argparse
import joblib

# Import our custom modules
from fencast.utils.paths import load_config, PROJECT_ROOT, PROCESSED_DATA_DIR
from fencast.dataset import FencastDataset
from fencast.models import DynamicFFNN, DynamicCNN
from fencast.utils.tools import setup_logger

logger = setup_logger("feature_importance")

def calculate_rmse(model, data_loader, device):
    """Calculates the RMSE for a given model and data loader."""
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
    return np.sqrt(mean_squared_error(all_labels, all_predictions))

def get_cnn_channel_map(config: dict) -> dict:
    """Creates a map from a CNN channel index to its (variable, level) identity."""
    var_names = list(config['era5_var_names'].values())
    levels = config['feature_level']
    
    channel_map = {}
    channel_idx = 0
    for var in var_names:
        for level in levels:
            channel_map[channel_idx] = {'var': var, 'level': level}
            channel_idx += 1
    return channel_map

def run_feature_importance(config_name: str, model_type: str):
    """Performs grouped permutation feature importance analysis for a given model type."""
    logger.info(f"--- Starting Feature Importance Analysis for '{model_type}' model ---")
    
    # 1. SETUP & LOAD DATA/MODEL
    # =================================================================================
    config = load_config(config_name)
    setup_name = config.get('setup_name', 'default_setup')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading best trained model...")
    final_model_path = PROJECT_ROOT / "model" / f"{setup_name}_{model_type}_best_model.pth"
    if not final_model_path.exists():
        raise FileNotFoundError(f"Model file not found at {final_model_path}")
    
    checkpoint = torch.load(final_model_path, map_location=device)
    
    # Dynamically instantiate the correct model
    if checkpoint.get('model_type') == 'cnn' or model_type == 'cnn':
        model = DynamicCNN(**checkpoint['model_args']).to(device)
    else:
        model = DynamicFFNN(**checkpoint['model_args']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    logger.info(f"Loading test set data for '{model_type}' (un-normalized)...")
    dataset_obj = FencastDataset(config=config, mode='test', model_type=model_type, apply_normalization=False)
    X_processed, y_processed = dataset_obj.X, dataset_obj.y
    
    # 2. CALCULATE BASELINE SCORE & DEFINE GROUPS
    # =================================================================================
    all_importances = {}
    
    if model_type == 'ffnn':
        # --- FFNN PATH ---
        scaler_path = PROCESSED_DATA_DIR / f"{setup_name}_ffnn_scaler.gz"
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X_processed)
        
        # Baseline
        temp_dataset = torch.utils.data.TensorDataset(torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y_processed.values, dtype=torch.float32))
        temp_loader = DataLoader(temp_dataset, batch_size=256)
        baseline_rmse = calculate_rmse(model, temp_loader, device)
        logger.info(f"Baseline RMSE on test set: {baseline_rmse:.6f}")

        # Feature Groups by parsing column names
        var_names = list(config['era5_var_names'].values())
        var_groups = {var: [col for col in X_processed.columns if col.startswith(f'{var}_')] for var in var_names}
        levels = sorted(list(set([col.split('_')[1] for col in X_processed.columns if col.count('_') > 1])))
        level_groups = {f'{level} hPa': [col for col in X_processed.columns if f'_{level}_' in col] for level in levels}
        
        feature_groups_to_test = {"By Variable": var_groups, "By Level": level_groups}

        # Permutation Importance Loop for FFNN
        for group_type, feature_groups in feature_groups_to_test.items():
            logger.info(f"\n--- Calculating Importance {group_type} ---")
            importances = {}
            for group_name, columns in feature_groups.items():
                if not columns: continue
                
                X_permuted = X_processed.copy()
                # Permute by shuffling the values within the selected columns
                perm_indices = np.random.permutation(X_permuted.index)
                X_permuted[columns] = X_permuted.loc[perm_indices, columns].values

                X_permuted_scaled = scaler.transform(X_permuted)
                
                permuted_dataset = torch.utils.data.TensorDataset(torch.tensor(X_permuted_scaled, dtype=torch.float32), torch.tensor(y_processed.values, dtype=torch.float32))
                permuted_loader = DataLoader(permuted_dataset, batch_size=256)
                
                permuted_rmse = calculate_rmse(model, permuted_loader, device)
                importance = permuted_rmse - baseline_rmse
                importances[group_name] = importance
                logger.info(f"  Importance of '{group_name}': {importance:.6f}")
            all_importances[group_type] = importances

    elif model_type == 'cnn':
        # --- CNN PATH ---
        scaler_path = PROCESSED_DATA_DIR / f"{setup_name}_cnn_scaler.npz"
        with np.load(scaler_path) as data:
            mean, std = data['mean'], data['std']
        X_scaled = (X_processed - mean) / std

        # Baseline
        temp_dataset = torch.utils.data.TensorDataset(torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y_processed.values, dtype=torch.float32))
        temp_loader = DataLoader(temp_dataset, batch_size=256)
        baseline_rmse = calculate_rmse(model, temp_loader, device)
        logger.info(f"Baseline RMSE on test set: {baseline_rmse:.6f}")

        # Feature Groups by mapping channel indices
        channel_map = get_cnn_channel_map(config)
        var_groups = {var: [idx for idx, info in channel_map.items() if info['var'] == var] for var in config['era5_var_names'].values()}
        level_groups = {f"{level} hPa": [idx for idx, info in channel_map.items() if info['level'] == level] for level in config['feature_level']}
        
        feature_groups_to_test = {"By Variable": var_groups, "By Level": level_groups}

        # Permutation Importance Loop for CNN
        for group_type, feature_groups in feature_groups_to_test.items():
            logger.info(f"\n--- Calculating Importance {group_type} ---")
            importances = {}
            for group_name, channel_indices in feature_groups.items():
                if not channel_indices: continue
                
                X_permuted = X_processed.copy()
                n_samples = X_permuted.shape[0]
                perm_indices = np.random.permutation(n_samples)
                
                # Permute by shuffling samples *within* the selected channels
                X_permuted[:, channel_indices, :, :] = X_permuted[perm_indices][:, channel_indices, :, :]
                
                X_permuted_scaled = (X_permuted - mean) / std
                
                permuted_dataset = torch.utils.data.TensorDataset(torch.tensor(X_permuted_scaled, dtype=torch.float32), torch.tensor(y_processed.values, dtype=torch.float32))
                permuted_loader = DataLoader(permuted_dataset, batch_size=256)

                permuted_rmse = calculate_rmse(model, permuted_loader, device)
                importance = permuted_rmse - baseline_rmse
                importances[group_name] = importance
                logger.info(f"  Importance of '{group_name}': {importance:.6f}")
            all_importances[group_type] = importances

    # 3. VISUALIZE RESULTS
    # =================================================================================
    for group_type, importances in all_importances.items():
        if not importances: continue
        sorted_importances = sorted(importances.items(), key=lambda item: item[1], reverse=True)
        names = [item[0] for item in sorted_importances]
        scores = [item[1] for item in sorted_importances]

        plt.figure(figsize=(10, 8))
        plt.barh(names, scores)
        plt.xlabel("Increase in RMSE (Importance)")
        plt.title(f"Permutation Feature Importance {group_type} for {model_type.upper()} Model")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        save_path = PROJECT_ROOT / "results" / f"feature_importance_{model_type}_{group_type.replace(' ', '_').lower()}.png"
        plt.savefig(save_path)
        logger.info(f"Saved {group_type} importance plot to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run permutation feature importance analysis.')
    parser.add_argument(
        '--config', '-c', 
        default='datapp_de',
        help='Configuration file name (default: datapp_de)'
    )
    parser.add_argument(
        '--model-type', '-m',
        required=True,
        choices=['ffnn', 'cnn'],
        help='The model architecture to analyze.'
    )
    args = parser.parse_args()
    run_feature_importance(config_name=args.config, model_type=args.model_type)