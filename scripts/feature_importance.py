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
from fencast.utils.tools import setup_logger, get_latest_study_dir

logger = setup_logger("feature_importance")

def calculate_rmse(model, data_loader, device, model_type: str):
    """Calculates the RMSE for a given model and data loader."""
    model.eval()
    all_predictions, all_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            if model_type == 'cnn':
                # For CNN, unpack three tensors (or two if temporal is missing)
                if len(batch) == 3:
                    spatial_features, temporal_features, labels = batch
                    spatial_features, temporal_features = spatial_features.to(device), temporal_features.to(device)
                    outputs = model(spatial_features, temporal_features)
                else: # Fallback for TensorDataset without temporal
                    spatial_features, labels = batch
                    raise ValueError("CNN evaluation requires temporal features which are missing from the DataLoader.")

            else: # FFNN
                features, labels = batch
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

def run_feature_importance(config_name: str, model_type: str, setup_name: str, study_name: str):
    """Performs grouped permutation feature importance analysis for a given model type."""
    logger.info(f"--- Starting Feature Importance Analysis for '{model_type}' model ---")
    # =====================================
    # 1. SETUP & LOAD DATA/MODEL
    # =====================================
    config = load_config(config_name)
    setup_name = config.get('setup_name', 'default_setup')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if study_name == 'latest':
        study_dir = get_latest_study_dir(PROJECT_ROOT / "results", model_type)
        logger.info("Loading best trained model...")
    else:
        study_dir = PROJECT_ROOT / "results" / study_name
        if not study_dir.exists():
            raise FileNotFoundError(f"Study directory '{study_dir}' does not exist.")
        logger.info(f"Loading best trained model from specified study '{study_name}'...")
        
    final_model_path = study_dir / "best_model.pth"
    if not final_model_path.exists():
        raise FileNotFoundError(f"Model file not found at {final_model_path}")
    
    checkpoint = torch.load(final_model_path, map_location=device)
    
    if checkpoint.get('model_type') == 'cnn' or model_type == 'cnn':
        model = DynamicCNN(**checkpoint['model_args']).to(device)
    else:
        model = DynamicFFNN(**checkpoint['model_args']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    logger.info(f"Loading test set data for '{model_type}' (un-normalized)...")
    dataset_obj = FencastDataset(config=config, mode='test', model_type=model_type, apply_normalization=False)
    X_processed, y_processed = dataset_obj.X, dataset_obj.y

    # =============================================
    # 2. PERMUTATION IMPORTANCE CALCULATION
    # =============================================

    all_importances = {}

    # --- 2.a FFNN MODEL ---
    if model_type == 'ffnn':
        scaler_path = PROCESSED_DATA_DIR / f"{setup_name}_ffnn_scaler.gz"
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X_processed)
        
        temp_dataset = torch.utils.data.TensorDataset(torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y_processed.values, dtype=torch.float32))
        temp_loader = DataLoader(temp_dataset, batch_size=256)
        baseline_rmse = calculate_rmse(model, temp_loader, device, model_type='ffnn') # Pass model_type
        logger.info(f"Baseline RMSE on test set: {baseline_rmse:.6f}")
        
        var_names = list(config['era5_var_names'].values())
        var_groups = {var: [col for col in X_processed.columns if col.startswith(f'{var}_')] for var in var_names}
        levels = sorted(list(set([col.split('_')[1] for col in X_processed.columns if col.count('_') > 1])))
        level_groups = {f'{level} hPa': [col for col in X_processed.columns if f'_{level}_' in col] for level in levels}
        feature_groups_to_test = {"By Variable": var_groups, "By Level": level_groups}

        for group_type, feature_groups in feature_groups_to_test.items():
            logger.info(f"\n--- Calculating Importance {group_type} ---")
            importances = {}
            for group_name, columns in feature_groups.items():
                if not columns: continue
                X_permuted = X_processed.copy()
                perm_indices = np.random.permutation(X_permuted.index)
                X_permuted[columns] = X_permuted.loc[perm_indices, columns].values
                X_permuted_scaled = scaler.transform(X_permuted)
                
                permuted_dataset = torch.utils.data.TensorDataset(torch.tensor(X_permuted_scaled, dtype=torch.float32), torch.tensor(y_processed.values, dtype=torch.float32))
                permuted_loader = DataLoader(permuted_dataset, batch_size=256)
                
                permuted_rmse = calculate_rmse(model, permuted_loader, device, model_type='ffnn') # Pass model_type
                importance = permuted_rmse - baseline_rmse
                importances[group_name] = importance
                logger.info(f"  Importance of '{group_name}': {importance:.6f}")
            all_importances[group_type] = importances

    # --- 2.b CNN MODEL ---
    elif model_type == 'cnn':
        scaler_path = PROCESSED_DATA_DIR / f"{setup_name}_cnn_scaler.npz"
        with np.load(scaler_path) as data:
            mean, std = data['mean'], data['std']
        X_scaled = (X_processed - mean) / std
        
        # get the temporal features from the dataset object
        temporal_features = dataset_obj.temporal_features

        # Baseline: Create TensorDataset with required inputs
        temp_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_scaled, dtype=torch.float32), 
            torch.tensor(temporal_features.values, dtype=torch.float32),
            torch.tensor(y_processed.values, dtype=torch.float32)
        )
        temp_loader = DataLoader(temp_dataset, batch_size=256)
        baseline_rmse = calculate_rmse(model, temp_loader, device, model_type='cnn')
        logger.info(f"Baseline RMSE on test set: {baseline_rmse:.6f}")

        channel_map = get_cnn_channel_map(config)
        var_groups = {var: [idx for idx, info in channel_map.items() if info['var'] == var] for var in config['era5_var_names'].values()}
        level_groups = {f"{level} hPa": [idx for idx, info in channel_map.items() if info['level'] == level] for level in config['feature_level']}
        feature_groups_to_test = {"By Variable": var_groups, "By Level": level_groups}

        for group_type, feature_groups in feature_groups_to_test.items():
            logger.info(f"\n--- Calculating Importance {group_type} ---")
            importances = {}
            for group_name, channel_indices in feature_groups.items():
                if not channel_indices: continue
                
                X_permuted = X_processed.copy()
                n_samples = X_permuted.shape[0]
                perm_indices = np.random.permutation(n_samples)
                X_permuted[:, channel_indices, :, :] = X_permuted[perm_indices][:, channel_indices, :, :]
                X_permuted_scaled = (X_permuted - mean) / std
                
                # Create permuted dataset with required inputs
                permuted_dataset = torch.utils.data.TensorDataset(
                    torch.tensor(X_permuted_scaled, dtype=torch.float32), 
                    torch.tensor(temporal_features.values, dtype=torch.float32),
                    torch.tensor(y_processed.values, dtype=torch.float32)
                )
                permuted_loader = DataLoader(permuted_dataset, batch_size=256)

                permuted_rmse = calculate_rmse(model, permuted_loader, device, model_type='cnn')
                importance = permuted_rmse - baseline_rmse
                importances[group_name] = importance
                logger.info(f"  Importance of '{group_name}': {importance:.6f}")
            all_importances[group_type] = importances

    # ===============================
    # 3. VISUALIZE RESULTS
    # ===============================
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
        save_path = PROJECT_ROOT / "results" / setup_name / f"feature_importance_{model_type}_{group_type.replace(' ', '_').lower()}.png"
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved {group_type} importance plot to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run permutation feature importance analysis.')
    parser.add_argument('--config', '-c', default='datapp_de', help='Configuration file name (default: datapp_de)')
    parser.add_argument('--model-type', '-m', required=True, choices=['ffnn', 'cnn'], help='The model architecture to analyze.')
    parser.add_argument('--study', '-s', default='latest', help='The study name to use for loading results.')
    args = parser.parse_args()
    run_feature_importance(config_name=args.config, model_type=args.model_type, setup_name=args.setup, study_name=args.study_name)