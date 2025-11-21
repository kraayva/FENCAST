import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import torch.nn as nn
import argparse
import seaborn as sns
import joblib
from pathlib import Path

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
                spatial_features, temporal_features, labels = batch
                spatial_features, temporal_features = spatial_features.to(device), temporal_features.to(device)
                outputs = model(spatial_features, temporal_features)
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
    var_names = list(config['feature_var_names'].values())
    levels = config['feature_level']
    channel_map = {}
    channel_idx = 0
    for var in var_names:
        for level in levels:
            channel_map[channel_idx] = {'var': var, 'level': level}
            channel_idx += 1
    return channel_map

def run_feature_importance(config_name: str, model_type: str, study_name: str):
    """Performs grouped permutation feature importance analysis for a given model type."""
    logger.info(f"--- Starting Feature Importance Analysis for '{model_type}' model ---")
    
    # 1. SETUP & LOAD MODEL/DATA
    config = load_config(config_name)
    setup_name = config.get('setup_name', 'default_setup')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results_parent_dir = PROJECT_ROOT / "results" / setup_name
    try:
        if study_name == 'latest':
            study_dir = get_latest_study_dir(results_parent_dir, model_type)
        else:
            study_dir = results_parent_dir / study_name
        logger.info(f"Using results from study directory: {study_dir.name}")
    except FileNotFoundError as e:
        logger.error(e)
        return

    logger.info("Loading best trained model...")
    final_model_path = study_dir / "best_model.pth"
    if not final_model_path.exists():
        raise FileNotFoundError(f"Model file not found at {final_model_path}")
    
    checkpoint = torch.load(final_model_path, map_location=device, weights_only=False)
    
    if checkpoint.get('model_type') == 'cnn' or model_type == 'cnn':
        model = DynamicCNN(**checkpoint['model_args']).to(device)
    else:
        model = DynamicFFNN(**checkpoint['model_args']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    logger.info(f"Loading test set data for '{model_type}' (un-normalized)...")
    dataset_obj = FencastDataset(config=config, mode='test', model_type=model_type, apply_normalization=False)
    X_processed, y_processed = dataset_obj.X, dataset_obj.y
    
    # --- SECTION: 2. CALCULATE GROUPED IMPORTANCE FOR BAR CHARTS
    # =================================================================================
    all_importances = {}
    
    if model_type == 'ffnn':
        scaler_path = PROCESSED_DATA_DIR / f"{setup_name}_ffnn_scaler.gz"
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X_processed)
        
        temp_dataset = torch.utils.data.TensorDataset(torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y_processed.values, dtype=torch.float32))
        temp_loader = DataLoader(temp_dataset, batch_size=256)
        baseline_rmse = calculate_rmse(model, temp_loader, device, model_type='ffnn')
        logger.info(f"Baseline RMSE on test set: {baseline_rmse:.6f}")
        
        var_names = list(config['feature_var_names'].values())
        var_groups = {var: [col for col in X_processed.columns if col.startswith(f'{var}_')] for var in var_names}
        # Combine u and v into a single 'Wind' group
        var_groups['Wind'] = var_groups['u'] + var_groups['v']
        del var_groups['u']
        del var_groups['v']
        var_groups['Temporal Features'] = ['day_of_year_sin', 'day_of_year_cos'] # Add temporal to var_groups

        levels = sorted(list(set([col.split('_')[1] for col in X_processed.columns if col.count('_') > 1])))
        level_groups = {f'{level} hPa': [col for col in X_processed.columns if f'_{level}_' in col] for level in levels}

        feature_groups_to_test = {
            "By Variable": var_groups, 
            "By Level": level_groups
        }

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
                
                permuted_rmse = calculate_rmse(model, permuted_loader, device, model_type='ffnn')
                importance = permuted_rmse - baseline_rmse
                importances[group_name] = importance
                logger.info(f"  Importance of '{group_name}': {importance:.6f}")
            all_importances[group_type] = importances

    elif model_type == 'cnn':
        scaler_path = PROCESSED_DATA_DIR / f"{setup_name}_cnn_scaler.npz"
        with np.load(scaler_path) as data:
            mean, std = data['mean'], data['std']
        X_scaled = (X_processed - mean) / std
        
        temporal_features = dataset_obj.temporal_features

        temp_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_scaled, dtype=torch.float32), 
            torch.tensor(temporal_features.values, dtype=torch.float32),
            torch.tensor(y_processed.values, dtype=torch.float32)
        )
        temp_loader = DataLoader(temp_dataset, batch_size=256)
        baseline_rmse = calculate_rmse(model, temp_loader, device, model_type='cnn')
        logger.info(f"Baseline RMSE on test set: {baseline_rmse:.6f}")
        
        channel_map = get_cnn_channel_map(config)
        var_groups = {var: [idx for idx, info in channel_map.items() if info['var'] == var] for var in config['feature_var_names'].values()}
        # Combine u and v into a single 'Wind' group
        var_groups['Wind'] = var_groups['u'] + var_groups['v']
        del var_groups['u']
        del var_groups['v']
        
        level_groups = {f"{level} hPa": [idx for idx, info in channel_map.items() if info['level'] == level] for level in config['feature_level']}

        feature_groups_to_test = {
            "By Variable": var_groups, 
            "By Level": level_groups
        }

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
            
            # Add temporal features to the "By Variable" group only
            if group_type == "By Variable":
                logger.info("  Calculating temporal feature importance...")
                permuted_temporal = temporal_features.sample(frac=1).values
                permuted_dataset = torch.utils.data.TensorDataset(
                    torch.tensor(X_scaled, dtype=torch.float32), 
                    torch.tensor(permuted_temporal, dtype=torch.float32),
                    torch.tensor(y_processed.values, dtype=torch.float32)
                )
                permuted_loader = DataLoader(permuted_dataset, batch_size=256)
                permuted_rmse = calculate_rmse(model, permuted_loader, device, model_type='cnn')
                temporal_importance = permuted_rmse - baseline_rmse
                importances["Day of Year"] = temporal_importance
                logger.info(f"  Importance of 'Day of Year': {temporal_importance:.6f}")
            
            all_importances[group_type] = importances
    # --- SECTION 3: HEATMAP CALCULATION ---
    # =================================================================================
    logger.info("\n--- Calculating Importance for Heatmap ---")
    
    physical_vars = {k: v for k, v in config['feature_var_names'].items() if v not in ['u', 'v']}
    physical_vars['Wind'] = ['u', 'v'] # Group u and v as 'Wind'
    levels = config['feature_level']
    
    # Prepare a DataFrame to store heatmap scores
    heatmap_df = pd.DataFrame(index=levels, columns=list(physical_vars.keys()))

    for var_name, var_codes in physical_vars.items():
        for level in levels:
            group_name = f"{var_name} at {level} hPa"
            
            if model_type == 'ffnn':
                # Find all columns corresponding to this var/level combo
                columns_to_permute = [
                    col for col in X_processed.columns 
                    if any(f'{code}_{level}' in col for code in var_codes)
                ]
                if not columns_to_permute: continue
                
                X_permuted = X_processed.copy()
                perm_indices = np.random.permutation(X_permuted.index)
                X_permuted[columns_to_permute] = X_permuted.loc[perm_indices, columns_to_permute].values
                X_permuted_scaled = scaler.transform(X_permuted)
                
                permuted_dataset = torch.utils.data.TensorDataset(torch.tensor(X_permuted_scaled, dtype=torch.float32), torch.tensor(y_processed.values, dtype=torch.float32))

            elif model_type == 'cnn':
                # Find all channel indices for this var/level combo
                channels_to_permute = [
                    idx for idx, info in channel_map.items()
                    if info['var'] in var_codes and info['level'] == level
                ]
                if not channels_to_permute: continue

                X_permuted = X_processed.copy()
                n_samples = X_permuted.shape[0]
                perm_indices = np.random.permutation(n_samples)
                X_permuted[:, channels_to_permute, :, :] = X_permuted[perm_indices][:, channels_to_permute, :, :]
                X_permuted_scaled = (X_permuted - mean) / std

                permuted_dataset = torch.utils.data.TensorDataset(
                    torch.tensor(X_permuted_scaled, dtype=torch.float32), 
                    torch.tensor(temporal_features.values, dtype=torch.float32),
                    torch.tensor(y_processed.values, dtype=torch.float32)
                )

            permuted_loader = DataLoader(permuted_dataset, batch_size=256)
            permuted_rmse = calculate_rmse(model, permuted_loader, device, model_type)
            importance = permuted_rmse - baseline_rmse
            heatmap_df.loc[level, var_name] = importance
            logger.info(f"  Importance of '{group_name}': {importance:.6f}")

    # ---  SECTION 4: VISUALIZE RESULTS ---
    # =================================================================================
    
    # ---  Heatmap Visualization ---
    plt.figure(figsize=(12, 7))
    sns.heatmap(
        heatmap_df.astype(float), 
        annot=True,          # Write the data value in each cell
        fmt=".4f",           # Format numbers to 4 decimal places
        cmap="viridis",      # Color scheme
        linewidths=.5
    )
    plt.title(f"Feature Importance Heatmap for {model_type.upper()} Model")
    plt.ylabel("Pressure Level (hPa)")
    plt.xlabel("Physical Variable")
    plt.tight_layout()
    save_path = study_dir / f"feature_importance_{model_type}_heatmap.png"
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved heatmap plot to {save_path}")

    # --- Bar Chart Visualizations ---
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
        save_path = study_dir / f"feature_importance_{model_type}_{group_type.replace(' ', '_').lower()}.png"
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved {group_type} importance plot to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run permutation feature importance analysis.')
    parser.add_argument('--config', '-c', default='datapp_de', help='Configuration file name (default: datapp_de)')
    parser.add_argument('--model-type', '-m', default='cnn', help='The model architecture to analyze.')
    parser.add_argument('--study-name', '-s', default='latest', help='The study name to use for loading results.')
    args = parser.parse_args()
    
    run_feature_importance(
        config_name=args.config, 
        model_type=args.model_type, 
        study_name=args.study_name
    )