# scripts/feature_importance.py

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import torch.nn as nn

# Import our custom modules
from fencast.utils.paths import load_config, PROJECT_ROOT
from fencast.dataset import FencastDataset
from fencast.models import DynamicFFNN
from fencast.utils.tools import setup_logger

logger = setup_logger("feature_importance")

def calculate_rmse(model, data_loader, device):
    """Calculates the RMSE for a given model and data loader."""
    # (This function is correct and needs no changes)
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

def run_feature_importance():
    """Performs grouped permutation feature importance analysis."""
    logger.info("--- Starting Feature Importance Analysis ---")
    
    # 1. SETUP & LOAD DATA/MODEL
    config = load_config("datapp_de")
    setup_name = config.get('setup_name', 'default_setup')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading test set data (un-normalized)...")
    dataset_obj = FencastDataset(config=config, mode='test', apply_normalization=False)
    X_processed, y_processed = dataset_obj.get_data()
    
    logger.info("Loading best trained model...")
    final_model_path = PROJECT_ROOT / "model" / f"{setup_name}_best_model.pth"
    checkpoint = torch.load(final_model_path, map_location=device, weights_only=False)
    model = DynamicFFNN(**checkpoint['model_args']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    scaler = FencastDataset(config=config, mode='test').scaler
    
    # 2. CALCULATE BASELINE SCORE
    X_scaled = scaler.transform(X_processed)
    temp_dataset = torch.utils.data.TensorDataset(torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y_processed.values, dtype=torch.float32))
    temp_loader = DataLoader(temp_dataset, batch_size=256)
    baseline_rmse = calculate_rmse(model, temp_loader, device)
    logger.info(f"Baseline RMSE on test set: {baseline_rmse:.6f}")

    # 3. DEFINE FEATURE GROUPS
    # Group by physical variable (using the full name, not the short code)
    var_groups = {var_name: [col for col in X_processed.columns if col.startswith(f'{var_name}_')] 
                  for var_char, var_name in config['era5_var_names'].items()}

    # Group by pressure level
    levels = sorted(list(set([col.split('_')[1] for col in X_processed.columns])))
    level_groups = {f'{level} hPa': [col for col in X_processed.columns if col.split('_')[1] == level] 
                    for level in levels}

    # 4. PERFORM PERMUTATION IMPORTANCE
    all_importances = {}
    feature_groups_to_test = {"By Variable": var_groups, "By Level": level_groups}

    for group_type, feature_groups in feature_groups_to_test.items():
        logger.info(f"\n--- Calculating Importance {group_type} ---")
        importances = {}
        for group_name, columns in feature_groups.items():
            if not columns: # Add a check for empty column lists
                logger.warning(f"  Skipping '{group_name}' as no columns were found.")
                continue
            
            X_permuted = X_processed.copy()
            X_permuted[columns] = X_permuted[columns].sample(frac=1, replace=False, axis=0).values
            X_permuted_scaled = scaler.transform(X_permuted)
            
            permuted_dataset = torch.utils.data.TensorDataset(torch.tensor(X_permuted_scaled, dtype=torch.float32), torch.tensor(y_processed.values, dtype=torch.float32))
            permuted_loader = DataLoader(permuted_dataset, batch_size=256)
            
            permuted_rmse = calculate_rmse(model, permuted_loader, device)
            importance = permuted_rmse - baseline_rmse
            importances[group_name] = importance
            logger.info(f"  Importance of '{group_name}': {importance:.6f} (RMSE increased to {permuted_rmse:.6f})")
        
        all_importances[group_type] = importances

    # 5. VISUALIZE RESULTS
    # (This section is correct and needs no changes)
    for group_type, importances in all_importances.items():
        if not importances: continue # Skip plotting if no importances were calculated
        sorted_importances = sorted(importances.items(), key=lambda item: item[1], reverse=True)
        names = [item[0] for item in sorted_importances]
        scores = [item[1] for item in sorted_importances]

        plt.figure(figsize=(10, 6))
        plt.bar(names, scores)
        plt.ylabel("Increase in RMSE (Importance)")
        plt.title(f"Feature Importance {group_type}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        save_path = PROJECT_ROOT / "results" / f"feature_importance_{group_type.replace(' ', '_').lower()}.png"
        plt.savefig(save_path)
        logger.info(f"Saved {group_type} importance plot to {save_path}")

if __name__ == '__main__':
    run_feature_importance()