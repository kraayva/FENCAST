# scripts/run_training.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import optuna
import json

# Import our custom modules
from fencast.utils.paths import load_config, PROJECT_ROOT
from fencast.dataset import FencastDataset
from fencast.models import DynamicFFNN 
from fencast.utils.tools import setup_logger

logger = setup_logger("training")

def run_training(
    learning_rate: float,
    batch_size: int,
    epochs: int,
    hidden_layers: list,
    activation_name: str,
    dropout_rate: float,
    input_size: int,
    output_size: int
):
    """
    Main function to orchestrate the model training and validation process.
    """
    # 1. SETUP
    # =================================================================================
    logger.info("--- 1. Setting up experiment ---")
    config = load_config("datapp_de")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 2. DATA LOADING
    # =================================================================================
    logger.info("--- 2. Loading data ---")
    train_dataset = FencastDataset(config=config, mode='train')
    validation_dataset = FencastDataset(config=config, mode='validation')
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)

    # 3. MODEL, LOSS, and OPTIMIZER
    # =================================================================================
    logger.info("--- 3. Initializing model, loss, and optimizer ---")

    # Create a dictionary of model arguments for saving
    model_args = {
        'input_size': input_size,
        'output_size': output_size,
        'hidden_layers': hidden_layers,
        'dropout_rate': dropout_rate,
        'activation_fn': getattr(nn, activation_name)()
    }
    
    # Use the DynamicFFNN with the specified arguments
    model = DynamicFFNN(**model_args).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 4. TRAINING LOOP
    # =================================================================================
    logger.info("--- 4. Starting training ---")

    best_validation_loss = float('inf')
    model_save_path = PROJECT_ROOT / "model" / f"{config['setup_name']}_best_model.pth"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            max_norm = config.get('training', {}).get('gradient_clip_max_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
        avg_train_loss = np.mean(train_losses)

        model.eval()
        validation_losses = []
        with torch.no_grad():
            for features, labels in validation_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                validation_losses.append(loss.item())
        avg_validation_loss = np.mean(validation_losses)

        # --- Save the best model checkpoint ---
        if avg_validation_loss < best_validation_loss:
            best_validation_loss = avg_validation_loss
            
            # Save the complete checkpoint dictionary
            torch.save({
                'model_args': model_args,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_save_path)
            
            logger.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_validation_loss:.6f} (saved)")
        else:
            logger.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_validation_loss:.6f}")

    logger.info("\n--- Training finished ---")
    logger.info(f"Best model validation loss: {best_validation_loss:.6f}")
    logger.info(f"Model saved to: {model_save_path}")


if __name__ == '__main__':
    # Load best hyperparameters from the latest tuning results
    logger.info("--- Loading best hyperparameters from the latest study ---")
    config = load_config("datapp_de")
    setup_name = config.get('setup_name')
    results_parent_dir = PROJECT_ROOT / "results" / setup_name
    latest_study_dir = sorted(results_parent_dir.iterdir(), key=lambda f: f.stat().st_mtime, reverse=True)[0]
    study_name = latest_study_dir.name
    storage_name = f"sqlite:///{latest_study_dir / study_name}.db"
    
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    best_params = study.best_trial.params

    # Reconstruct the hidden_layers list from the format Optuna saved
    best_hidden_layers = [best_params[f"n_units_l{i}"] for i in range(best_params["n_layers"])]

    # Nicely log the parameters we're about to use
    final_params = {
        "learning_rate": best_params["lr"],
        "hidden_layers": best_hidden_layers,
        "activation_name": best_params["activation"],
        "dropout_rate": best_params["dropout"]
    }
    logger.info(f"Best hyperparameters found:\n{json.dumps(final_params, indent=4)}")

    # Run final training with the best hyperparameters
    run_training(
        input_size=config['input_size_flat'],  # number of features
        output_size=config['target_size'],  # number of targets
        epochs=config.get('training', {}).get('final_epochs', 30), # Number of epochs from config
        learning_rate=final_params["learning_rate"],
        hidden_layers=final_params["hidden_layers"],
        activation_name=final_params["activation_name"],
        dropout_rate=final_params["dropout_rate"],
        batch_size=config.get('model', {}).get('batch_sizes', {}).get('training', 64)
    )