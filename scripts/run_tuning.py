# scripts/run_tuning.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import optuna

# Import our custom modules
from fencast.utils.paths import load_config
from fencast.dataset import FencastDataset
from fencast.models import DynamicFFNN

def objective(trial: optuna.Trial) -> float:
    
    # 1. SETUP & HYPERPARAMETER SUGGESTIONS
    # =================================================================================
    config = load_config("datapp_de")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float("dropout", 0.1, 0.5)
    n_layers = trial.suggest_int("n_layers", 1, 3)
    
    hidden_layers = []
    for i in range(n_layers):
        n_units = trial.suggest_int(f"n_units_l{i}", 256, 2048)
        hidden_layers.append(n_units)

    activation_name = trial.suggest_categorical("activation", ["ReLU", "ELU"])
    activation_fn = getattr(nn, activation_name)()

    INPUT_SIZE = 20295
    OUTPUT_SIZE = 37
    BATCH_SIZE = 64
    EPOCHS = 20

    # 2. DATA LOADING
    # =================================================================================
    train_dataset = FencastDataset(config=config, mode='train')
    validation_dataset = FencastDataset(config=config, mode='validation')
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. MODEL, LOSS, and OPTIMIZER
    # =================================================================================
    model = DynamicFFNN(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
        activation_fn=activation_fn
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 4. TRAINING & VALIDATION LOOP
    # =================================================================================
    best_validation_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        validation_losses = []
        with torch.no_grad():
            for features, labels in validation_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                validation_losses.append(loss.item())
        
        avg_validation_loss = np.mean(validation_losses)

        if avg_validation_loss < best_validation_loss:
            best_validation_loss = avg_validation_loss
            
        # 1. Report the intermediate validation loss to Optuna
        trial.report(avg_validation_loss, epoch)

        # 2. Check if the trial should be pruned
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_validation_loss


if __name__ == '__main__':
    # --- NEW: Add a pruner to the study ---
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction='minimize', pruner=pruner)
    
    study.optimize(objective, n_trials=50)
    
    print("\n--- Tuning Finished ---")
    print("Best trial:")
    trial = study.best_trial
    
    print(f"  Value (Best Validation Loss): {trial.value:.6f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")