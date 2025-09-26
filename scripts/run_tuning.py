# scripts/run_tuning.py
#%%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import optuna
from datetime import datetime
import json

# Import our custom modules
from fencast.utils.paths import load_config, PROJECT_ROOT
from fencast.dataset import FencastDataset
from fencast.models import DynamicFFNN
from fencast.utils.tools import setup_logger
#%%

# Setup logger once at the start of the script
logger = setup_logger("hyperparameter_tuning")

def objective(trial: optuna.Trial) -> float:
    
    # 1. SETUP & HYPERPARAMETER SUGGESTIONS
    # =================================================================================
    logger.info(f"--- Starting Trial {trial.number} ---")
    
    config = load_config("datapp_de")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get tuning configuration
    tuning_config = config.get('tuning', {})
    lr_config = tuning_config.get('learning_rate', {})
    dropout_config = tuning_config.get('dropout', {})
    layers_config = tuning_config.get('hidden_layers', {})
    
    # Suggest hyperparameters from config
    params = {
        'lr': trial.suggest_float("lr", 
                                lr_config.get('min', 1e-5), 
                                lr_config.get('max', 1e-2), 
                                log=lr_config.get('log_scale', True)),
        'dropout_rate': trial.suggest_float("dropout", 
                                          dropout_config.get('min', 0.1), 
                                          dropout_config.get('max', 0.5)),
        'n_layers': trial.suggest_int("n_layers", 
                                    layers_config.get('min_layers', 1), 
                                    layers_config.get('max_layers', 3)),
        'activation_name': trial.suggest_categorical("activation", 
                                                   tuning_config.get('activations', ["ReLU", "ELU"]))
    }
    
    hidden_layers = []
    for i in range(params['n_layers']):
        n_units = trial.suggest_int(f"n_units_l{i}", 
                                  layers_config.get('min_units', 256), 
                                  layers_config.get('max_units', 2048))
        hidden_layers.append(n_units)
    params['hidden_layers'] = hidden_layers
    
    # Log the chosen hyperparameters for this trial
    logger.info(f"Trial {trial.number} Parameters: {json.dumps(params, indent=4)}")

    INPUT_SIZE = config['input_size_flat'] 
    OUTPUT_SIZE = config['target_size']  
    BATCH_SIZE = config.get('model', {}).get('batch_sizes', {}).get('tuning', 64)
    EPOCHS = tuning_config.get('epochs', 20)

    # 2. DATA LOADING
    # =================================================================================
    logger.info(f"Trial {trial.number}: Loading data...")
    train_dataset = FencastDataset(config=config, mode='train')
    validation_dataset = FencastDataset(config=config, mode='validation')
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
    logger.info(f"Trial {trial.number}: Data loading complete.")

    # 3. MODEL, LOSS, and OPTIMIZER
    # =================================================================================
    model = DynamicFFNN(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        hidden_layers=params['hidden_layers'],
        dropout_rate=params['dropout_rate'],
        activation_fn=getattr(nn, params['activation_name'])()
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    # 4. TRAINING & VALIDATION LOOP
    # =================================================================================
    best_validation_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        # Training pass (omitted for brevity, no logging inside the tightest loop)
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
            
        trial.report(avg_validation_loss, epoch)

        if trial.should_prune():
            logger.warning(f"--- Trial {trial.number} pruned at epoch {epoch+1} ---")
            raise optuna.exceptions.TrialPruned()

    logger.info(f"--- Trial {trial.number} finished. Best validation loss: {best_validation_loss:.6f} ---")
    return best_validation_loss


if __name__ == '__main__':
    # ... (Setup logic remains the same)
    current_date = datetime.now().strftime('%Y%m%d')
    study_name = f"study_{current_date}"
    config = load_config("datapp_de")
    setup_name = config.get('setup_name', 'default_setup')
    results_dir = PROJECT_ROOT / "results" / setup_name / study_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting new study: '{study_name}'")
    logger.info(f"Study results will be saved in: {results_dir}")

    db_path = results_dir / f"{study_name}.db"
    storage_name = f"sqlite:///{db_path}"

    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction='minimize',
        pruner=pruner
    )
    
    n_trials = config.get('tuning', {}).get('trials', 50)
    study.optimize(objective, n_trials=n_trials, n_jobs=4)
    
    # --- 3. Log and Save Results ---
    logger.info("--- Tuning Finished ---")
    logger.info(f"Study statistics:")
    logger.info(f"  Number of finished trials: {len(study.trials)}")
    
    trial = study.best_trial
    logger.info("Best trial:")
    logger.info(f"  Value (Best Validation Loss): {trial.value:.6f}")
    
    # Nicely format the parameters for logging
    params_str = json.dumps(trial.params, indent=4)
    logger.info(f"  Params: \n{params_str}")

    # --- 4. Visualize and Save Plots ---
    logger.info("--- Generating and saving visualizations ---")

    #%%
    config = load_config("datapp_de")
    setup_name = config.get('setup_name', 'default_setup')
    study_name = "study_20250922"
    results_dir = PROJECT_ROOT / "results" / "de_uvtzq_scf_NUTS2" / study_name
    db_path = results_dir / f"{study_name}.db"
    storage_name = f"sqlite:///{db_path}"

    # Load the study from the database if needed
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction='minimize'
    )
    if len(study.trials) > 0:
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig2 = optuna.visualization.plot_param_importances(study)
        fig3 = optuna.visualization.plot_slice(study)
        fig4 = optuna.visualization.plot_contour(study)
        fig5 = optuna.visualization.plot_parallel_coordinate(study)
        fig6 = optuna.visualization.plot_edf(study)
        fig7 = optuna.visualization.plot_intermediate_values(study)
        
        fig1.write_html(results_dir / "optimization_history.html")
        fig2.write_html(results_dir / "param_importances.html")
        fig3.write_html(results_dir / "slice_plot.html")
        fig4.write_html(results_dir / "contour_plot.html")
        fig5.write_html(results_dir / "parallel_coordinate.html")
        fig6.write_html(results_dir / "edf_plot.html")
        fig7.write_html(results_dir / "intermediate_values.html")
        logger.info(f"Plots saved to {results_dir}")
# %%
