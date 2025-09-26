# scripts/run_tuning.py

# Import standard libraries
import argparse
import json
from datetime import datetime
import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import custom modules
from fencast.dataset import FencastDataset
from fencast.models import DynamicCNN, DynamicFFNN
from fencast.utils.paths import PROJECT_ROOT, load_config
from fencast.utils.tools import setup_logger

# Setup logger once at the start of the script
logger = setup_logger("hyperparameter_tuning")


## --------------------------------- ##
## --- OPTUNA OBJECTIVE FUNCTION --- ##
## --------------------------------- ##

def objective(trial: optuna.Trial, model_type: str, config: dict) -> float:
    """
    Optuna objective function to tune hyperparameters for a given model architecture.
    """
    # 1. SETUP & HYPERPARAMETER SUGGESTIONS
    # ============================================================================
    logger.info(f"--- Starting Trial {trial.number} for model_type='{model_type}' ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the correct tuning configuration for the specified model
    tuning_config = config.get('tuning', {})
    model_tuning_config = tuning_config.get(model_type, {})
    if not model_tuning_config:
        raise ValueError(f"Tuning configuration for model_type '{model_type}' not found in config.")

    params = {}

    # --- Model-specific hyperparameter suggestions ---
    if model_type == 'ffnn':
        lr_config = model_tuning_config.get('learning_rate', {})
        dropout_config = model_tuning_config.get('dropout', {})
        layers_config = model_tuning_config.get('hidden_layers', {})

        params['lr'] = trial.suggest_float("lr", lr_config.get('min'), lr_config.get('max'), log=lr_config.get('log_scale'))
        params['dropout_rate'] = trial.suggest_float("dropout", dropout_config.get('min'), dropout_config.get('max'))
        params['n_layers'] = trial.suggest_int("n_layers", layers_config.get('min_layers'), layers_config.get('max_layers'))

        hidden_layers = []
        for i in range(params['n_layers']):
            n_units = trial.suggest_int(f"n_units_l{i}", layers_config.get('min_units'), layers_config.get('max_units'))
            hidden_layers.append(n_units)
        params['hidden_layers'] = hidden_layers

    elif model_type == 'cnn':
        lr_config = model_tuning_config.get('learning_rate', {})
        dropout_config = model_tuning_config.get('dropout', {})
        conv_layers_config = model_tuning_config.get('conv_layers', {})
        filters_config = model_tuning_config.get('filters', {})

        params['lr'] = trial.suggest_float("lr", lr_config.get('min'), lr_config.get('max'), log=lr_config.get('log_scale'))
        params['dropout_rate'] = trial.suggest_float("dropout", dropout_config.get('min'), dropout_config.get('max'))
        params['n_conv_layers'] = trial.suggest_int("n_conv_layers", conv_layers_config.get('min_layers'), conv_layers_config.get('max_layers'))
        params['kernel_size'] = trial.suggest_categorical("kernel_size", model_tuning_config.get('kernel_size', [3, 5]))

        out_channels = []
        for i in range(params['n_conv_layers']):
            n_filters = trial.suggest_int(f"n_filters_l{i}", filters_config.get('min'), filters_config.get('max'), step=filters_config.get('step', 1))
            out_channels.append(n_filters)
        params['out_channels'] = out_channels

    # --- Common hyperparameter suggestions ---
    params['activation_name'] = trial.suggest_categorical("activation", tuning_config.get('activations', ["ReLU", "ELU"]))
    logger.info(f"Trial {trial.number} Parameters: {json.dumps(params, indent=4)}")

    # General parameters from config
    output_size = config['target_size']
    batch_size = config.get('model', {}).get('batch_sizes', {}).get('tuning', 64)
    epochs = tuning_config.get('epochs', 20)

    # 2. DATA LOADING
    # ============================================================================
    # The FencastDataset class should handle the data shape based on model_type
    logger.info(f"Trial {trial.number}: Loading data...")
    train_dataset = FencastDataset(config=config, mode='train', model_type=model_type)
    validation_dataset = FencastDataset(config=config, mode='validation', model_type=model_type)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)
    logger.info(f"Trial {trial.number}: Data loading complete.")

    # 3. MODEL, LOSS, and OPTIMIZER INITIALIZATION
    # ============================================================================
    model = None
    activation_fn = getattr(nn, params['activation_name'])()

    if model_type == 'ffnn':
        model = DynamicFFNN(
            input_size=config['input_size_flat'],
            output_size=output_size,
            hidden_layers=params['hidden_layers'],
            dropout_rate=params['dropout_rate'],
            activation_fn=activation_fn
        ).to(device)
    elif model_type == 'cnn':
        model = DynamicCNN(
            input_channels=config.get('input_channels', 1),
            output_size=output_size,
            out_channels_list=params['out_channels'],
            kernel_size=params['kernel_size'],
            dropout_rate=params['dropout_rate'],
            activation_fn=activation_fn
        ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    # 4. TRAINING & VALIDATION LOOP
    # ============================================================================
    best_validation_loss = float('inf')

    for epoch in range(epochs):
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

        trial.report(avg_validation_loss, epoch)

        if trial.should_prune():
            logger.warning(f"--- Trial {trial.number} pruned at epoch {epoch + 1} ---")
            raise optuna.exceptions.TrialPruned()

    logger.info(f"--- Trial {trial.number} finished. Best validation loss: {best_validation_loss:.6f} ---")
    return best_validation_loss


## ------------------------------ ##
## --- MAIN SCRIPT EXECUTION ---- ##
## ------------------------------ ##

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run hyperparameter tuning for a given model architecture.")
    parser.add_argument(
        '--model-type', '-m',
        type=str,
        choices=['ffnn', 'cnn'],
        required=True,
        help="The type of model architecture to tune ('ffnn' or 'cnn')."
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='datapp_de',
        help='Configuration file name (e.g., datapp_de) without the .yaml extension.'
    )
    args = parser.parse_args()

    # --- 1. SETUP STUDY ---
    config = load_config(args.config)
    current_date = datetime.now().strftime('%Y%m%d')
    setup_name = config.get('setup_name', 'default_setup')
    study_name = f"study_{args.model_type}_{setup_name}_{current_date}"
    
    results_dir = PROJECT_ROOT / "results" / setup_name / study_name
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Config '{args.config}' loaded.")
    logger.info(f"Starting new study: '{study_name}' for model '{args.model_type}'")
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

    # --- 2. RUN OPTIMIZATION ---
    n_trials = config.get('tuning', {}).get('trials', 50)
    # Use a lambda function to pass additional arguments to the objective function
    study.optimize(lambda trial: objective(trial, model_type=args.model_type, config=config), n_trials=n_trials, n_jobs=4)

    # --- 3. LOG AND SAVE RESULTS ---
    logger.info("--- Tuning Finished ---")
    logger.info("Study statistics: ")
    logger.info(f"  Number of finished trials: {len(study.trials)}")
    
    best_trial = study.best_trial
    logger.info("Best trial:")
    logger.info(f"  Value (Best Validation Loss): {best_trial.value:.6f}")
    params_str = json.dumps(best_trial.params, indent=4)
    logger.info(f"  Params: \n{params_str}")

    # --- 4. VISUALIZE AND SAVE PLOTS ---
    logger.info("--- Generating and saving visualizations for the finished study ---")
    if len(study.trials) > 0:
        plots = {
            "optimization_history": optuna.visualization.plot_optimization_history,
            "param_importances": optuna.visualization.plot_param_importances,
            "slice": optuna.visualization.plot_slice,
            "parallel_coordinate": optuna.visualization.plot_parallel_coordinate,
            "intermediate_values": optuna.visualization.plot_intermediate_values,
        }
        
        for name, plot_func in plots.items():
            try:
                fig = plot_func(study)
                fig.write_html(results_dir / f"{name}_plot.html")
            except (ValueError, ZeroDivisionError) as e:
                logger.warning(f"Could not generate plot '{name}': {e}")
                
        logger.info(f"Plots saved to {results_dir}")
    else:
        logger.info("No trials completed, skipping plot generation.")
