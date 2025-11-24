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


def suggest_parameter(trial: optuna.Trial, param_name: str, param_config, logger):
    """
    Suggest parameters based on new unified config structure:
    - Single value (str/int/float): Use as fixed parameter (no tuning)
    - List: Use as categorical choices
    - Dict with 'type': Tune based on type specification
    """
    if isinstance(param_config, dict):
        if 'type' in param_config:
            
            if param_config['type'] == 'list':
                # Categorical tuning from list values
                values = param_config.get('values', [])
                if len(values) > 1:
                    return trial.suggest_categorical(param_name, values)
                else:
                    logger.info(f"Parameter '{param_name}' has single-item list - treating as fixed: {values[0] if values else None}")
                    return values[0] if values else None
                    
            elif param_config['type'] == 'int':
                # Integer range tuning
                min_val = param_config['min']
                max_val = param_config['max']
                step = param_config.get('step', 1)
                return trial.suggest_int(param_name, min_val, max_val, step=step)
                
            elif param_config['type'] == 'float':
                # Float range tuning
                min_val = param_config['min']
                max_val = param_config['max']
                log_scale = param_config.get('log_scale', False)
                return trial.suggest_float(param_name, min_val, max_val, log=log_scale)
                
            else:
                logger.warning(f"Parameter '{param_name}' has unknown type '{param_config['type']}' - treating as fixed")
                return param_config
        else:
            # Dict without 'type' - treat as fixed
            logger.error(f"Parameter '{param_name}' is dict without 'type'")
            
    elif isinstance(param_config, list):
        # Direct list - use as categorical
        if len(param_config) > 1:
            return trial.suggest_categorical(param_name, param_config)
        else:
            logger.info(f"Parameter '{param_name}' has single-item list - treating as fixed: {param_config[0]}")
            return param_config[0]
    else:
        # Single value - fixed parameter
        logger.info(f"Parameter '{param_name}' is fixed: {param_config}")
        return param_config


def objective(trial: optuna.Trial, model_type: str, config: dict) -> float:
    """
    Optuna objective function to tune hyperparameters for a given model architecture.
    Intelligently determines what to tune vs. what to keep fixed based on config structure.
    """
    # 1. SETUP & HYPERPARAMETER SUGGESTIONS
    # ============================================================================
    logger.info(f"--- Starting Trial {trial.number} for model_type='{model_type}' ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Merge general tuning config with model-specific config
    tuning_config = config.get('tuning', {})
    model_specific_key = f"{model_type}_tuning"
    model_tuning_config = config.get(model_specific_key, {})
    
    # Combine both configs - model-specific overrides general
    combined_config = {**tuning_config, **model_tuning_config}
    
    logger.info(f"Using combined config from 'tuning' and '{model_specific_key}'")
    
    params = {}

    # Loop through all parameters in combined config and suggest values
    
    for param_name, param_config in combined_config.items():
        # Skip non-hyperparameter keys
        if param_name in ['trials', 'epochs', 'early_stopping_patience']:
            continue
            
        # Check if param_config is an integer, float, or string (fixed parameter)
        if isinstance(param_config, (int, float, str)):
            params[param_name] = param_config
        else:
            params[param_name] = suggest_parameter(trial, param_name, param_config, logger)
    
    # Handle special cases for model-specific parameters
    if model_type == 'ffnn':
        # Handle hidden layers configuration
        if 'hidden_layers' in params:
            n_layers = params['hidden_layers']
            if 'hidden_layers_units' in params:
                # Create hidden layers with tuned number of units
                hidden_layers = [params['hidden_layers_units']] * n_layers
                params['hidden_layers'] = hidden_layers
                logger.info(f"Created {n_layers} hidden layers with {params['hidden_layers_units']} units each")
            else:
                logger.info(f"Using fixed hidden layers: {params['hidden_layers']}")
    
    elif model_type == 'cnn':
        # Handle filters configuration - create filter list for each conv layer
        if 'filters' in params and 'n_conv_layers' in params:
            n_filters = params['filters']
            n_layers = params['n_conv_layers']
            params['out_channels'] = [n_filters] * n_layers
            logger.info(f"Created {n_layers} conv layers with {n_filters} filters each")
            # Remove the individual filter parameter as it's now in out_channels
            del params['filters']
        
    # Set defaults for missing required parameters
    if 'lr' not in params:
        params['lr'] = 1e-3
        logger.warning("No learning rate specified, using default 1e-3")
    if 'activation_name' not in params:
        params['activation_name'] = 'ELU'
        logger.warning("No activation specified, using default ELU")
    logger.info(f"Trial {trial.number} Parameters: {json.dumps(params, indent=4)}")

    output_size = config['target_size']
    batch_size = config.get('model', {}).get('batch_sizes', {}).get('tuning', 64)
    epochs = tuning_config.get('epochs', 30)

    # 2. DATA LOADING
    # ============================================================================
    logger.info(f"Trial {trial.number}: Loading data...")
    train_dataset = FencastDataset(config=config, mode='train', model_type=model_type)
    validation_dataset = FencastDataset(config=config, mode='validation', model_type=model_type)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)
    logger.info(f"Trial {trial.number}: Data loading complete.")

    # 3. MODEL, LOSS, and OPTIMIZER INITIALIZATION
    # ============================================================================
    model = None
    if model_type == 'ffnn':
        model = DynamicFFNN(
            input_size=config['input_size_flat'],
            output_size=output_size,
            hidden_layers=params['hidden_layers'],
            dropout_rate=params['dropout_rate'],
            activation_fn=getattr(nn, params['activation_name'])()
        ).to(device)
    elif model_type == 'cnn':
        try:
            model = DynamicCNN(
                config=config,
                params=params
            ).to(device)
        except ValueError as e:
            # Handle BatchNorm spatial dimension errors
            if "Expected more than 1 value per channel" in str(e):
                logger.warning(f"Trial {trial.number}: CNN architecture invalid (spatial dims too small), returning poor score")
                return 1.0  # Return a poor score instead of crashing
            else:
                raise e

    criterion = nn.MSELoss()
    
    # Create optimizer based on configuration
    optimizer_name = params.get('optimizer_name', 'Adam')
    if optimizer_name == 'AdamW':
        weight_decay = params.get('weight_decay', 0.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=weight_decay)
        logger.info(f"Trial {trial.number}: Using optimizer: {optimizer_name} with weight_decay={weight_decay:.2e}")
    else:  # Default to Adam
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
        logger.info(f"Trial {trial.number}: Using optimizer: {optimizer_name}")
    
    scheduler = None
    if params.get('scheduler_name') == "ReduceLROnPlateau":
        logger.info(f"Trial {trial.number}: Using ReduceLROnPlateau scheduler.")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=params.get('scheduler_factor', 0.2),
            patience=params.get('scheduler_patience', 3)
        )

    # 4. TRAINING & VALIDATION LOOP
    # ============================================================================
    best_validation_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        training_losses = []
        for batch in train_loader:
            if model_type == 'cnn':
                spatial_features, temporal_features, labels = batch
                spatial_features, temporal_features, labels = spatial_features.to(device), temporal_features.to(device), labels.to(device)
                outputs = model(spatial_features, temporal_features)
            else: # FFNN
                features, labels = batch
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
            
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            training_losses.append(loss.item())
        avg_training_loss = np.mean(training_losses)

        model.eval()
        validation_losses = []
        with torch.no_grad():
            for batch in validation_loader:
                if model_type == 'cnn':
                    spatial_features, temporal_features, labels = batch
                    spatial_features, temporal_features, labels = spatial_features.to(device), temporal_features.to(device), labels.to(device)
                    outputs = model(spatial_features, temporal_features)
                else: # FFNN
                    features, labels = batch
                    features, labels = features.to(device), labels.to(device)
                    outputs = model(features)
                
                loss = criterion(outputs, labels)
                validation_losses.append(loss.item())

        avg_validation_loss = np.mean(validation_losses)

        if avg_validation_loss < best_validation_loss:
            best_validation_loss = avg_validation_loss

        if scheduler:
            scheduler.step(avg_validation_loss)

        # Report validation loss (primary metric for optimization)
        trial.report(avg_validation_loss, epoch)
        
        # Set training loss as user attribute for logging/visualization
        trial.set_user_attr(f'training_loss_epoch_{epoch}', avg_training_loss)

        if trial.should_prune():
            logger.warning(f"--- Trial {trial.number} pruned at epoch {epoch + 1} ---")
            raise optuna.exceptions.TrialPruned()

    logger.info(f"--- Trial {trial.number} finished. Best validation loss: {best_validation_loss:.6f} ---")
    return best_validation_loss


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
    parser.add_argument(
        '--study-name', '-s',
        type=str,
        default=None,
        help='Optional study name for the Optuna study.'
    )
    args = parser.parse_args()

    config = load_config(args.config)
    current_date = datetime.now().strftime('%Y%m%d')
    setup_name = config.get('setup_name', 'default_setup')
    study_name = args.study_name or f"study_{args.model_type}_{setup_name}_{current_date}"

    results_dir = PROJECT_ROOT / "results" / setup_name / study_name
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Config '{args.config}' loaded.")
    logger.info(f"Starting new study: '{study_name}' for model '{args.model_type}'")
    logger.info(f"Study results will be saved in: {results_dir}")

    db_path = results_dir / f"{study_name}.db"
    storage_name = f"sqlite:///{db_path}"
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, # number of trials before pruning
        n_warmup_steps=15 # number of epochs before pruning
        
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction='minimize',
        pruner=pruner
    )

    n_trials = config.get('tuning', {}).get('trials', 50)
    study.optimize(lambda trial: objective(trial, model_type=args.model_type, config=config), n_trials=n_trials, n_jobs=2)
    
    logger.info("--- Tuning Finished ---")
    if len(study.trials) > 0:
        logger.info("Study statistics: ")
        logger.info(f"  Number of finished trials: {len(study.trials)}")
        
        best_trial = study.best_trial
        logger.info("Best trial:")
        logger.info(f"  Value (Best Validation Loss): {best_trial.value:.6f}")
        params_str = json.dumps(best_trial.params, indent=4)
        logger.info(f"  Params: \n{params_str}")

        logger.info("--- Generating and saving visualizations for the finished study ---")
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
        logger.info("No trials were completed. Skipping results and plots.")