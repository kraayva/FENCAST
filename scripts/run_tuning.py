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


def objective(trial: optuna.Trial, model_type: str, config: dict) -> float:
    """
    Optuna objective function to tune hyperparameters for a given model architecture.
    """
    # 1. SETUP & HYPERPARAMETER SUGGESTIONS
    # ============================================================================
    logger.info(f"--- Starting Trial {trial.number} for model_type='{model_type}' ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tuning_config = config.get('tuning', {})
    model_tuning_config = tuning_config.get(model_type, {})
    if not model_tuning_config:
        raise ValueError(f"Tuning configuration for model_type '{model_type}' not found in config.")

    params = {}

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

        # Dynamically suggest learning rate scheduler parameters if a scheduler is used
        params['lr'] = trial.suggest_float("lr", lr_config.get('min'), lr_config.get('max'), log=lr_config.get('log_scale'))
        scheduler_name = trial.suggest_categorical("scheduler", ["None", "ReduceLROnPlateau"])
        scheduler_params = {}
        if scheduler_name == "ReduceLROnPlateau":
            # Let Optuna tune the scheduler's patience and reduction factor
            scheduler_params['patience'] = trial.suggest_int("patience", 2, 5)
            scheduler_params['factor'] = trial.suggest_float("factor", 0.1, 0.5)
        
        # Other hyperparameters
        params['dropout_rate'] = trial.suggest_float("dropout", dropout_config.get('min'), dropout_config.get('max'))
        params['n_conv_layers'] = trial.suggest_int("n_conv_layers", conv_layers_config.get('min_layers'), conv_layers_config.get('max_layers'))
        params['kernel_size'] = trial.suggest_categorical("kernel_size", model_tuning_config.get('kernel_size', [3, 5]))

        out_channels = []
        for i in range(params['n_conv_layers']):
            n_filters = trial.suggest_int(f"n_filters_l{i}", filters_config.get('min'), filters_config.get('max'), step=filters_config.get('step', 1))
            out_channels.append(n_filters)
        params['out_channels'] = out_channels
        

    params['activation_name'] = trial.suggest_categorical("activation", tuning_config.get('activations', ["ReLU", "ELU"]))
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
        model = DynamicCNN(
            config=config,
            params=params
        ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    scheduler = None
    if scheduler_name == "ReduceLROnPlateau":
        logger.info(f"Trial {trial.number}: Using ReduceLROnPlateau scheduler.")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_params['factor'],
            patience=scheduler_params['patience']
        )

    # 4. TRAINING & VALIDATION LOOP
    # ============================================================================
    best_validation_loss = float('inf')

    for epoch in range(epochs):
        model.train()
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

        trial.report(avg_validation_loss, epoch)

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
        n_warmup_steps=3 # number of epochs before pruning
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