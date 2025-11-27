# src/fencast/utils/experiment_management.py

import torch
import optuna
import json
from pathlib import Path
from typing import Dict, Any

from fencast.utils.paths import PROJECT_ROOT
from fencast.utils.tools import get_latest_study_dir, setup_logger
from fencast.models import DynamicCNN

logger = setup_logger("experiment_management")


def load_best_params_from_study(results_parent_dir: Path, study_name: str = 'latest') -> Dict[str, Any]:
    """
    Loads the best hyperparameters from a specified Optuna study.

    Args:
        results_parent_dir (Path): The parent directory containing all study results.
        study_name (str): The name of the study to load from. 'latest' will find the most recent one.

    Returns:
        Dict[str, Any]: A dictionary of the best hyperparameters.
    """
    try:
        if study_name == 'latest':
            study_dir = get_latest_study_dir(results_parent_dir)
        else:
            study_dir = results_parent_dir / study_name

        study_db_name = study_dir.name
        storage_name = f"sqlite:///{study_dir / study_db_name}.db"

        study = optuna.load_study(study_name=study_db_name, storage=storage_name)
        params = study.best_trial.params

        logger.info(f"Loaded best parameters from study: '{study_dir.name}'")
        logger.info(f"Best trial value (loss): {study.best_trial.value:.6f}")
        logger.info(f"Best parameters:\n{json.dumps(params, indent=4)}")

        return params, study_dir

    except Exception as e:
        logger.error(f"Failed to load study '{study_name}': {e}")
        raise


def load_trained_model(study_dir: Path, use_final_model: bool = False, device: str = 'cpu') -> torch.nn.Module:
    """
    Loads a trained model checkpoint from a study directory.

    Args:
        study_dir (Path): The directory of the specific study.
        use_final_model (bool): If True, loads 'final_model.pth'. Otherwise, loads 'best_model.pth'.
        device (str): The device to map the model to ('cpu' or 'cuda').

    Returns:
        torch.nn.Module: The loaded and initialized model in evaluation mode.
    """
    model_file = "final_model.pth" if use_final_model else "best_model.pth"
    model_path = study_dir / model_file
    logger.info(f"Loading model from: {model_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    checkpoint = torch.load(model_path, map_location=torch.device(device), weights_only=False)
    model_class = DynamicCNN
    model = model_class(**checkpoint['model_args'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info("Model loaded successfully and set to evaluation mode.")
    return model