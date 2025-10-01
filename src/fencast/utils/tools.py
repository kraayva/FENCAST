# src/fencast/utils/tools.py

import logging
from datetime import datetime
from pathlib import Path
from fencast.utils.paths import LOG_DIR

def setup_logger(prefix: str = "default"):
    """
    Configures and returns a logger to be used throughout the project.
    
    The logger will write to both a file and the console.
    """
    # Create a unique log file name for each script using a timestamp
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_name = f"{run_timestamp}_{prefix}.log"
    
    # Ensure the log directory exists
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file_path = LOG_DIR / log_file_name

    # Get the root logger
    logger = logging.getLogger("fencast")
    logger.setLevel(logging.INFO) # Set the minimum level of messages to log

    # Prevent logs from being propagated to the root logger if it has other handlers
    logger.propagate = False
    
    # If handlers are already present, don't add more
    if logger.hasHandlers():
        return logger

    # --- Create Handlers ---
    # 1. File Handler: writes log messages to a file
    file_handler = logging.FileHandler(log_file_path)
    
    # 2. Stream Handler: writes log messages to the console (e.g., your terminal)
    stream_handler = logging.StreamHandler()

    # --- Create Formatter ---
    # Defines the format of the log messages
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # --- Add Handlers to the Logger ---
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    logger.info(f"Logger initialized. Log file at: {log_file_path}")

    return logger


def get_latest_study_dir(results_parent_dir: Path, model_type: str) -> Path:
    prefix = f"study_{model_type}"
    model_studies = [d for d in results_parent_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    if not model_studies:
        raise FileNotFoundError(f"No study found for model type '{model_type}' in {results_parent_dir}")
    return sorted(model_studies, key=lambda f: f.stat().st_mtime, reverse=True)[0]