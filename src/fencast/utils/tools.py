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


def load_era5_data(var_name: str) -> 'xr.Dataset':
    """Loads ERA5 reference data for a specific variable."""
    import xarray as xr
    from fencast.utils.paths import RAW_DATA_DIR
    
    era5_file = RAW_DATA_DIR / f"era5_de_{var_name}.nc"
    if not era5_file.exists():
        raise FileNotFoundError(f"ERA5 reference file not found: {era5_file}")
    
    return xr.open_dataset(era5_file)


def get_mlwp_forecast_lead_time(mlwp_name: str, timedelta_str: str, var_name: str) -> float:
    """Get the actual forecast lead time in days from consolidated MLWP file."""
    import numpy as np
    import xarray as xr
    from fencast.utils.paths import RAW_DATA_DIR
    
    # New consolidated file structure: one file per variable with all timedeltas
    mlwp_file = RAW_DATA_DIR / f"{mlwp_name}_de_{var_name}.nc"
    if not mlwp_file.exists():
        raise FileNotFoundError(f"MLWP prediction file not found: {mlwp_file}")
    
    # Extract timedelta index from string (e.g., "td03" -> 3)
    td_index = int(timedelta_str.replace('td', '').lstrip('0') or '0')
    
    ds = xr.open_dataset(mlwp_file)
    try:
        # Convert to timedelta for selection
        target_timedelta = np.timedelta64(td_index, 'D')
        
        # Find the specific timedelta in the array
        prediction_timedeltas = ds['prediction_timedelta'].values
        
        # Check if our target timedelta exists
        if target_timedelta in prediction_timedeltas:
            return float(target_timedelta / np.timedelta64(1, 'D'))
        else:
            # Fallback: return the td_index directly (should be the same)
            available_days = [float(td / np.timedelta64(1, 'D')) for td in prediction_timedeltas]
            if td_index in available_days:
                return float(td_index)
            else:
                raise ValueError(f"Timedelta {td_index} days not found. Available: {available_days}")
    finally:
        ds.close()


def load_mlwp_data(mlwp_name: str, timedelta_str: str, var_name: str) -> 'xr.Dataset':
    """Loads MLWP prediction data for a specific variable and timedelta from consolidated file."""
    import xarray as xr
    import numpy as np
    from fencast.utils.paths import RAW_DATA_DIR
    
    # New consolidated file structure: one file per variable with all timedeltas
    mlwp_file = RAW_DATA_DIR / f"{mlwp_name}_de_{var_name}.nc"
    if not mlwp_file.exists():
        raise FileNotFoundError(f"MLWP prediction file not found: {mlwp_file}")
    
    # Load the full dataset
    ds = xr.open_dataset(mlwp_file)
    
    # Extract timedelta index from string (e.g., "td03" -> 3)
    td_index = int(timedelta_str.replace('td', '').lstrip('0') or '0')
    
    # Convert to days for selection (td_index corresponds to days: td03 = 3 days)
    target_timedelta = np.timedelta64(td_index, 'D')
    
    try:
        # Select the specific timedelta from the consolidated file
        selected_ds = ds.sel(prediction_timedelta=target_timedelta)
        
        # Check if prediction_timedelta dimension still exists and remove it
        # This maintains compatibility with existing code that expects no timedelta dimension
        if 'prediction_timedelta' in selected_ds.dims:
            selected_ds = selected_ds.squeeze('prediction_timedelta', drop=True)
        
        return selected_ds
        
    except KeyError as e:
        ds.close()
        available_tds = ds.prediction_timedelta.values
        available_days = [float(td / np.timedelta64(1, 'D')) for td in available_tds]
        raise ValueError(f"Timedelta {td_index} days not found in {mlwp_file}. "
                        f"Available timedeltas: {available_days} days") from e
    except Exception as e:
        ds.close()
        raise e


def calculate_persistence_baseline(config: dict, timedelta_days: int, logger=None) -> float:
    """Calculates persistence baseline RMSE for a specific forecast lead time."""
    import pandas as pd
    import numpy as np
    from sklearn.metrics import mean_squared_error
    from fencast.utils.paths import PROJECT_ROOT
    
    # Load the full raw target dataset
    gt_file = PROJECT_ROOT / config['target_data_raw']
    full_df = pd.read_csv(gt_file, index_col='Date', parse_dates=True)
    full_df.index = full_df.index + pd.Timedelta(hours=12)  # Apply noon-shift
    
    # Drop columns to match the model's target
    drop_cols = config.get('data_processing', {}).get('drop_columns', [])
    if drop_cols:
        full_df = full_df.drop(columns=drop_cols, errors='ignore')
        
    # Filter for the test set years
    test_years = config['split_years']['test']
    test_gt = full_df[full_df.index.year.isin(test_years)].dropna()
    
    # Persistence baseline: use data from `timedelta_days` days ago
    persistence_preds = test_gt.shift(timedelta_days)
    
    # Find valid overlapping data
    valid_mask = ~(test_gt.isna().any(axis=1) | persistence_preds.isna().any(axis=1))
    valid_gt = test_gt[valid_mask]
    valid_preds = persistence_preds[valid_mask]
    
    if len(valid_gt) > 0:
        rmse = np.sqrt(mean_squared_error(valid_gt, valid_preds))
        
        # Debug logging if logger provided
        if logger and timedelta_days <= 20:
            logger.info(f"Persistence {timedelta_days}d: RMSE={rmse:.4f}, samples={len(valid_gt)}, "
                       f"date_range={valid_gt.index.min().strftime('%Y-%m-%d')} to {valid_gt.index.max().strftime('%Y-%m-%d')}")
            
            # Check for weekly patterns
            if timedelta_days in [7, 14]:
                gt_dow = valid_gt.index.dayofweek
                pred_dow = valid_preds.index.dayofweek
                same_dow_ratio = (gt_dow == pred_dow).mean()
                logger.info(f"  Same day-of-week ratio: {same_dow_ratio:.3f} (1.0 = always same weekday)")
        
        return rmse
    else:
        if logger:
            logger.warning(f"No valid data for persistence baseline at {timedelta_days} days")
        return np.nan


def calculate_climatology_baseline(config: dict) -> float:
    """Calculates climatology baseline RMSE (same for all lead times)."""
    import pandas as pd
    import numpy as np
    from sklearn.metrics import mean_squared_error
    from fencast.utils.paths import PROJECT_ROOT
    
    # Load the full raw target dataset
    gt_file = PROJECT_ROOT / config['target_data_raw']
    full_df = pd.read_csv(gt_file, index_col='Date', parse_dates=True)
    full_df.index = full_df.index + pd.Timedelta(hours=12)  # Apply noon-shift
    
    # Drop columns to match the model's target
    drop_cols = config.get('data_processing', {}).get('drop_columns', [])
    if drop_cols:
        full_df = full_df.drop(columns=drop_cols, errors='ignore')
        
    # Filter for the test set years
    test_years = config['split_years']['test']
    test_gt = full_df[full_df.index.year.isin(test_years)].dropna()
    
    # Climatology baseline: use historical averages (1990-1999)
    historical_data = full_df.loc['1990':'1999']
    daily_climatology = historical_data.groupby([historical_data.index.month, historical_data.index.day]).mean()
    daily_climatology.index.names = ['month', 'day']
    
    preds_df = pd.DataFrame(index=test_gt.index)
    preds_df['month'] = preds_df.index.month
    preds_df['day'] = preds_df.index.day
    
    merged = pd.merge(preds_df, daily_climatology, on=['month', 'day'], how='left').ffill().bfill()
    merged.index = test_gt.index
    climatology_preds = merged[test_gt.columns]
    
    return np.sqrt(mean_squared_error(test_gt, climatology_preds))