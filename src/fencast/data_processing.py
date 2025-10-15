# src/fencast/data_processing.py

"""
This module contains the core data processing functions.
To execute the data processing pipeline, run the corresponding script:
`python scripts/run_data_processing.py --model-target [ffnn|cnn]`
"""

import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
from typing import Tuple, Union
from fencast.utils.paths import PROJECT_ROOT, DATA_DIR, RAW_DATA_DIR
from fencast.utils.tools import setup_logger

logger = setup_logger("data_processing")

def load_and_prepare_data(
    config: dict,
    model_target: str,
    feature_prefix: str = "era5_de"
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], pd.DataFrame]:
    """
    Loads weather and CF data, aligns them in time, and processes them into a format
    suitable for the specified model target.

    Args:
        config (dict): The project's configuration dictionary.
        model_target (str): The target model architecture, either 'ffnn' or 'cnn'.
                            Determines the output shape of the feature data.
        feature_prefix (str, optional): Prefix for feature data files. Default is "era5_de".

    Returns:
        A tuple containing:
        - X (pd.DataFrame or np.ndarray): The processed feature data.
        - y (pd.DataFrame): The processed label data.
    """
    logger.info(f"Starting data preparation for model target: '{model_target}'...")

    # --- 1. Load and combine the weather data (Input Features) ---
    feature_var_names = config['feature_var_names']
    feature_files = [RAW_DATA_DIR / f'{feature_prefix}_{var}.nc' for var in feature_var_names.keys()]
    logger.info(f"Loading {len(feature_files)} feature files")
    datasets = [xr.open_dataset(f) for f in feature_files]
    weather_data = xr.merge(datasets, compat='override', join='inner')
    weather_data = weather_data.rename(config['feature_var_names'])
    logger.info("Weather data successfully loaded and merged.")

    # --- 2. Load the target data (Output Labels) ---
    cf_file = Path(config['target_data_raw'])
    df_cf = pd.read_csv(cf_file, index_col='Date', parse_dates=True)
    logger.info("Capacity factor (CF) data loaded.")

    # Drop columns from target data as specified in config
    drop_cols = config.get('data_processing', {}).get('drop_columns', [])
    if drop_cols:
        df_cf = df_cf.drop(columns=drop_cols, errors='ignore')
        logger.info(f"Dropped columns from CF data: {drop_cols}")

    # --- 3. Filter both datasets to 12:00 timestamps only ---
    logger.info("Filtering data to 12:00 timestamps only...")
    
    # Filter weather data to 12:00 only
    weather_data = weather_data.sel(time=weather_data['time'].dt.hour == 12)
    logger.info(f"Weather data filtered to {len(weather_data.time)} timestamps at 12:00")
    
    # Filter CF data to 12:00 only  
    df_cf = df_cf[df_cf.index.hour == 12]
    logger.info(f"CF data filtered to {len(df_cf)} timestamps at 12:00")
    
    # Find common timestamps (should be straightforward now)
    weather_index = weather_data.time.to_index()
    cf_index = df_cf.index
    common_index = weather_index.intersection(cf_index)
    
    if len(common_index) == 0:
        logger.error("No overlapping timestamps between weather and CF data at 12:00.")
        raise ValueError("No overlapping timestamps between weather and CF data at 12:00")
    
    logger.info(f"Found {len(common_index)} common timestamps")
    
    # Filter both datasets to the common timestamps
    weather_data = weather_data.sel(time=common_index)
    df_cf = df_cf.loc[common_index]
    
    # Filter by the overall time range specified in the config
    start_date = config['time_start']
    end_date = config['time_end']
    weather_data = weather_data.sel(time=slice(start_date, end_date))
    df_cf = df_cf.loc[start_date:end_date]
    logger.info(f"Data aligned and filtered to range {start_date} - {end_date}.")

    # --- 4. Process data based on the model target ---
    if model_target == 'ffnn':
        # --- FFNN Path: Flatten data into a 2D DataFrame ---
        logger.info("Processing for FFNN: Flattening spatial and level dimensions...")
        
        # Convert to DataFrame and unstack spatial/level dimensions into columns
        df_weather = weather_data.to_dataframe()
        df_weather_flat = df_weather.unstack(level=['level', 'latitude', 'longitude'])
        df_weather_flat.columns = ['_'.join(map(str, col)) for col in df_weather_flat.columns]
        
        # Add cyclical temporal features
        logger.info("Adding cyclical temporal features...")
        day_of_year = df_weather_flat.index.dayofyear
        norm_denom = config.get('data_processing', {}).get('day_of_year_normalize_denominator', 365.0)
        day_of_year_rad = ((day_of_year - 1) / norm_denom) * 2 * np.pi
        
        temporal_features = pd.DataFrame({
            'day_of_year_sin': np.sin(day_of_year_rad),
            'day_of_year_cos': np.cos(day_of_year_rad)
        }, index=df_weather_flat.index)
        
        # Combine weather features with temporal features
        X = pd.concat([df_weather_flat, temporal_features], axis=1)
        y = df_cf

    elif model_target == 'cnn':
        # --- CNN Path: Structure data into a 4D NumPy array ---
        logger.info("Processing for CNN: Structuring data into a 4D array...")
        
        # Get the variable names to ensure consistent ordering
        var_names = list(config['feature_var_names'].values())
        
        # Convert the xarray Dataset to a DataArray, stacking variables along a new dimension
        # The resulting shape will be (time, variable, level, latitude, longitude)
        da_weather = weather_data[var_names].to_array(dim='variable')
        
        # Transpose to get a standard (samples, channels, height, width) format
        # Treat 'variable' and 'level' as channel dimensions
        # Shape: (time, variable, level, latitude, longitude)
        da_weather = da_weather.transpose('time', 'variable', 'level', 'latitude', 'longitude')

        # Get the raw NumPy array
        X_np = da_weather.values
        
        # Reshape to combine 'variable' and 'level' into a single channel dimension
        # Shape: (n_samples, n_variables, n_levels, n_lat, n_lon) -> (n_samples, n_variables * n_levels, n_lat, n_lon)
        n_samples, n_vars, n_levels, n_lat, n_lon = X_np.shape
        X = X_np.reshape(n_samples, n_vars * n_levels, n_lat, n_lon)
        y = df_cf

    else:
        logger.error(f"Invalid model_target: '{model_target}'. Must be 'ffnn' or 'cnn'.")
        raise ValueError(f"Invalid model_target: '{model_target}'. Must be 'ffnn' or 'cnn'.")

    logger.info(f"Data processing complete. X shape: {X.shape}, y shape: {y.shape}")
    return X, y