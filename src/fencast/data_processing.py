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

def load_and_prepare_data(config: dict, model_target: str) -> Tuple[Union[pd.DataFrame, np.ndarray], pd.DataFrame]:
    """
    Loads ERA5 and CF data, aligns them in time, and processes them into a format
    suitable for the specified model target.

    Args:
        config (dict): The project's configuration dictionary.
        model_target (str): The target model architecture, either 'ffnn' or 'cnn'.
                            Determines the output shape of the feature data.

    Returns:
        A tuple containing:
        - X (pd.DataFrame or np.ndarray): The processed feature data.
        - y (pd.DataFrame): The processed label data.
    """
    print(f"Starting data preparation for model target: '{model_target}'...")

    # --- 1. Load and combine the weather data (Input Features) ---
    era5_files = list(config['era5_data_raw'].values())
    print(f"Loading {len(era5_files)} ERA5 files specified in config...")
    datasets = [xr.open_dataset(f) for f in era5_files]
    weather_data = xr.merge(datasets, compat='override', join='inner')
    weather_data = weather_data.rename(config['era5_var_names'])
    print("Weather data successfully loaded and merged.")

    # --- 2. Load the target data (Output Labels) ---
    cf_file = Path(config['target_data_raw'])
    df_cf = pd.read_csv(cf_file, index_col='Date', parse_dates=True)
    print("Capacity factor (CF) data loaded.")

    # Drop columns from target data as specified in config
    drop_cols = config.get('data_processing', {}).get('drop_columns', [])
    if drop_cols:
        df_cf = df_cf.drop(columns=drop_cols, errors='ignore')
        print(f"Dropped columns from CF data: {drop_cols}")

    # --- 3. Align data by finding common timestamps ---
    weather_index = weather_data.time.to_index()
    cf_index = df_cf.index
    
    if weather_index.tz is not None:
        weather_index = weather_index.tz_localize(None)
    if cf_index.tz is not None:
        cf_index = cf_index.tz_localize(None)
        
    # Find common timestamps
    common_index = weather_index.intersection(cf_index)
    if len(common_index) == 0:
        print("WARNING: No exact timestamp matches found!")
        print("Attempting to find the closest matching timestamps...")
        
        # Try to resample or find nearest matches
        # This handles cases where weather data is hourly and CF data is daily
        
        # First align by date only
        weather_dates = weather_index.normalize()  # Remove time component
        cf_dates = cf_index.normalize()
        
        common_dates = weather_dates.intersection(cf_dates)
        
        if len(common_dates) > 0:
            # Use the weather data timestamps that fall on matching dates
            matching_weather_times = weather_index[weather_index.normalize().isin(common_dates)]
            matching_cf_times = cf_index[cf_index.normalize().isin(common_dates)]

            common_index = matching_weather_times.intersection(matching_cf_times)
            
            if len(common_index) == 0:
                print("Still no matches - will use weather timestamps and interpolate CF data")
                common_index = matching_weather_times
                df_cf = df_cf.reindex(common_index, method='nearest', tolerance=pd.Timedelta('12 hours'))
            
        else:
            raise ValueError("No overlapping dates between weather and CF data")
    
    # Filter both datasets to the common, aligned timestamps
    weather_data = weather_data.sel(time=common_index)
    df_cf = df_cf.loc[common_index]
    
    # Filter by the overall time range specified in the config
    start_date = config['time_start']
    end_date = config['time_end']
    weather_data = weather_data.sel(time=slice(start_date, end_date))
    df_cf = df_cf.loc[start_date:end_date]
    print(f"Data aligned and filtered to range {start_date} - {end_date}.")

    # --- 4. Process data based on the model target ---
    if model_target == 'ffnn':
        # --- FFNN Path: Flatten data into a 2D DataFrame ---
        print("Processing for FFNN: Flattening spatial and level dimensions...")
        
        # Convert to DataFrame and unstack spatial/level dimensions into columns
        df_weather = weather_data.to_dataframe()
        df_weather_flat = df_weather.unstack(level=['level', 'latitude', 'longitude'])
        df_weather_flat.columns = ['_'.join(map(str, col)) for col in df_weather_flat.columns]
        
        # Add cyclical temporal features
        print("Adding cyclical temporal features...")
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
        print("Processing for CNN: Structuring data into a 4D array...")
        
        # Get the variable names to ensure consistent ordering
        var_names = list(config['era5_var_names'].values())
        
        # Convert the xarray Dataset to a DataArray, stacking variables along a new dimension
        # The resulting shape will be (time, variable, level, latitude, longitude)
        da_weather = weather_data[var_names].to_array(dim='variable')
        
        # Transpose to get a standard (samples, channels, height, width) format
        # Here, we treat 'variable' and 'level' as channel dimensions
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
        raise ValueError(f"Invalid model_target: '{model_target}'. Must be 'ffnn' or 'cnn'.")

    print(f"Data processing complete. X shape: {X.shape}, y shape: {y.shape}")
    return X, y