# src/fencast/data_processing.py

import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path

# --- Import path manager ---
from fencast.utils.paths import load_config, PROCESSED_DATA_DIR

# Columns to drop will be read from config

def load_and_prepare_data(config: dict):
    """
    Loads ERA5 and CF data based on a given config, merges them, and prepares them for training.

    Args:
        config (dict): A dictionary loaded from a data processing config file (e.g., datapp_de.yaml).
    """
    print("Starting data preparation...")

    # --- Load and combine the weather data (Input Features) ---
    era5_files = list(config['era5_data_raw'].values())
    print(f"Loading {len(era5_files)} ERA5 files specified in config...")

    datasets = [xr.open_dataset(f) for f in era5_files]
    weather_data = xr.merge(datasets, compat='override', join='inner')
    
    print("\nWeather data successfully loaded and merged.")

    # --- Flatten the spatial and level dimensions ---
    df_weather = weather_data.to_dataframe()
    df_weather = df_weather.rename(columns=config['era5_var_names'])
    
    vars_to_drop = [col for col in df_weather.columns if col not in config['era5_var_names'].values()]
    df_weather = df_weather.drop(columns=vars_to_drop)
    
    df_weather_flat = df_weather.unstack(level=['level', 'latitude', 'longitude'])
    print("Spatial data and levels unstacked into columns.")

    df_weather_flat.columns = [f'{var}_{level}_{lat:.2f}_{lon:.2f}' for var, level, lat, lon in df_weather_flat.columns]
    df_weather_flat.index = df_weather_flat.index.tz_localize('UTC')

    # --- Load the target data (Output Labels) ---
    cf_file = Path(config['target_data_raw'])
    df_cf = pd.read_csv(cf_file, index_col='Date', parse_dates=True)
    df_cf.index = df_cf.index.tz_localize('UTC')
    print("\nCapacity factor (CF) data loaded.")

    # Drop columns as specified in config
    drop_cols = config.get('data_processing', {}).get('drop_columns', [])
    if drop_cols:
        df_cf = df_cf.drop(columns=drop_cols, errors='ignore')
        print(f"Dropped columns from CF data: {drop_cols}")

    # --- Merge, filter, and split ---
    combined_data = pd.merge(df_weather_flat, df_cf, left_index=True, right_index=True, how='inner')
    start_date = config['time_start']
    end_date = config['time_end']
    combined_data = combined_data.loc[start_date:end_date]
    
    # --- Add cyclical temporal features ---
    print("Adding cyclical temporal features...")
    
    # Day of year (1-365/366) normalized to [0, 1] then converted to radians
    day_of_year = combined_data.index.dayofyear
    normalize_denominator = config.get('data_processing', {}).get('day_of_year_normalize_denominator', 365.0)
    day_of_year_normalized = (day_of_year - 1) / normalize_denominator  # Normalize to [0, 1] 
    day_of_year_rad = day_of_year_normalized * 2 * np.pi  # Convert to radians [0, 2π]
    
    # Create sin and cos features
    temporal_features = pd.DataFrame({
        'day_of_year_sin': np.sin(day_of_year_rad),
        'day_of_year_cos': np.cos(day_of_year_rad)
    }, index=combined_data.index)
    
    # Combine weather features with temporal features
    X = pd.concat([combined_data[df_weather_flat.columns], temporal_features], axis=1)
    y = combined_data[df_cf.columns]

    print(f"Split complete. {X.shape[1]} features ({len(df_weather_flat.columns)} weather + {len(temporal_features.columns)} temporal) and {y.shape[1]} labels.")
    return X, y


# This module contains data processing functions.
# To run data processing, use: python scripts/run_data_processing.py