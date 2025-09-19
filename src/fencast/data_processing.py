# %%
# src/fencast/data_processing.py

import pandas as pd
import xarray as xr
import yaml
import os
from pathlib import Path

# Set the project root and change the working directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

def load_config(name="global"):
    """Loads a YAML configuration file from the 'configs' directory."""
    config_path = PROJECT_ROOT / f"configs/{name}.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Load global and data-specific configurations
cfg = load_config("global")
cfm = load_config("datapp_de")

# %% Data Loading and Preparation

def load_and_prepare_data():
    """
    Loads ERA5 and CF data based on config files, merges them, and prepares them for training.

    This version flattens the entire spatial grid into a single long vector per timestep.
    (lat * lon * level * variable) -> features

    Returns:
        pd.DataFrame: A DataFrame containing the weather features as columns.
        pd.DataFrame: A DataFrame containing the CF target values for each NUTS-2 region.
    """
    print("Starting data preparation...")

    # --- 2. Load and combine the weather data (Input Features) ---
    # Get the list of NetCDF file paths from the data processing config
    era5_files = list(cfm['era5_data_raw'].values())
    
    print(f"Loading {len(era5_files)} ERA5 files specified in config...")

    datasets = [xr.open_dataset(f) for f in era5_files]
    weather_data = xr.merge(datasets, compat='override', join='inner')
    
    print("\nWeather data successfully loaded and merged.")
    print("Dimensions of raw data:", weather_data.dims)

    # --- 3. Flatten the spatial and level dimensions ---
    df_weather = weather_data.to_dataframe()

    df_weather = df_weather.rename(columns=cfm['era5_var_names'])
    
    vars_to_drop = [col for col in df_weather.columns if col not in cfm['era5_var_names'].values()]
    df_weather = df_weather.drop(columns=vars_to_drop)
    
    df_weather_flat = df_weather.unstack(level=['level', 'latitude', 'longitude'])
    print("Spatial data and levels unstacked into columns.")

    # Format column names into a single string: 't_500_47.00_5.00'
    df_weather_flat.columns = [
        f'{var}_{level}_{lat:.2f}_{lon:.2f}' for var, level, lat, lon in df_weather_flat.columns
    ]
    print(f"DataFrame flattened. Number of features: {len(df_weather_flat.columns)}")

    # --- Make the weather data index timezone-aware ---
    df_weather_flat.index = df_weather_flat.index.tz_localize('UTC')

    # --- 4. Load the target data (Output Labels) ---
    cf_file = Path(cfm['target_data_raw'])
    if not cf_file.exists():
        raise FileNotFoundError(f"CF file not found: {cf_file}")
        
    df_cf = pd.read_csv(cf_file, index_col='Date', parse_dates=True)
    df_cf.index = df_cf.index.tz_localize('UTC')
    print("\nCapacity factor (CF) data loaded.")

    # --- 5. Merge Features and Labels ---
    combined_data = pd.merge(df_weather_flat, df_cf, left_index=True, right_index=True, how='inner')
    print(f"Data merged. Found {len(combined_data)} overlapping timesteps.")

    # --- 6. Filter data to the specified time range from config ---
    start_date = cfm['time_start']
    end_date = cfm['time_end']
    print(f"Filtering data to range: {start_date} to {end_date}")
    combined_data = combined_data.loc[start_date:end_date]
    print(f"Final dataset contains {len(combined_data)} timesteps.")
    
    # Separate the data back into features (X) and labels (y)
    feature_columns = df_weather_flat.columns
    label_columns = df_cf.columns
    
    X = combined_data[feature_columns]
    y = combined_data[label_columns]

    print(f"Split complete. {X.shape[1]} features and {y.shape[1]} labels.")
    return X, y


if __name__ == '__main__':
    try:
        X_processed, y_processed = load_and_prepare_data()
        
        print("\n--- Results ---")
        print("Shape of features (X):", X_processed.shape)
        print("Shape of labels (y):", y_processed.shape)
        print("\nFirst 5 rows of features (only the first 3 columns):")
        print(X_processed.iloc[:, :3].head())
        print("\nFirst 5 rows of labels:")
        print(y_processed.head())
        
        # Ask the user if they want to save the data
        if input("\nSave processed data as Parquet files? (y/n): ").lower() == 'y':
            
            # Convert to float32 for ~50% smaller files
            X_processed = X_processed.astype('float32')
            y_processed = y_processed.astype('float32')
            print("Converting data to float32 for efficiency.")

            # Use pathlib for robust path creation
            processed_dir = Path(cfg['data_processed_dir'])
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as Parquet files
            features_path = processed_dir / f"{cfm['setup_name']}_features.parquet"
            labels_path = processed_dir / f"{cfm['setup_name']}_labels.parquet"

            X_processed.to_parquet(features_path)
            y_processed.to_parquet(labels_path)
            
            print(f"Processed data saved to {processed_dir}")
        else:
            print("Processed data not saved.")
            
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure your paths in '*.yaml' are correct.")
    except KeyError as e:
        print(f"\nError: Missing key {e} in a configuration file.")
        print("Please check your YAML files for completeness.")