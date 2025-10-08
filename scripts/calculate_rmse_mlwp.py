# scripts/calculate_rmse_mlwp.py

import numpy as np
import pandas as pd
import xarray as xr
import argparse
from pathlib import Path
from sklearn.metrics import mean_squared_error

# Import our custom modules
from fencast.utils.paths import load_config, PROJECT_ROOT, RAW_DATA_DIR
from fencast.utils.tools import setup_logger

logger = setup_logger("mlwp_rmse_calculation")

def load_era5_data(config: dict, var_name: str) -> xr.Dataset:
    """Loads ERA5 reference data for a specific variable."""
    era5_file = RAW_DATA_DIR / f"era5_de_{var_name}.nc"
    if not era5_file.exists():
        raise FileNotFoundError(f"ERA5 reference file not found: {era5_file}")
    
    logger.info(f"Loading ERA5 data: {era5_file}")
    return xr.open_dataset(era5_file)

def load_mlwp_data(config: dict, mlwp_name: str, timedelta_str: str, var_name: str) -> xr.Dataset:
    """Loads MLWP prediction data for a specific variable and timedelta."""
    mlwp_file = RAW_DATA_DIR / f"{mlwp_name}_{timedelta_str}_de_{var_name}.nc"
    if not mlwp_file.exists():
        raise FileNotFoundError(f"MLWP prediction file not found: {mlwp_file}")
    
    logger.info(f"Loading MLWP data: {mlwp_file}")
    return xr.open_dataset(mlwp_file)

def calculate_variable_rmse(era5_data: xr.Dataset, mlwp_data: xr.Dataset, var_name: str, config: dict) -> dict:
    """
    Calculates normalized RMSE between ERA5 and MLWP data for a specific variable across all levels.
    Each variable/level combination is normalized by ERA5 statistics to make them comparable.
    
    Returns:
        dict: Normalized RMSE values for each pressure level
    """
    # Get the renamed variable name from config
    var_key = var_name
    if var_name in config['feature_var_names']:
        var_key = config['feature_var_names'][var_name]
    
    # Align datasets by time - find common timestamps
    era5_times = pd.to_datetime(era5_data.time.values)
    mlwp_times = pd.to_datetime(mlwp_data.time.values)
    
    # Remove timezone info if present
    if era5_times.tz is not None:
        era5_times = era5_times.tz_localize(None)
    if mlwp_times.tz is not None:
        mlwp_times = mlwp_times.tz_localize(None)
    
    common_times = era5_times.intersection(mlwp_times)
    
    if len(common_times) == 0:
        logger.warning(f"No overlapping timestamps found for {var_name}")
        return {}
    
    logger.info(f"Found {len(common_times)} overlapping timestamps for {var_name}")
    
    # Select common time range
    era5_aligned = era5_data.sel(time=common_times)
    mlwp_aligned = mlwp_data.sel(time=common_times)
    
    # Get the variable data (use original variable name for data access)
    if var_key in era5_aligned:
        era5_var = era5_aligned[var_key]
        mlwp_var = mlwp_aligned[var_key]
    else:
        # Fallback to original variable name
        era5_var = era5_aligned[var_name]
        mlwp_var = mlwp_aligned[var_name]
    
    rmse_by_level = {}
    
    # Calculate RMSE for each pressure level
    levels = config.get('feature_level', [])
    for level in levels:
        try:
            # Select data for this pressure level
            era5_level = era5_var.sel(level=level)
            mlwp_level = mlwp_var.sel(level=level)
            
            # Flatten spatial dimensions
            era5_flat = era5_level.values.flatten()
            mlwp_flat = mlwp_level.values.flatten()
            
            # Remove NaN values
            valid_mask = ~(np.isnan(era5_flat) | np.isnan(mlwp_flat))
            era5_valid = era5_flat[valid_mask]
            mlwp_valid = mlwp_flat[valid_mask]
            
            if len(era5_valid) > 0:
                # Calculate ERA5 normalization statistics
                era5_mean = np.mean(era5_valid)
                era5_std = np.std(era5_valid)
                
                # Normalize both datasets using ERA5 statistics
                if era5_std > 0:
                    era5_normalized = (era5_valid - era5_mean) / era5_std
                    mlwp_normalized = (mlwp_valid - era5_mean) / era5_std
                    
                    # Calculate RMSE on normalized data (dimensionless, comparable across variables)
                    normalized_rmse = np.sqrt(mean_squared_error(era5_normalized, mlwp_normalized))
                    
                    # Store normalized RMSE (dimensionless, ~0-3 range typically)
                    rmse_by_level[f"{var_name}_{level}hPa"] = normalized_rmse
                    
                    logger.info(f"  {var_name} at {level} hPa: Normalized RMSE = {normalized_rmse:.4f} (std units)")
                else:
                    logger.warning(f"  {var_name} at {level} hPa: Zero standard deviation, cannot normalize")
                    rmse_by_level[f"{var_name}_{level}hPa"] = np.nan
            else:
                logger.warning(f"  No valid data for {var_name} at {level} hPa")
                rmse_by_level[f"{var_name}_{level}hPa"] = np.nan
                
        except Exception as e:
            logger.error(f"Error calculating RMSE for {var_name} at {level} hPa: {e}")
            rmse_by_level[f"{var_name}_{level}hPa"] = np.nan
    
    return rmse_by_level

def calculate_variable_rmse_summary(result_row: dict, feature_var_names: dict, feature_levels: list) -> dict:
    """
    Calculate RMSE for each variable averaged across all its pressure levels.
    
    Args:
        result_row: Dictionary containing individual variable RMSE values
        feature_var_names: Dictionary of variable names from config
        feature_levels: List of pressure levels from config
    
    Returns:
        dict: Variable-level RMSE values
    """
    variable_rmse = {}
    
    for var_name in feature_var_names.keys():
        var_rmse_values = []
        
        # Collect RMSE values for this variable across all levels
        for level in feature_levels:
            rmse_key = f"{var_name}_{level}hPa"
            if rmse_key in result_row and not np.isnan(result_row[rmse_key]):
                var_rmse_values.append(result_row[rmse_key])
        
        if len(var_rmse_values) > 0:
            avg_rmse = np.mean(var_rmse_values)
            variable_rmse[f"{var_name}_avg_rmse"] = avg_rmse
            logger.info(f"  {var_name} average RMSE (across {len(var_rmse_values)} levels): {avg_rmse:.4f}")
        else:
            variable_rmse[f"{var_name}_avg_rmse"] = np.nan
            logger.warning(f"  No valid RMSE values found for {var_name}")
    
    return variable_rmse

def calculate_total_rmse(result_row: dict, feature_var_names: dict, feature_levels: list) -> float:
    """
    Calculate overall RMSE across all variables and pressure levels.
    
    Args:
        result_row: Dictionary containing individual variable RMSE values
        feature_var_names: Dictionary of variable names from config
        feature_levels: List of pressure levels from config
    
    Returns:
        float: Total RMSE across all variables and levels, or NaN if no valid data
    """
    rmse_values = []
    
    # Collect all individual RMSE values
    for var_name in feature_var_names.keys():
        for level in feature_levels:
            rmse_key = f"{var_name}_{level}hPa"
            if rmse_key in result_row and not np.isnan(result_row[rmse_key]):
                rmse_values.append(result_row[rmse_key])
    
    if len(rmse_values) > 0:
        # Calculate mean of all RMSE values
        total_rmse = np.mean(rmse_values)
        logger.info(f"  Total RMSE (mean across {len(rmse_values)} variables/levels): {total_rmse:.4f}")
        return total_rmse
    else:
        logger.warning("  No valid RMSE values found for total calculation")
        return np.nan

def calculate_mlwp_rmse(config_name: str, output_file: str):
    """
    Main function to calculate RMSE between MLWP predictions and ERA5 reference data.
    """
    logger.info("--- Starting MLWP Weather Prediction RMSE Calculation ---")
    
    # Load configuration
    config = load_config(config_name)
    setup_name = config.get('setup_name', 'default_setup')
    
    # Get experiment parameters
    mlwp_names = config.get('mlwp_names', [])
    mlwp_timedeltas = config.get('mlwp_timedelta', [])
    feature_var_names = config.get('feature_var_names', {})
    
    if not mlwp_names or not mlwp_timedeltas:
        logger.error("Config keys 'mlwp_names' or 'mlwp_timedelta' are missing or empty.")
        return
        
    if not feature_var_names:
        logger.error("Config key 'feature_var_names' is missing or empty.")
        return
    
    logger.info(f"Calculating RMSE for {len(mlwp_names)} MLWP models, {len(mlwp_timedeltas)} timedeltas, {len(feature_var_names)} variables")
    
    all_results = []
    
    # Loop through all combinations
    for mlwp_name in mlwp_names:
        for td in mlwp_timedeltas:
            td_str = f"td{td:02d}"
            logger.info(f"\n--- Processing {mlwp_name} {td_str} ---")
            
            result_row = {
                'mlwp_model': mlwp_name,
                'timedelta_days': td,
                'timedelta_hours': td * 24
            }
            
            for var_name in feature_var_names.keys():
                logger.info(f"Processing variable: {var_name}")
                
                try:
                    # Load ERA5 reference data
                    era5_data = load_era5_data(config, var_name)
                    
                    # Load MLWP prediction data
                    mlwp_data = load_mlwp_data(config, mlwp_name, td_str, var_name)
                    
                    # Calculate RMSE for this variable
                    variable_rmse = calculate_variable_rmse(era5_data, mlwp_data, var_name, config)
                    
                    # Add to result row
                    result_row.update(variable_rmse)
                    
                    # Close datasets to free memory
                    era5_data.close()
                    mlwp_data.close()
                    
                except FileNotFoundError as e:
                    logger.error(f"Skipping {var_name} for {mlwp_name} {td_str}: {e}")
                    # Add NaN values for missing data
                    for level in config.get('feature_level', []):
                        result_row[f"{var_name}_{level}hPa"] = np.nan
                    # Add NaN for variable average
                    result_row[f"{var_name}_avg_rmse"] = np.nan
                        
                except Exception as e:
                    logger.error(f"Error processing {var_name} for {mlwp_name} {td_str}: {e}")
                    # Add NaN values for error cases
                    for level in config.get('feature_level', []):
                        result_row[f"{var_name}_{level}hPa"] = np.nan
                    # Add NaN for variable average
                    result_row[f"{var_name}_avg_rmse"] = np.nan
            
            # Calculate variable-level RMSE (averaged across height levels)
            variable_rmse = calculate_variable_rmse_summary(result_row, feature_var_names, config.get('feature_level', []))
            result_row.update(variable_rmse)
            
            # Calculate total RMSE across all variables and levels
            total_rmse = calculate_total_rmse(result_row, feature_var_names, config.get('feature_level', []))
            result_row['total_rmse'] = total_rmse
            
            all_results.append(result_row)
    
    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(all_results)
    
    # Save results
    output_path = PROJECT_ROOT / output_file
    results_df.to_csv(output_path, index=False)
    logger.info(f"\nRMSE results saved to: {output_path}")
    
    # Print summary
    logger.info(f"\nSummary:")
    logger.info(f"  Total rows: {len(results_df)}")
    logger.info(f"  Columns: {list(results_df.columns)}")
    
    # Show sample of results
    if len(results_df) > 0:
        logger.info(f"\nSample results:")
        print(results_df.head())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate RMSE between MLWP predictions and ERA5 reference data.')
    parser.add_argument(
        '--config', '-c', 
        default='datapp_de',
        help='Configuration file name (default: datapp_de)'
    )
    parser.add_argument(
        '--output', '-o',
        default='results/mlwp_weather_rmse.csv',
        help='Output CSV file path relative to project root (default: results/mlwp_weather_rmse.csv)'
    )
    
    args = parser.parse_args()
    
    calculate_mlwp_rmse(
        config_name=args.config,
        output_file=args.output
    )
