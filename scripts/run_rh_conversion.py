# scripts/run_data_processing.py

import argparse
from pathlib import Path
import numpy as np # Import numpy for saving arrays
import xarray as xr

from fencast.utils.paths import RAW_DATA_DIR
from fencast.data_processing import specific_humidity_to_relative_humidity
from fencast.utils.tools import setup_logger
from fencast.utils.parser import get_parser

def create_specific_humidity_files(file_prefix: str = "era5_de"):
    """
    Create relative humidity files from specific humidity raw data.
    
    Args:
        file_prefix (str): Prefix for feature data files (e.g., "era5_de", "ifs_de", "pangu_de").
    """

    logger = setup_logger("specific_humidity_conversion")
    
    try:
        logger.info(f"Processing {file_prefix}...")
        
        # Load temperature and specific humidity data
        temp_file = RAW_DATA_DIR / f"{file_prefix}_temperature.nc"
        q_file = RAW_DATA_DIR / f"{file_prefix}_specific_humidity.nc"
        
        if not temp_file.exists() or not q_file.exists():
            logger.error(f"Required files not found for {file_prefix}")
            return
            
        data_t_raw = xr.open_dataset(temp_file)
        data_q_raw = xr.open_dataset(q_file)
        
        # Get data arrays
        data_t = data_t_raw['temperature'].values
        data_q = data_q_raw['specific_humidity'].values
        
        # Get pressure levels and broadcast to match data shape
        levels = data_t_raw['level'].values
        
        if file_prefix == "era5_de":
            # ERA5 shape: (time, level, lat, lon)
            # Broadcast pressure to shape: (1, level, 1, 1)
            data_p = levels[np.newaxis, :, np.newaxis, np.newaxis]
            data_p = np.broadcast_to(data_p, data_t.shape)
        else:
            # MLWP shape: (time, prediction_timedelta, level, lat, lon)
            # Broadcast pressure to shape: (1, 1, level, 1, 1)
            data_p = levels[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
            data_p = np.broadcast_to(data_p, data_t.shape)

        # Convert specific humidity to relative humidity
        logger.info("Converting specific humidity to relative humidity...")
        rh_values = specific_humidity_to_relative_humidity(
            q=data_q,
            T=data_t,
            p=data_p
        )

        # Create new xarray Dataset with relative humidity
        # Use copy(deep=False) to avoid copying data, then drop and add variables
        rh_ds = data_q_raw.drop_vars('specific_humidity')
        
        if file_prefix == "era5_de":
            rh_ds['relative_humidity'] = (('time', 'level', 'latitude', 'longitude'), rh_values)
        else:
            rh_ds['relative_humidity'] = (('time', 'prediction_timedelta', 'level', 'latitude', 'longitude'), rh_values)

        # Save the new dataset
        output_path = RAW_DATA_DIR / f"{file_prefix}_relative_humidity.nc"
        logger.info(f"Saving to {output_path}...")
        rh_ds.to_netcdf(output_path)
                
        logger.info(f"Conversion complete for {file_prefix}. Saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error processing {file_prefix}: {e}")
        raise
    

if __name__ == '__main__':
    parser = get_parser(['file_prefix'], description='Run specific humidity to relative humidity conversion.')
    args = parser.parse_args()

    if args.file_prefix == "all":
        prefixes = ["era5_de", "ifs_de", "pangu_de"]
        for prefix in prefixes:
            create_specific_humidity_files(file_prefix=prefix)
    else:
        create_specific_humidity_files(file_prefix=args.file_prefix)
    