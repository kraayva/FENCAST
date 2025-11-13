# initial setup
from click import progressbar
import pandas as pd
import xarray as xr
import yaml
import numpy as np
import os
from pathlib import Path
from dask.diagnostics import ProgressBar
from itertools import zip_longest

from fencast.utils.tools import setup_logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
os.chdir(PROJECT_ROOT)

def load_config(name="global"):
    with open(f"configs/{name}.yaml", "r") as f:
        return yaml.safe_load(f)
    
cfg = load_config()
cfm = load_config("conf_02")

logger = setup_logger("wb2_download")

# ---------------------------------------
# ------------- ERA5 DATA ---------------
# ---------------------------------------

# load ERA5 data
ds_era5 = xr.open_zarr(cfm["era5_dataset_url"])

# cut out region
new_ds_era5 = (
    ds_era5
    .sel(latitude=slice(cfm["feature_region"]["lat_max"], cfm["feature_region"]["lat_min"]),
         longitude=slice(cfm["feature_region"]["lon_min"], cfm["feature_region"]["lon_max"]))
    [cfm["feature_variables"].keys()]
    .sel(time=ds_era5['time'].dt.hour == 12)
)

# create dataset dictionary
era5_datasets = {var: new_ds_era5[var] for var in new_ds_era5.data_vars}
logger.info(era5_datasets)
# download the datasets
for var, ds in era5_datasets.items():
    output_path = cfg["data_raw_dir"] + f"/era5_de_{var}.nc"
    # skip if existing:
    if Path(output_path).exists():
        logger.info(f"File {output_path} already exists. Skipping download.")
        continue
    ds.to_netcdf(output_path)


# ---------------------------------------
# ------------- PANGU DATA --------------
# ---------------------------------------

# load PANGU data

timedeltas = cfm["mlwp_timedeltas"]
ds_pangu = xr.open_zarr(cfg["gsWB2_pangu_data"], decode_timedelta=True)

# cut out region
new_ds_pangu = (
    ds_pangu
    .sel(latitude=slice(cfm["feature_region"]["lat_max"], cfm["feature_region"]["lat_min"]),
         longitude=slice(cfm["feature_region"]["lon_min"], cfm["feature_region"]["lon_max"]))
    [cfm["feature_variables"].keys()]
    .sel(time=ds_pangu['time'].dt.hour == 12)
)

# Select all required timedeltas at once for each variable
pangu_datasets = {}
for var in cfm["feature_variables"].keys():
    # Select all timedeltas for the variable
    pangu_datasets[var] = new_ds_pangu[var][:, timedeltas['pangu'], :, :, :]
# download the datasets
for var in pangu_datasets.keys():
    logger.info(f"downloading variable: {var}")
    output_path = cfg["data_raw_dir"] + f"/pangu_de_{var}.nc"
    ds = pangu_datasets[var]
    if not os.path.exists(output_path):
        # Load data into memory first (with progress)
        logger.info("Loading data into memory...")
        with ProgressBar():
            # Compute the data to trigger actual download from zarr
            computed_data = ds.compute()

        logger.info("Writing to NetCDF file...")
        # Now write the computed data to file
        computed_data.to_netcdf(output_path)
        logger.info(f"saved to: {output_path}")
    else:
        logger.info(f"file already exists: {output_path}, skipping.")


# ---------------------------------------
# ------------- IFS DATA ----------------
# ---------------------------------------

# load IFS data
ds_ifs = xr.open_zarr(cfg["gsWB2_ifs_data"], decode_timedelta=True)

# cut out region
new_ds_ifs = (
    ds_ifs
    .sel(latitude=slice(cfm["feature_region"]["lat_min"], cfm["feature_region"]["lat_max"]),
         longitude=slice(cfm["feature_region"]["lon_min"], cfm["feature_region"]["lon_max"]))
    [cfm["feature_variables"].keys()]
    .sel(time=ds_ifs['time'].dt.hour == 12)
)
logger.info(f"Loaded IFS data: {new_ds_ifs}")

# Select all required timedeltas at once for each variable
ifs_datasets = {}
for var in cfm["feature_variables"].keys():
    # Select all timedeltas for the variable
    ifs_datasets[var] = new_ds_ifs[var][:, timedeltas['ifs'], :, :, :]
# download the datasets
for var in ifs_datasets.keys():
    logger.info(f"downloading variable: {var}")
    output_path = cfg["data_raw_dir"] + f"/ifs_de_{var}.nc"
    ds = ifs_datasets[var]
    if not os.path.exists(output_path):
        # Load data into memory first (with progress)
        logger.info("Loading data into memory...")
        with ProgressBar():
            # Compute the data to trigger actual download from zarr
            computed_data = ds.compute()
        
        logger.info("Writing to NetCDF file...")
        # Now write the computed data to file
        computed_data.to_netcdf(output_path)
        logger.info(f"saved to: {output_path}")
    else:
        logger.info(f"file already exists: {output_path}, skipping.")

logger.info("All data processed.")
