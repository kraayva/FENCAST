# /src/fencast/data/download_wb2_ifs_alltd.py

# initial setup
import xarray as xr
import yaml
import os
from pathlib import Path
from dask.diagnostics import ProgressBar
from fencast.utils.tools import setup_logger

logger = setup_logger("ifs_download")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
os.chdir(PROJECT_ROOT)

def load_config(name="global"):
    with open(f"configs/{name}.yaml", "r") as f:
        return yaml.safe_load(f)
    
cfg = load_config()
cfm = load_config("datapp_de")

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
    .sel(level=cfm["feature_level"])
    .sel(time=ds_ifs['time'].dt.hour == 12)
)
logger.info(f"Loaded IFS data: {new_ds_ifs}")

# Select all required timedeltas at once for each variable
ifs_datasets = {}
timedeltas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for var in ['specific_humidity']:
    # Select all timedeltas for the variable
    ifs_datasets[var] = new_ds_ifs[var][:, timedeltas, :, :]
# download the datasets
for var in ifs_datasets.keys():
    logger.info(f"downloading variable: {var}")
    output_path = cfg["data_raw_dir"] + f"/ifs_new/ifs_{var}.nc"
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
