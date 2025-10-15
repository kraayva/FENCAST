#%% initial setup
from click import progressbar
import pandas as pd
import xarray as xr
import yaml
import numpy as np
import os
from pathlib import Path
from dask.diagnostics import ProgressBar
from itertools import zip_longest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
os.chdir(PROJECT_ROOT)

def load_config(name="global"):
    with open(f"configs/{name}.yaml", "r") as f:
        return yaml.safe_load(f)
    
cfg = load_config()
cfm = load_config("datapp_de")

#%%
# ---------------------------------------
# ------------- PANGU DATA --------------
# ---------------------------------------

# load PANGU data
ds_pangu = xr.open_zarr(cfg["gsWB2_pangu_data"], decode_timedelta=True)

# cut out region
new_ds_pangu = (
    ds_pangu
    .sel(latitude=slice(cfm["feature_region"]["lat_max"], cfm["feature_region"]["lat_min"]),
         longitude=slice(cfm["feature_region"]["lon_min"], cfm["feature_region"]["lon_max"]))
    [cfm["feature_variables"].keys()]
    .sel(level=cfm["feature_level"])
    .sel(time=ds_pangu['time'].dt.hour == 12)
)

# create dataset dictionary
timedeltas = cfm["mlwp_timedelta"]
pangu_datasets = {}
for td in timedeltas:
    pangu_datasets[td] = {var: new_ds_pangu[var][:, td, :, :] for var in new_ds_pangu.data_vars}
#%% download the datasets
for td in timedeltas:
    for var, ds in pangu_datasets[td].items():
        print(f"downloading variable: {var}, timedelta: {td}")
        output_path = cfg["data_raw_dir"] + f"/pangu_td{td:02d}_de_{var}.nc"
        if not os.path.exists(output_path):
            # Load data into memory first (with progress)
            print("📊 Loading data into memory...")
            with ProgressBar():
                # Compute the data to trigger actual download from zarr
                computed_data = ds.compute()
            
            print("💾 Writing to NetCDF file...")
            # Now write the computed data to file
            computed_data.to_netcdf(output_path)
            print(f"saved to: {output_path}")
        else:
            print(f"file already exists: {output_path}, skipping.")

#%%