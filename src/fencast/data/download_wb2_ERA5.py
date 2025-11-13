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
cfm = load_config("conf_02")

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
print(era5_datasets)
# download the datasets
for var, ds in era5_datasets.items():
    output_path = cfg["data_raw_dir"] + f"/era5_de_{var}.nc"
    ds.to_netcdf(output_path)
