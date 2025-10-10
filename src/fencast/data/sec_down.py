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
# load IFS data
ds_ifs = xr.open_zarr(cfg["gsWB2_ifs_data"], decode_timedelta=True)

# cut out region
new_ds_ifs = (
    ds_ifs
    .sel(latitude=slice(cfm["feature_region"]["lat_min"], cfm["feature_region"]["lat_max"]),
         longitude=slice(cfm["feature_region"]["lon_min"], cfm["feature_region"]["lon_max"]))
    [cfm["feature_variables"].keys()]
    .sel(level=cfm["feature_level"])
    .sel(time=ds_ifs['time'].dt.hour == 0)
)
print(new_ds_ifs)

#%% create dataset dictionary
timedeltas = cfm["mlwp_timedelta"]
ifs_datasets = {}
for td in timedeltas:
    ifs_datasets[td] = {var: new_ds_ifs[var][:, td, :, :] for var in new_ds_ifs.data_vars}

#%% download the datasets
for td in timedeltas:
    for var, ds in ifs_datasets[td].items():
        print(f"downloading variable: {var}, timedelta: {td}")
        output_path = cfg["data_raw_dir"] + f"/ifs_td{td:02d}_de_{var}.nc"
        ds.to_netcdf(output_path)
        print(f"saved to: {output_path}")

#%%