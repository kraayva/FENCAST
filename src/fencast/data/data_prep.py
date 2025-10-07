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
# ------------- TARGET DATA -------------
# ---------------------------------------

# load data
target_data_raw = cfm["target_data_raw"]
df = pd.read_csv(target_data_raw)

# filter for 12:00
df_day = df[df['Date'].str.endswith('12:00:00')].copy()
df_day['Date'] = df_day['Date'].str[:-9]

# filter time range
df_day = df_day[(df_day['Date'] >= cfm["time_start"]) & (df_day['Date'] <= cfm["time_end"])]

# save target data
data_file_out = cfg["data_processed_dir"] + "/cfr_NUTS2-DE_daily.csv"
df_day.to_csv(data_file_out, index=False)

#%%
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
    .sel(level=cfm["feature_level"])
    .sel(time=ds_era5['time'].dt.hour == 12)
)

#%% create dataset dictionary
era5_datasets = {var: new_ds_era5[var] for var in new_ds_era5.data_vars}
print(era5_datasets)
#%% download the datasets
for var, ds in era5_datasets.items():
    output_path = cfg["data_raw_dir"] + f"/era5_de_{var}.nc"
    ds.to_netcdf(output_path)

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

#%% create dataset dictionary
timedeltas = cfm["mlwp_timedelta"]
pangu_datasets = {}
for td in timedeltas:
    pangu_datasets[td] = {var: new_ds_pangu[var][:, td, :, :] for var in new_ds_pangu.data_vars}
#%% download the datasets
for td in timedeltas:
    for var, ds in pangu_datasets[td].items():
        print(f"downloading variable: {var}, timedelta: {td}")
        output_path = cfg["data_raw_dir"] + f"/pangu_td{td:02d}_de_{var}.nc"
        ds.to_netcdf(output_path)
        print(f"saved to: {output_path}")

#%%