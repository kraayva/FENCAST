import pandas as pd
import xarray as xr
import yaml
import numpy as np

def load_config(name="global"):
    with open(f"configs/{name}.yaml", "r") as f:
        return yaml.safe_load(f)
    
cfg = load_config()
cfm = load_config("datapp_de")

# ---------------------------------------
# ------------- TARGET DATA -------------
# ---------------------------------------

# load data
data_file = cfg["data_raw_dir"] + "/cfr_NUTS2-DE.csv"
df = pd.read_csv(data_file)

# filter for 12:00
df_day = df[df['Date'].str.endswith('12:00:00')].copy()
df_day['Date'] = df_day['Date'].str[:-9]

# filter time range
df_day = df_day[(df_day['Date'] >= cfm["time_start"]) & (df_day['Date'] <= cfm["time_end"])]

# save target data
data_file_out = cfg["data_processed_dir"] + "/cfr_NUTS2-DE_daily.csv"
df_day.to_csv(data_file_out, index=False)

# ---------------------------------------
# ------------- ERA5 DATA ---------------
# ---------------------------------------

# load ERA5 data
ds_era5 = xr.open_zarr(cfg["era5_dataset_url"])

# cut out Germany
ds_era5 = ds_era5.sel(latitude=slice(cfm["ERA5_region"]["lat_min"], cfm["ERA5_region"]["lat_max"]),
                       longitude=slice(cfm["ERA5_region"]["lon_min"], cfm["ERA5_region"]["lon_max"]))
