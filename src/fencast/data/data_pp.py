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

# ---------------------------------------
# ------------- ERA5 DATA ---------------
# ---------------------------------------

# load ERA5 data
ds_era5 = xr.open_zarr(cfm["era5_dataset_url"])

# cut out region
ds_era5 = ds_era5.sel(latitude=slice(cfm["feature_region"]["lat_max"], cfm["feature_region"]["lat_min"]),
                       longitude=slice(cfm["feature_region"]["lon_min"], cfm["feature_region"]["lon_max"]))

ds_era5 = ds_era5[cfm["feature_variables"].keys()] # filter variables
ds_era5 = ds_era5.sel(level=cfm["feature_level"]) # filter levels
ds_era5 = ds_era5.sel(time=ds_era5['time'].dt.hour == 12) # filter time to 12:00 only

output_path = cfg["data_processed_dir"] + "/era5_de.zarr"
ds_era5.to_zarr(output_path)
