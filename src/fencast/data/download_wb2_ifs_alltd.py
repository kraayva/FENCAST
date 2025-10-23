#%% initial setup
import xarray as xr
import yaml
import os
from pathlib import Path
from dask.diagnostics import ProgressBar

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
os.chdir(PROJECT_ROOT)

def load_config(name="global"):
    with open(f"configs/{name}.yaml", "r") as f:
        return yaml.safe_load(f)
    
cfg = load_config()
cfm = load_config("datapp_de")

#%%
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
print(new_ds_ifs)

# Select all required timedeltas at once for each variable
ifs_datasets = {}
timedeltas = cfm["ifs_timedelta"]
for var in cfm["feature_variables"].keys():
    # Select all timedeltas for the variable
    ifs_datasets[var] = new_ds_ifs[var][:, timedeltas, :, :]
#%% download the datasets
for var in ifs_datasets.keys():
    print(f"downloading variable: {var}")
    output_path = cfg["data_raw_dir"] + f"/ifs_{var}.nc"
    ds = ifs_datasets[var]
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