import xarray as xr
from pathlib import Path
import os

from fencast.utils.paths import RAW_DATA_DIR

zeros = {}
nans = {}

for variable in ['u_component_of_wind', 'v_component_of_wind', 'specific_humidity', 'temperature']:
    dataset_names = [f.name for f in Path(RAW_DATA_DIR).glob(f"pangu*{variable}.nc")]

    for ds_name in dataset_names:
        if ds_name == "pangu_td35_de_temperature.nc":
            continue
        print(ds_name)
        print("...")
        ds = xr.open_dataset(RAW_DATA_DIR / ds_name)
        data = ds[variable].values
        zeros[ds_name] = (data == 0).sum()
        nans[ds_name] = xr.ufuncs.isnan(data).sum()
        ds.close()


print("-------------")
print("----Zeros----")
print("-------------")
for d in dataset_names:
    print(f"{d}: {zeros[d]}")

print("-------------")
print("-----nans----")
print("-------------")
for d in dataset_names:
    print(f"{d}: {nans[d]}")
