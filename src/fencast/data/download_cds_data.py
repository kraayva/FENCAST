import cdsapi
import pandas as pd
import zipfile
from pathlib import Path
from shutil import rmtree
import sys

here = Path(__file__).resolve().parent
repo_root = here.parent.parent.parent

dataset = "sis-energy-derived-reanalysis"
client = cdsapi.Client()

months = [f"{month:02d}" for month in range(1, 13)]
years = [str(year) for year in range(2000, 2009)]

tmp_dir = Path(f"{repo_root}/data/tmp")
target_dir = Path(f"{repo_root}/data/cfr_NUTS2-DE")
tmp_dir.mkdir(parents=True, exist_ok=True)
target_dir.mkdir(parents=True, exist_ok=True)

variables = ["wind_power_generation_onshore"]#, "solar_photovoltaic_power_generation"]

#for year in years:
#    for month in months:
for var in variables:
    request = {
        "variable": [var],
        "spatial_aggregation": [
            #"original_grid",
            #"country_level",
            "sub_country_level"
        ], ## NUTS2 is sub-country level
        "energy_product_type": ["capacity_factor_ratio"],
        "temporal_aggregation": ["hourly"],
        #"year": [year],
        #"month": [month]
    }
    print(f"Downloading {dataset} for {var}...")
    zip_path = Path(tmp_dir / (dataset + "_" + var + ".zip"))
    client.retrieve(dataset, request).download(zip_path)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(tmp_dir)

    # Filter CSV for DE only
    csv_files = list(tmp_dir.glob("*.csv"))
    csv_path = next((p for p in csv_files if "_PhM02" in p.name), csv_files[0])
    df = pd.read_csv(csv_path, header=52)
    df_de = df.filter(regex="^DE", axis=1)
    df_de = pd.concat([df.iloc[:, 0], df_de], axis=1)

    # Save only DE columns
    out_path = Path(target_dir / (f"{var}_cfr_NUTS2-DE.csv"))
    df_de.to_csv(out_path, index=False) 

    # Clean up tmp directory
    rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

