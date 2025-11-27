import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from pathlib import Path
import json

from fencast.utils.tools import setup_logger, get_latest_study_dir
from fencast.utils.paths import load_config


def create_region_scp_rmse_maps(time_deltas: list[int] = list(range(1, 10)), 
                                config: dict = None, logger=None, 
                                study_dir: Path = None, 
                                model_name: str = "final_model", 
                                mlwp_name: str = "pangu"):

    output_dir = study_dir / model_name / "region_scp_rmse_maps"
    os.makedirs(output_dir, exist_ok=True)

    if not config:
        logger.error("No config provided.")
        return

    # load metrics from json files
    metrics = {}

    for i in time_deltas:
        i_str = str(i).zfill(2)
        p = Path(f"{study_dir}/{model_name}/mlwp_evaluation/{mlwp_name}/metrics_td{i_str}.json")
        with p.open("r", encoding="utf-8") as f:
            metrics_td = json.load(f)
        metrics[f"td{i_str}"] = metrics_td

    # extract region codes
    region_codes = metrics[f"td01"]["region_metrics"].keys()

    # load shapefile for NUTS regions and filter for Germany and level 2
    shape_file = gpd.read_file("data/regions/NUTS_RG_20M_2024_3035.gpkg")

    NUTS2 = shape_file[shape_file["LEVL_CODE"] == 2]
    NUTS2DE = NUTS2.where(NUTS2["CNTR_CODE"] == "DE")
    NUTS2DE = NUTS2DE.dropna(how="all")
    NUTS2DE = NUTS2DE.drop(columns=["MOUNT_TYPE", "URBN_TYPE", "COAST_TYPE"])
    NUTS2DE.head()

    NUTS2DE_metrics = NUTS2DE.copy()
    for td in time_deltas:
        td_str = "TD" + str(td).zfill(2)
        NUTS2DE_metrics[f"{td_str}_MAE"] = np.nan
        NUTS2DE_metrics[f"{td_str}_RMSE"] = np.nan

        for rc in region_codes:
            NUTS2DE_metrics.loc[NUTS2DE_metrics["NUTS_ID"] == rc, f"{td_str}_MAE"] = metrics[f"td{td_str[-2:]}"]["region_metrics"][f"{rc}"]["mae"]
            NUTS2DE_metrics.loc[NUTS2DE_metrics["NUTS_ID"] == rc, f"{td_str}_RMSE"] = metrics[f"td{td_str[-2:]}"]["region_metrics"][f"{rc}"]["rmse"]

    # create plots for each combination of time delta and region
    for td in time_deltas:
        td_str = "TD" + str(td).zfill(2)
        fig_path = output_dir / f"{mlwp_name}_{td_str}_RMSE.png"
        fig, ax = plt.subplots(1, 1, figsize=(8, 10))
        NUTS2DE_metrics.plot(
            column=f"{td_str}_RMSE",
            edgecolor="black",
            ax=ax,
            cmap="coolwarm",
            legend=True,
            missing_kwds={"color": "lightgrey", "label": "No data"}
        )
        plt.title(f"{mlwp_name} forecast lead time {td} days, SCF RMSE")
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
    logger.info(f"Region SCF RMSE plots created and saved under {output_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create NUTS-2 region SCF RMSE plots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('config', nargs='?', default='datapp_de', help='Configuration file name (default: datapp_de)')
    parser.add_argument('--time-deltas', nargs='+', type=int, default=list(range(1, 11)),
                        help='List of time deltas to plot (default: 1-10)')
    
    parser.add_argument('--study-name', default='latest',
                        help='Study name to load results from (default: "latest")')
    parser.add_argument('--mlwp', default=None,
                        help='Name of the MLWP to use for labeling (default: "pangu, ifs")')
    parser.add_argument('--model-name', default='final_model',
                        help='Model directory name to use (default: final_model)')
    args = parser.parse_args()

    cfm = load_config(args.config)

    if args.study_name == 'latest':
        results_parent_dir = Path("results") / cfm.get("setup_name")
        study_dir = get_latest_study_dir(results_parent_dir)
    else:
        study_dir = Path("results") / cfm.get("setup_name") / args.study_name

    logger = setup_logger("region_scp_rmse_map")
    logger.info(f"Using study directory: {study_dir}")

    if not args.mlwp:
        mlwp_names = ["pangu", "ifs"]
    else:
        mlwp_names = [args.mlwp]

    for n in mlwp_names:
        create_region_scp_rmse_maps(time_deltas=args.time_deltas, 
                                    config=cfm, 
                                    logger=logger, 
                                    study_dir=study_dir, 
                                    model_name=args.model_name,
                                    mlwp_name=n)
