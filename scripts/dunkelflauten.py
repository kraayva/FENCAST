# scripts/dunkelflauten.py

'''Script to analyze and visualize dunkelflauten (periods of low wind energy production).'''

import argparse
import pandas as pd
import xarray as xr
from fencast.utils.paths import load_config, PROJECT_ROOT
from fencast.utils.tools import setup_logger, get_latest_study_dir

logger = setup_logger("evaluation")

def solar_dunkelflauten(scf_data: pd.DataFrame, threshold: float = None) -> pd.DataFrame:
    """Identify solar dunkelflauten periods based on SCF data.
    
    Args:
        scf_data (pd.DataFrame): DataFrame with solar capacity factor data for NUTS2 regions
        threshold (float, optional): Threshold below which a dunkelflaute is identified.
                                     If None, uses the 10th percentile of the data.
    """

    if threshold is None:
        threshold = scf_data.quantile(0.1).values[0]
        logger.info(f"Using 10th percentile as threshold: {threshold:.4f}")
    