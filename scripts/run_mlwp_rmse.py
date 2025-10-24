# /scripts/run_mlwp_rmse.py
"""
Entry point script for calculating weather prediction RMSE from MLWP models.

This script compares MLWP weather model predictions against ERA5 reference data
and calculates normalized RMSE metrics across different forecast lead times.
"""

import argparse
from pathlib import Path

from fencast.utils.paths import load_config, PROJECT_ROOT
from fencast.utils.tools import setup_logger
from fencast.mlwp_analysis import calculate_mlwp_weather_rmse


def main():
    parser = argparse.ArgumentParser(
        description="Calculate weather prediction RMSE for MLWP models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('config', nargs='?', 
                        default='datapp_de', 
                        help='Configuration file name (default: datapp_de)')
    parser.add_argument('--output-file', '-o', 
                       default='weather_rmse_results.csv',
                       help='Output CSV file name')
    parser.add_argument('--mlwp-model', '-m',
                       default='pangu',
                       help='MLWP model name to evaluate')
    parser.add_argument('--timedeltas', nargs='+', type=int,
                       default=None,
                       help='Forecast lead times in days')
    parser.add_argument('--levels', nargs='+', type=int,
                       default=[1000, 850, 500],
                       help='Pressure levels to evaluate')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("mlwp_weather_rmse")
    logger.info(f"Starting weather RMSE calculation for config: {args.config}")
    
    # Load configuration
    config = load_config(args.config)
    setup_name = config.get('setup_name', 'default_setup')
    
    # Set output path in results directory with MLWP model name
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Create filename with MLWP model names
    mlwp_name = args.mlwp_model
    output_filename = f"weather_rmse_{mlwp_name}.csv"
    output_path = results_dir / output_filename
    
    # Calculate weather RMSE
    try:
        calculate_mlwp_weather_rmse(
            config=config,
            output_file=output_path,
            mlwp_name=args.mlwp_model,
            timedeltas=args.timedeltas,
            pressure_levels=args.levels
        )
        logger.info(f"Weather RMSE calculation completed. Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error during weather RMSE calculation: {e}")
        raise


if __name__ == "__main__":
    main()