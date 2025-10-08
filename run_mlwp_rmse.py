#!/usr/bin/env python3
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
    parser.add_argument('config', help='Configuration file name (e.g., datapp_de)')
    parser.add_argument('--output-file', '-o', 
                       default='weather_rmse_results.csv',
                       help='Output CSV file name')
    parser.add_argument('--mlwp-models', nargs='+',
                       default=['pangu'],
                       help='MLWP model names to evaluate')
    parser.add_argument('--timedeltas', nargs='+', type=int,
                       default=[0, 1, 2, 3, 5, 8, 14],
                       help='Forecast lead times in days')
    parser.add_argument('--max-years', type=int, default=5,
                       help='Maximum number of years to process for efficiency')
    parser.add_argument('--variables', nargs='+',
                       default=['t', 'q', 'u', 'v', 'z'],
                       help='Weather variables to evaluate')
    parser.add_argument('--levels', nargs='+', type=int,
                       default=[1000, 850, 500, 200],
                       help='Pressure levels to evaluate')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("mlwp_weather_rmse")
    logger.info(f"Starting weather RMSE calculation for config: {args.config}")
    
    # Load configuration
    config = load_config(args.config)
    setup_name = config.get('setup_name', 'default_setup')
    
    # Set output path
    output_path = PROJECT_ROOT / args.output_file
    
    # Calculate weather RMSE
    try:
        calculate_mlwp_weather_rmse(
            config=config,
            output_file=output_path,
            mlwp_models=args.mlwp_models,
            timedeltas=args.timedeltas,
            max_years=args.max_years,
            variables=args.variables,
            pressure_levels=args.levels
        )
        logger.info(f"Weather RMSE calculation completed. Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error during weather RMSE calculation: {e}")
        raise


if __name__ == "__main__":
    main()