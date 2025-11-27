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
from fencast.utils.parser import get_parser


def main():
    parser = get_parser(['config', 'output_file', 'mlwp_model', 'mlwp_timedeltas', 'levels'],
                        description="Calculate weather prediction RMSE for MLWP models")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("mlwp_weather_rmse")
    logger.info(f"Starting weather RMSE calculation for config: {args.config}")
    
    # Load configuration
    config = load_config(args.config)
    setup_name = config.get('setup_name', 'default_setup')

    if args.levels == 'all':
        args.levels = config.get('feature_level')

    if args.output_file == 'get_from_config':
        args.output_file = f"weather_rmse_{config.get('setup_name')}.csv"

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