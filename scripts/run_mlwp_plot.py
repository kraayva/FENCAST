# /scripts/run_mlwp_plot.py
"""
Entry point script for creating MLWP evaluation plots.

This script generates comprehensive visualizations comparing energy prediction
performance against weather prediction quality, with configurable baseline
comparisons and flexible plotting options.
"""

import argparse
from pathlib import Path

from fencast.utils.tools import setup_logger
from fencast.visualization import create_mlwp_plot


def main():
    parser = argparse.ArgumentParser(
        description="Create MLWP evaluation plots with flexible options",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('config', nargs='?', default='datapp_de', help='Configuration file name (default: datapp_de)')
    parser.add_argument('model_type', nargs='?', default='cnn', help='Model architecture (default: cnn)')
    parser.add_argument('study_name', nargs='?', default='latest', help='Study name to load results from (default: "latest")')
    
    # Weather data options
    parser.add_argument('--weather-rmse-file', 
                       help='Path to weather RMSE CSV file for comparison')
    
    # Plot content options
    parser.add_argument('--no-persistence', action='store_true',
                       help='Disable persistence baseline')
    parser.add_argument('--no-climatology', action='store_true',
                       help='Disable climatology baseline')
    parser.add_argument('--no-weather-total', action='store_true',
                       help='Disable total weather RMSE')
    parser.add_argument('--show-weather-variables', action='store_true',
                       help='Show individual weather variable RMSE')
    
    # Baseline options
    parser.add_argument('--persistence-lead-times', nargs='+', type=int,
                       help='Custom lead times for persistence baseline')
    
    # Plot formatting
    parser.add_argument('--figsize', nargs=2, type=float, default=[16, 8],
                       help='Figure size as width height')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("mlwp_plot")
    logger.info(f"Creating MLWP plot for study: {args.study_name}")
    
    # Create the plot
    try:
        create_mlwp_plot(
            config_name=args.config,
            model_type=args.model_type,
            study_name=args.study_name,
            weather_rmse_file=args.weather_rmse_file,
            show_persistence=not args.no_persistence,
            show_climatology=not args.no_climatology,
            show_weather_total=not args.no_weather_total,
            show_weather_variables=args.show_weather_variables,
            persistence_lead_times=args.persistence_lead_times,
            figsize=tuple(args.figsize)
        )
        logger.info("MLWP plot creation completed successfully")
        
    except Exception as e:
        logger.error(f"Error during plot creation: {e}")
        raise


if __name__ == "__main__":
    main()