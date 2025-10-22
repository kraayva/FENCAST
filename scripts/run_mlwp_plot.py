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
from fencast.visualization import create_mlwp_plot, create_mlwp_seasonal_plot, create_mlwp_rmse_mae_plot


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
    parser.add_argument('--per-region', action='store_true',
                       help='Show one line per region instead of overall average')
    parser.add_argument('--seasonal', action='store_true',
                       help='Create seasonal analysis plot (Winter/Spring/Summer/Autumn)')
    parser.add_argument('--rmse-mae', action='store_true',
                       help='Create RMSE vs MAE comparison plot')
    parser.add_argument('--model-name', default='best_model',
                       help='Model directory name to use (default: best_model, can be "final_model" or custom name)')
    parser.add_argument('--mlwp-name', '-n', default='pangu',
                       help='MLWP model name for filename and data loading (default: pangu)')
    
    # Baseline options
    parser.add_argument('--persistence-lead-times', nargs='+', type=int,
                          default=list(range(1, 11)),  
                       help='Custom lead times for persistence baseline (default: 1-10 days)')
    
    # Plot formatting
    parser.add_argument('--figsize', nargs=2, type=float, default=[16, 8],
                       help='Figure size as width height')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("mlwp_plot")
    logger.info(f"Creating MLWP plot for study: {args.study_name}")
    
    # Automatically disable persistence for per-region plots to reduce clutter
    show_persistence = not args.no_persistence and not args.per_region
    
    if args.per_region and not args.no_persistence:
        logger.info("Automatically disabling persistence baseline for per-region view to reduce plot clutter")
    
    # Create the appropriate plot
    try:
        if args.seasonal:
            # Create seasonal plot
            create_mlwp_seasonal_plot(
                config_name=args.config,
                model_type=args.model_type,
                study_name=args.study_name,
                persistence_lead_times=args.persistence_lead_times,
                figsize=tuple(args.figsize),
                mlwp_name=args.mlwp_name,
                model_name=args.model_name
            )
            logger.info("MLWP seasonal plot creation completed successfully")
        elif args.rmse_mae:
            # Create RMSE vs MAE plot
            create_mlwp_rmse_mae_plot(
                config_name=args.config,
                model_type=args.model_type,
                study_name=args.study_name,
                figsize=tuple(args.figsize),
                mlwp_name=args.mlwp_name,
                model_name=args.model_name
            )
            logger.info("MLWP RMSE vs MAE plot creation completed successfully")
        else:
            # Create regular plot
            create_mlwp_plot(
                config_name=args.config,
                model_type=args.model_type,
                study_name=args.study_name,
                weather_rmse_file=args.weather_rmse_file,
                show_persistence=show_persistence,
                show_climatology=not args.no_climatology,
                show_weather_total=not args.no_weather_total,
                show_weather_variables=args.show_weather_variables,
                persistence_lead_times=args.persistence_lead_times,
                figsize=tuple(args.figsize),
                per_region=args.per_region,
                mlwp_name=args.mlwp_name,
                model_name=args.model_name
            )
            logger.info("MLWP plot creation completed successfully")
        
    except Exception as e:
        logger.error(f"Error during plot creation: {e}")
        raise


if __name__ == "__main__":
    main()