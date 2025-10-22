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

    # Unified plot type selector (replaces several boolean flags)
    parser.add_argument('--plot-type', '-p', choices=['weather', 'regions', 'seasons', 'rmse_mae'],
                        default=None,
                        help="Plot type: 'weather' (show weather variables), 'regions' (per-region view), 'seasons' (seasonal plot), 'rmse_mae' (RMSE vs MAE). If omitted, creates regular plot.")

    # Backwards-compatible individual options (still accepted but plot-type takes precedence)
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
    
    # Auto-load weather RMSE file if showing weather variables but no file specified
    weather_file = args.weather_rmse_file
    if args.show_weather_variables and not weather_file:
        weather_file = f"results/weather_rmse_{args.mlwp_name}.csv"
        logger.info(f"Auto-loading weather RMSE file: {weather_file}")
    
    # Create the appropriate plot based on plot-type (or fall back to legacy flags)
    try:
        plot_type = args.plot_type

        # Backwards compatibility: if no explicit plot_type provided, infer from legacy flags
        if plot_type is None:
            if args.seasonal:
                plot_type = 'seasons'
            elif args.rmse_mae:
                plot_type = 'rmse_mae'
            elif args.per_region:
                plot_type = 'regions'
            elif args.show_weather_variables:
                plot_type = 'weather'

        if plot_type == 'seasons':
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

        elif plot_type == 'rmse_mae':
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
            # Regular or weather or per-region plot
            show_weather_variables = (plot_type == 'weather') or args.show_weather_variables
            per_region = (plot_type == 'regions') or args.per_region

            # Auto-load weather RMSE file when user requested weather plot and no file was provided
            if (plot_type == 'weather' or show_weather_variables) and not weather_file:
                weather_file = f"results/weather_rmse_{args.mlwp_name}.csv"
                logger.info(f"Auto-loading weather RMSE file for plot-type 'weather': {weather_file}")

            create_mlwp_plot(
                config_name=args.config,
                model_type=args.model_type,
                study_name=args.study_name,
                weather_rmse_file=weather_file,
                show_persistence=show_persistence,
                show_climatology=not args.no_climatology,
                show_weather_total=not args.no_weather_total,
                show_weather_variables=show_weather_variables,
                persistence_lead_times=args.persistence_lead_times,
                figsize=tuple(args.figsize),
                per_region=per_region,
                mlwp_name=args.mlwp_name,
                model_name=args.model_name
            )
            logger.info("MLWP plot creation completed successfully")

    except Exception as e:
        logger.error(f"Error during plot creation: {e}")
        raise


if __name__ == "__main__":
    main()