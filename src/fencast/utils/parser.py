import argparse

from fencast.utils.paths import PROJECT_ROOT, load_config


def get_parser(arguments: list, description: str = None) -> 'argparse.ArgumentParser':
    """Creates and returns an argument parser with specified arguments."""

    cfg = load_config()

    parser = argparse.ArgumentParser(
        description=description or "Argument Parser",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    ## Add arguments based on the provided list
    if 'config' in arguments:
        default_config = cfg.get('defaults').get('config')
        parser.add_argument('config', nargs='?',
                            default=default_config, 
                            help=f"Configuration file name")

    if 'timedeltas' in arguments:
        parser.add_argument('--time-deltas', nargs='+', type=int, default=list(range(1, 11)),
                            help='List of time deltas in days (default: 1-10)')
    
    if 'wpm_names' in arguments:
        parser.add_argument('--wpm', nargs='+', default=['pangu'],
                            help='List of weather prediction model names (default: ["pangu"])')

    if 'mlwp_models' in arguments:
        parser.add_argument('--mlwp-models', nargs='+',
                           help='Specific MLWP models to evaluate (default: all from config)')

    if 'mlwp_timedeltas' in arguments:
        parser.add_argument('--timedeltas', nargs='+', type=int,
                           help='Specific forecast lead times to evaluate (default: all from config)')

    if 'study_name' in arguments:
        default_study_name = cfg.get('defaults').get('study_name')
        parser.add_argument('--study-name', default=default_study_name,
                            help=f'Study name to load results from (default: "{default_study_name}")')
        
    ## Flags

    if 'override' in arguments:
        parser.add_argument('--override', '-o', action='store_true',
                            help='Flag to override existing files if they exist')

    if 'final_run' in arguments:
        parser.add_argument('--final-run', action='store_true',
                            help='Flag to train on all data (train + validation) to produce a final model.')
        
    if 'final_model' in arguments:
        parser.add_argument('--final-model', action='store_true',
                           help='Flag indicating to use the final model trained on all data')

    if 'force_save' in arguments:
        parser.add_argument('--force-save', '-f', action='store_true',
                            help='Save without prompting (overwrite existing files)')

    if 'feature_prefix' in arguments:
        parser.add_argument('--feature-prefix', '-p', default='era5_de',
                            help='Prefix for feature data files (default: era5_de)')

    if 'file_prefix' in arguments:
        parser.add_argument('--file-prefix', type=str, default='all', 
                            help='Prefix for feature data files.')

    if 'k_folds' in arguments:
        parser.add_argument('--k-folds', '-k', type=int, default=5,
                           help='Number of folds for cross validation')

    if 'results_dir' in arguments:
        parser.add_argument('--results-dir', '-r', 
                           help='Custom results directory (default: creates CV subdirectory in study dir)')

    if 'lead_times' in arguments:
        parser.add_argument('--lead-times', '-lt', nargs='+', type=int,
                           default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                           help='Lead times in days')

    if 'no_save' in arguments:
        parser.add_argument('--no-save', action='store_true',
                           help='Do not save results to file')

    if 'plot' in arguments:
        parser.add_argument('--plot', action='store_true',
                           help='Create and show visualization plot')

    if 'no_display' in arguments:
        parser.add_argument('--no-display', action='store_true',
                           help='Do not display plot (only save)')

    if 'weather_rmse_file' in arguments:
        parser.add_argument('--weather-rmse-file', 
                           help='Path to weather RMSE CSV file for comparison')

    if 'plot_type' in arguments:
        parser.add_argument('--plot-type', '-pt', choices=['weather', 'regions', 'seasons', 'rmse_mae'],
                            default=None,
                            help="Plot type: 'weather', 'regions', 'seasons', 'rmse_mae'. If omitted, creates regular plot.")

    if 'no_persistence' in arguments:
        parser.add_argument('--no-persistence', action='store_true',
                           help='Disable persistence baseline')

    if 'no_climatology' in arguments:
        parser.add_argument('--no-climatology', action='store_true',
                           help='Disable climatology baseline')

    if 'no_weather_total' in arguments:
        parser.add_argument('--no-weather-total', action='store_true',
                           help='Disable total weather RMSE')

    if 'model_name' in arguments:
        parser.add_argument('--model-name', default='best_model',
                           help='Model directory name to use (default: best_model)')

    if 'mlwp_name' in arguments:
        parser.add_argument('--mlwp-name', '-n', nargs='+', default=['pangu'],
                           help='MLWP model name(s) for data loading (default: pangu)')

    if 'persistence_lead_times' in arguments:
        parser.add_argument('--persistence-lead-times', nargs='+', type=int,
                              default=list(range(1, 11)),  
                           help='Custom lead times for persistence baseline (default: 1-10 days)')

    if 'figsize' in arguments:
        parser.add_argument('--figsize', nargs=2, type=float, default=[16, 8],
                           help='Figure size as width height')

    if 'output_file' in arguments:
        parser.add_argument('--output-file', '-o', 
                           default='get_from_config',
                           help='Output CSV file name')

    if 'mlwp_model' in arguments:
        parser.add_argument('--mlwp-model', '-m',
                           default='pangu',
                           help='MLWP model name to evaluate')

    if 'levels' in arguments:
        parser.add_argument('--levels', nargs='+',
                           default='all',
                           help='Pressure levels to evaluate')
        
    return parser