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
        default_config = cfg.get("default_config", "global")
        parser.add_argument('config', nargs='?', 
                            default=default_config, 
                            help=f"Configuration file name (default: {default_config})")

    if 'timedeltas' in arguments:
        parser.add_argument('--time-deltas', nargs='+', type=int, default=list(range(1, 11)),
                            help='List of time deltas in days (default: 1-10)')
    
    if 'wpm_names' in arguments:
        parser.add_argument('--wpm', nargs='+', default=['pangu'],
                            help='List of weather prediction model names (default: ["pangu"])')

    if 'study_name' in arguments:
        parser.add_argument('--study-name', default='latest',
                            help='Study name to load results from (default: "latest")')
        
    ## Flags

    if 'override' in arguments:
        parser.add_argument('--override', '-o', action='store_true',
                            help='Flag to override existing files if they exist')
        
    return parser