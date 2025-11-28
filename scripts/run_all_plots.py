# scripts/run_all_plots.py

"""Master script to run all plotting scripts sequentially."""

import argparse
import subprocess
import sys
import os
from fencast.utils.paths import load_config



def run_script(script_name: str, arguments: list = []):
    command = [sys.executable, script_name] + arguments
    print(f"\n--- Running: {script_name} {' '.join(arguments)} ---")
    try:
        # subprocess.run executes the command and waits for it to finish.
        # check=True raises a CalledProcessError if the exit code is non-zero (i.e., failure).
        result = subprocess.run(
            command,
            check=True,
            capture_output=True, # Captures stdout and stderr
            text=True            # Decodes output as text
        )
        
        print(f"✅ SUCCESS: {script_name} finished.")
        # Optional: Print the script's output
        # print("Script Output:\n" + result.stdout.strip())
        
        return result.stdout.strip()
        
    except FileNotFoundError:
        print(f"❌ ERROR: Could not find the script file: {script_name}")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        # Handles errors raised by the script itself (e.g., failed assertion, unhandled exception)
        print(f"❌ FATAL ERROR: {script_name} failed with exit code {e.returncode}")
        print(f"Stderr (Error Log):\n{e.stderr.strip()}")
        print("Stopping execution sequence.")
        # Terminate the master script
        sys.exit(1)


if __name__ == '__main__':
    # Run all plotting scripts sequentially

    cfg = load_config()

    default_config = cfg.get('defaults').get('config')

    mlwp_names = ['pangu', 'ifs']

    parser = argparse.ArgumentParser()
    parser.add_argument('--mlwp-names', nargs='+', default=mlwp_names)
    parser.add_argument('--config', type=str, default='')

    mlwp_names = parser.parse_args().mlwp_names

    script_name = 'scripts/run_mlwp_plot.py'

    for name in mlwp_names:
        for plot_type in ['regions', 'seasons', 'rmse_mae', 'weather']:
            run_script(script_name, ['--model-name', 'best_model', '-c', default_config, '--mlwp-name', name, '--plot-type', plot_type])

        run_script(script_name, ['--model-name', 'best_model', '-c', default_config, '--mlwp-name', name])
    run_script(script_name, ['--model-name', 'best_model', '-c', default_config])