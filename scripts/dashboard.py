from pathlib import Path
import argparse
import json
import configparser
import subprocess
import shutil
import sys
import re

from fencast.utils.tools import get_latest_study_dir
from fencast.utils.paths import load_config, PROJECT_ROOT
from fencast.utils.parser import get_parser

#!/usr/bin/env python3
"""
Create dashboard from config:
- parse --config / -c to get config file path
- extract setup name from config (JSON/YAML/TOML/INI/plain)
- find latest study under results/{setup_name} (sort names, pick last)
- launch optuna-dashboard pointing to results/{setup_name}/{latest_study}/{latest_study}.db
"""

def launch_optuna_dashboard(db_path: Path):
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")
    exe = shutil.which("optuna-dashboard")
    if not exe:
        raise SystemExit("optuna-dashboard not found in PATH")
    # build sqlite URL; for Windows ensure drive letter is included after third slash
    abs_path = db_path.resolve()
    sqlite_url = f"sqlite:///{abs_path.as_posix()}"
    # spawn dashboard
    subprocess.Popen([exe, sqlite_url])
    print(f"Launched optuna-dashboard for {db_path}")

def main():
    parser = get_parser(['config'], description="Open optuna-dashboard for latest study from config")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_name = config.get("setup_name")
    latest = get_latest_study_dir(PROJECT_ROOT / "results" / setup_name)
    # expected DB name: latest_study.db where latest_study is the latest.name
    db_name = f"{latest.name}.db"
    db_path = PROJECT_ROOT / "results" / setup_name / latest.name / db_name
    print(f"Using study directory: {db_path}")
    launch_optuna_dashboard(db_path)

if __name__ == "__main__":
    main()