from pathlib import Path

import yaml

def load_global():
    with open("configs/global.yaml", "r") as f:
        return yaml.safe_load(f)
    
cfg = load_global()
tmp_dir = Path(cfg["paths"]["data_tmp_dir"])

print(f"Tmp is {tmp_dir}")