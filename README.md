# FENCAST: Fine-tuning ENergy predictions from weather foreCASTs

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A machine learning project to predict solar power capacity factors (CF) for German NUTS-2 regions using ERA5 weather reanalysis data.

## About The Project

FENCAST investigates the capability of machine learning models to forecast energy generation values from meteorological data. This initial version implements a complete pipeline to:
1.  Process gridded, multi-level ERA5 weather data (temperature, wind, etc.).
2.  Train different machine learning architectures (simple FFNN, CNN) to predict solar capacity factors from the Copernicus Climate Change Service (C3S).
3.  Systematically tune the model's hyperparameters using Optuna.
4.  Evaluate the final model against a persistence baseline and analyze feature importance.

---
## Getting Started

Follow these steps to set up your local development environment.

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/your-username/FENCAST.git](https://github.com/your-username/FENCAST.git)
    cd FENCAST
    ```

2.  **Create and activate a virtual environment**
    ```bash
    # For Windows
    python -m venv .venv
    .\.venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install the project in editable mode** (Recommended for development)
    This ensures that changes you make to the source code are immediately available.
    ```bash
    pip install -e .
    ```

---
## Configuration

The project's behavior is controlled by YAML files in the `configs/` directory.

* `global.yaml`: Defines project-wide paths for data, logs, and results directories.
* `datapp_de.yaml`: Defines an experiment-specific setup. This includes paths to raw data files, the time range for analysis, the train/validation/test split years, and a unique name for the setup (`setup_name`).

---
## Workflow ⚙️

This is the standard end-to-end workflow for training and evaluating a model. Each step is performed by a dedicated script.

### 1. Process Raw Data

This script loads the raw NetCDF and CSV files specified in the config file (e.g. `configs/datapp_de.yaml`), processes them into a flat feature matrix and a label vector, and saves the clean data as Parquet files in the `data/processed/` directory.

To process raw data, run the following command and specify your configuration file using the `--config` option:

```bash
python src/fencast/data_processing.py --config datapp_de
```

`datapp_de` is the default option, and it will use the config file `config/datapp_de.yaml`. Replace it with the configuration you want to use to ensure the script uses the correct settings for data sources, time ranges, and region.
```

### 2. Find Optimal Hyperparameters

This script uses Optuna to run a hyperparameter tuning study. It will train dozens of models with different architectures and learning rates to find the combination that produces the lowest validation loss. The study progress is saved to a SQLite database (`.db`) in a timestamped folder inside `results/`.

```bash
python scripts/run_tuning.py
```
*Results, including visualization plots, are saved in the `results/{setup_name}/study_{date}/` directory.*

### 3. Train the Final Model

After tuning, this script loads the best hyperparameters found by the latest Optuna study. It then trains one final model on the combined training and validation data and saves a complete model checkpoint (`.pth`) in the `model/` directory.

```bash
python scripts/train_final_model.py
```

### 4. Evaluate the Model

This script loads your final trained model and evaluates its performance on the completely unseen **test set**. It calculates performance metrics (RMSE, MAE) and compares them against a persistence baseline.

```bash
python scripts/evaluate.py
```
*Result plots (time-series and scatter plots) are saved to the latest study directory in `results/`.*

### 5. Analyze Feature Importance

This script uses the final trained model to perform a permutation feature importance analysis. It helps to understand which weather variables (e.g., temperature vs. wind) and which pressure levels (e.g., surface vs. upper atmosphere) were most important for the model's predictions.

```bash
python scripts/feature_importance.py
```
*Importance bar charts are saved in the `results/` directory.*

---
## Project Structure

```
FENCAST/
├── configs/
│   ├── global.yaml
│   └── datapp_de.yaml
├── data/
│   ├── raw/
│   └── processed/
├── model/
│   └── *setup name*.pth
├── results/
│   └── *setup name*/
│       └── *study_date*/
│           ├── *study_date.db*
│           └── *.html, *.png
├── scripts/
│   ├── run_tuning.py
│   ├── train_final_model.py
│   ├── evaluate.py
│   └── feature_importance.py
└── src/
    └── fencast/
        ├── data_processing.py
        ├── dataset.py
        ├── models.py
        └── utils/
            ├── paths.py
            └── tools.py
```