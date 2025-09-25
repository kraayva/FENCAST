# FENCAST
Fine-Tuning MLWP Models for Improved Forecasts in the Energy Sector

FENCAST is a Projet to investigate the capability of MLWP models to forecast weather values applicable to predict power generation values. This document explains how to set up the project, train and evaluate a ML algorithm to predict power generation values and use this algorithm to compare the output of several MLWPs.

Setup: First, clone the repository and create a virtual environment inside the project folder. After that, activate the environment and install the dependencies from the requirements.txt file.

Development mode: If you are working on FENCAST itself, install it in editable mode using "pip install -e .". This ensures that any changes you make to the code are reflected immediately in your environment without reinstalling the package.

Using FENCAST: To use FENCAST after setting up the environment, install it in editable mode as described above. 

## Workflow

1. **Data Processing**: Process raw data into training-ready format
   ```bash
   python scripts/run_data_processing.py --config datapp_de
   ```

2. **Hyperparameter Tuning**: Find optimal model parameters
   ```bash
   python scripts/run_tuning.py
   ```

3. **Final Training**: Train the model with best hyperparameters
   ```bash
   python scripts/run_training.py
   ```

4. **Evaluation**: Evaluate the trained model
   ```bash
   python scripts/evaluate.py
   ```

5. **Feature Importance**: Analyze feature contributions
   ```bash
   python scripts/feature_importance.py
   ```
