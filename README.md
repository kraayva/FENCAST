# FENCAST
Fine-Tuning MLWP Models for Improved Forecasts in the Energy Sector

FENCAST is a Projet to investigate the capability of MLWP models to forecast weather values applicable to predict power generation values. This document explains how to set up the project, train and evaluate a ML algorithm to predict power generation values and use this algorithm to compare the output of several MLWPs.

Setup: First, clone the repository and create a virtual environment inside the project folder. After that, activate the environment and install the dependencies from the requirements.txt file.

Development mode: If you are working on FENCAST itself, install it in editable mode using "pip install -e .". This ensures that any changes you make to the code are reflected immediately in your environment without reinstalling the package.

Using FENCAST: To use FENCAST after setting up the environment, install it in editable mode as described above. You can then run any module or script inside the package using the -m flag. For example, to start the data download script, use: "python -m fencast.data.download_cds_data".
