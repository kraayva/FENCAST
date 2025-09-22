# src/fencast/dataset.py

import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

from fencast.utils.paths import PROCESSED_DATA_DIR

class FencastDataset(Dataset):
    """
    PyTorch Dataset for the FENCAST project.

    This class loads the processed data, splits it into train/validation/test sets
    based on years defined in the config, and handles normalization of the features.
    """
    def __init__(self, config: dict, mode: str):
        """
        Args:
            config (dict): The configuration dictionary (e.g., from datapp_de.yaml).
            mode (str): One of 'train', 'validation', or 'test'.
        """
        super().__init__()
        if mode not in ['train', 'validation', 'test']:
            raise ValueError("Mode must be 'train', 'validation', or 'test'")
        self.config = config
        self.mode = mode
        self.scaler_path = PROCESSED_DATA_DIR / f"{self.config['setup_name']}_scaler.gz"

        # Load the pre-processed data
        self._load_data()

        # Split data according to the mode and config years
        self._split_data()

        # Setup and apply normalization
        self._normalize_features()

    def _load_data(self):
        """Loads features and labels from Parquet files."""
        setup_name = self.config['setup_name']
        features_path = PROCESSED_DATA_DIR / f"{setup_name}_features.parquet"
        labels_path = PROCESSED_DATA_DIR / f"{setup_name}_labels.parquet"

        if not features_path.exists() or not labels_path.exists():
            raise FileNotFoundError(
                f"Processed data not found for setup '{setup_name}'. "
                "Please run data_processing.py first."
            )
        
        print(f"[{self.mode}] Loading data from {features_path.parent}...")
        self.X = pd.read_parquet(features_path)
        self.y = pd.read_parquet(labels_path)

    def _split_data(self):
        """Filters the data based on the years specified in the config for the current mode."""
        if self.mode == 'train':
            # For training, we use all years NOT in validation or test sets
            validation_years = self.config['split_years']['validation']
            test_years = self.config['split_years']['test']
            exclude_years = set(validation_years + test_years)
            print(f"[{self.mode}] Excluding years: {sorted(list(exclude_years))}")
            
            # Select rows where the index's year is NOT in the exclude list
            self.X = self.X[~self.X.index.year.isin(exclude_years)]
            self.y = self.y[~self.y.index.year.isin(exclude_years)]
        else:
            # For validation or test, we use the specific years listed in the config
            split_years = self.config['split_years'][self.mode]
            print(f"[{self.mode}] Filtering data for years: {split_years}")
            
            # Select rows where the index's year IS IN the list for the current mode
            self.X = self.X[self.X.index.year.isin(split_years)]
            self.y = self.y[self.y.index.year.isin(split_years)]

        if len(self.X) == 0:
            raise ValueError(f"No data found for the years specified for mode '{self.mode}'.")

    def _normalize_features(self):
        """Fits a scaler on the training data and transforms all sets, or loads an existing scaler."""
        if self.mode == 'train':
            print(f"[{self.mode}] Fitting new scaler and saving to {self.scaler_path}...")
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X)
            joblib.dump(self.scaler, self.scaler_path)
        else: # Validation or test mode
            if not self.scaler_path.exists():
                raise FileNotFoundError(f"Scaler file not found at {self.scaler_path}. Please run training first.")
            print(f"[{self.mode}] Loading existing scaler from {self.scaler_path}...")
            self.scaler = joblib.load(self.scaler_path)
            self.X = self.scaler.transform(self.X)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        """
        Retrieves the feature and label tensors for a given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: (feature_tensor, label_tensor)
        """
        # Get the numpy arrays for the given index
        features = self.X[idx]
        labels = self.y.iloc[idx].values

        # Convert numpy arrays to PyTorch tensors
        feature_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(labels, dtype=torch.float32)
        
        return feature_tensor, label_tensor