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
    def __init__(self, config: dict, mode: str, apply_normalization: bool = True):
        """
        Args:
            config (dict): The configuration dictionary (e.g., from datapp_de.yaml).
            mode (str): One of 'train', 'validation', or 'test'.
            apply_normalization (bool): If True, applies StandardScaler to features.
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

        if apply_normalization:
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
        # Get normalization exclusion patterns from config
        exclude_patterns = self.config.get('features', {}).get('normalization', {}).get('exclude_patterns', ['day_of_year'])
        
        # Identify features that should not be normalized
        keep_values_columns = []
        for pattern in exclude_patterns:
            keep_values_columns.extend([col for col in self.X.columns if pattern in col])
        normalize_columns = [col for col in self.X.columns if col not in keep_values_columns]
        
        if self.mode == 'train':
            print(f"[{self.mode}] Fitting new scaler on {len(normalize_columns)} features...")
            print(f"[{self.mode}] Excluding {len(keep_values_columns)} features from normalization: {keep_values_columns}")
            
            self.scaler = StandardScaler()
            # fit and transform columns
            weather_data_scaled = self.scaler.fit_transform(self.X[normalize_columns])
            
            # Combine scaled data with unscaled data
            self.X = pd.concat([
                pd.DataFrame(weather_data_scaled, columns=normalize_columns, index=self.X.index),
                self.X[keep_values_columns]
            ], axis=1)
            
            joblib.dump(self.scaler, self.scaler_path)
        else: # Validation or test mode
            if not self.scaler_path.exists():
                raise FileNotFoundError(f"Scaler file not found at {self.scaler_path}. Please run training first.")
            print(f"[{self.mode}] Loading existing scaler from {self.scaler_path}...")
            print(f"[{self.mode}] Excluding {len(keep_values_columns)} features from normalization: {keep_values_columns}")
            
            self.scaler = joblib.load(self.scaler_path)
            # Only transform weather columns
            weather_data_scaled = self.scaler.transform(self.X[normalize_columns])
            
            # Combine scaled weather data with unscaled temporal data
            self.X = pd.concat([
                pd.DataFrame(weather_data_scaled, columns=normalize_columns, index=self.X.index),
                self.X[keep_values_columns]
            ], axis=1)
            
        # Reorder columns to maintain consistent feature order
        self.X = self.X[normalize_columns + keep_values_columns]

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
        if isinstance(self.X, pd.DataFrame):
            features = self.X.iloc[idx].values
        else:
            features = self.X[idx]
        labels = self.y.iloc[idx].values

        # Convert numpy arrays to PyTorch tensors
        feature_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(labels, dtype=torch.float32)
        
        return feature_tensor, label_tensor
    
    def get_data(self):
        """Returns the processed but un-normalized X and y DataFrames."""
        return self.X, self.y