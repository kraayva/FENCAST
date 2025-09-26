# src/fencast/dataset.py

import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import numpy as np

from fencast.utils.paths import PROCESSED_DATA_DIR

class FencastDataset(Dataset):
    """
    PyTorch Dataset for the FENCAST project.

    This class loads processed data, splits it into train/validation/test sets,
    and handles normalization appropriate for the specified model type (FFNN or CNN).
    """
    def __init__(self, config: dict, mode: str, model_type: str, apply_normalization: bool = True):
        """
        Args:
            config (dict): The project's configuration dictionary.
            mode (str): One of 'train', 'validation', or 'test'.
            model_type (str): The target model architecture, 'ffnn' or 'cnn'.
            apply_normalization (bool): If True, applies normalization to features.
        """
        super().__init__()
        if mode not in ['train', 'validation', 'test']:
            raise ValueError("Mode must be 'train', 'validation', or 'test'")
        if model_type not in ['ffnn', 'cnn']:
            raise ValueError("model_type must be 'ffnn' or 'cnn'")
        
        self.config = config
        self.mode = mode
        self.model_type = model_type
        self.setup_name = self.config['setup_name']
        
        # Load the pre-processed data in the correct format
        self._load_data()

        # Split data according to the mode and config years
        self._split_data()

        if apply_normalization:
            self._normalize_features()

    def _load_data(self):
        """Loads features and labels based on the model_type."""
        print(f"[{self.mode}] Loading data for '{self.model_type}' model...")
        
        if self.model_type == 'ffnn':
            features_path = PROCESSED_DATA_DIR / f"{self.setup_name}_features_ffnn.parquet"
            labels_path = PROCESSED_DATA_DIR / f"{self.setup_name}_labels_ffnn.parquet"
            if not features_path.exists() or not labels_path.exists():
                raise FileNotFoundError(f"FFNN data not found for setup '{self.setup_name}'. Run data processing with --model-target ffnn.")
            self.X = pd.read_parquet(features_path)
            self.y = pd.read_parquet(labels_path)
        
        elif self.model_type == 'cnn':
            features_path = PROCESSED_DATA_DIR / f"{self.setup_name}_features_cnn.npz"
            labels_path = PROCESSED_DATA_DIR / f"{self.setup_name}_labels_cnn.parquet"
            if not features_path.exists() or not labels_path.exists():
                raise FileNotFoundError(f"CNN data not found for setup '{self.setup_name}'. Run data processing with --model-target cnn.")
            
            with np.load(features_path) as data:
                self.X = data['features']
            self.y = pd.read_parquet(labels_path)

    def _split_data(self):
        """
        Filters the data based on years. This logic is driven by the label's
        time index, which works for both NumPy arrays and DataFrames.
        """
        if self.mode == 'train':
            validation_years = self.config['split_years']['validation']
            test_years = self.config['split_years']['test']
            exclude_years = set(validation_years + test_years)
            print(f"[{self.mode}] Excluding years: {sorted(list(exclude_years))}")
            
            # Create a boolean mask from the label's index
            mask = ~self.y.index.year.isin(exclude_years)
        else:
            split_years = self.config['split_years'][self.mode]
            print(f"[{self.mode}] Filtering data for years: {split_years}")
            
            # Create a boolean mask from the label's index
            mask = self.y.index.year.isin(split_years)
        
        # Apply the mask to both features and labels
        self.X = self.X[mask]
        self.y = self.y[mask]

        if len(self.X) == 0:
            raise ValueError(f"No data found for the years specified for mode '{self.mode}'.")

    def _normalize_features(self):
        """Applies normalization based on the model type."""
        if self.model_type == 'ffnn':
            self._normalize_ffnn()
        elif self.model_type == 'cnn':
            self._normalize_cnn()

    def _normalize_ffnn(self):
        """Fits/loads a StandardScaler for 2D tabular data."""
        scaler_path = PROCESSED_DATA_DIR / f"{self.setup_name}_ffnn_scaler.gz"
        exclude_patterns = self.config.get('features', {}).get('normalization', {}).get('exclude_patterns', [])
        
        keep_values_columns = [col for col in self.X.columns for pattern in exclude_patterns if pattern in col]
        normalize_columns = [col for col in self.X.columns if col not in keep_values_columns]
        
        if self.mode == 'train':
            print(f"[{self.mode}] Fitting new FFNN scaler...")
            scaler = StandardScaler()
            self.X[normalize_columns] = scaler.fit_transform(self.X[normalize_columns])
            joblib.dump(scaler, scaler_path)
        else:
            if not scaler_path.exists():
                raise FileNotFoundError(f"Scaler not found at {scaler_path}. Run training first.")
            print(f"[{self.mode}] Loading existing FFNN scaler from {scaler_path}...")
            scaler = joblib.load(scaler_path)
            self.X[normalize_columns] = scaler.transform(self.X[normalize_columns])

    def _normalize_cnn(self):
        """Calculates/loads per-channel mean/std for 4D image-like data."""
        scaler_path = PROCESSED_DATA_DIR / f"{self.setup_name}_cnn_scaler.npz"
        
        if self.mode == 'train':
            print(f"[{self.mode}] Fitting new CNN scaler (per-channel mean/std)...")
            # Calculate mean and std per channel across all samples, height, and width
            # self.X shape: (samples, channels, height, width)
            mean = np.mean(self.X, axis=(0, 2, 3), keepdims=True)
            std = np.std(self.X, axis=(0, 2, 3), keepdims=True)
            # Add a small epsilon to std to prevent division by zero
            std[std == 0] = 1e-7
            
            self.X = (self.X - mean) / std
            np.savez(scaler_path, mean=mean, std=std)
        else:
            if not scaler_path.exists():
                raise FileNotFoundError(f"Scaler not found at {scaler_path}. Run training first.")
            print(f"[{self.mode}] Loading existing CNN scaler from {scaler_path}...")
            with np.load(scaler_path) as data:
                mean = data['mean']
                std = data['std']
            self.X = (self.X - mean) / std

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple:
        """
        Retrieves the feature and label tensors for a given index.
        """
        # This logic gracefully handles both pd.DataFrame and np.ndarray for self.X
        if isinstance(self.X, pd.DataFrame):
            features = self.X.iloc[idx].values
        else:
            features = self.X[idx]
        
        labels = self.y.iloc[idx].values

        # Convert numpy arrays to PyTorch tensors
        feature_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(labels, dtype=torch.float32)
        
        return feature_tensor, label_tensor