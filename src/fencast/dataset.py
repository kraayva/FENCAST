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
    and handles normalization appropriate for the specified model type.
    For CNNs, it provides two separate inputs: spatial and temporal features.
    """
    def __init__(self, config: dict, mode: str, model_type: str, apply_normalization: bool = True):
        super().__init__()
        # ... (initial checks remain the same) ...
        if mode not in ['train', 'validation', 'test']:
            raise ValueError("Mode must be 'train', 'validation', or 'test'")
        if model_type not in ['ffnn', 'cnn']:
            raise ValueError("model_type must be 'ffnn' or 'cnn'")
        
        self.config = config
        self.mode = mode
        self.model_type = model_type
        self.setup_name = self.config['setup_name']
        
        self._load_data()
        self._split_data()

        # For CNNs, create temporal features after splitting
        if self.model_type == 'cnn':
            self._create_temporal_features()

        if apply_normalization:
            self._normalize_features()

    def _load_data(self):
        # ... (this method is unchanged from our last version) ...
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
                self.X = data['features'] # This is only the spatial data
            self.y = pd.read_parquet(labels_path)

    def _split_data(self):
        # ... (this method is unchanged from our last version) ...
        """
        Filters the data based on years. This logic is driven by the label's
        time index, which works for both NumPy arrays and DataFrames.
        """
        if self.mode == 'train':
            validation_years = self.config['split_years']['validation']
            test_years = self.config['split_years']['test']
            exclude_years = set(validation_years + test_years)
            print(f"[{self.mode}] Excluding years: {sorted(list(exclude_years))}")
            mask = ~self.y.index.year.isin(exclude_years)
        else:
            split_years = self.config['split_years'][self.mode]
            print(f"[{self.mode}] Filtering data for years: {split_years}")
            mask = self.y.index.year.isin(split_years)
        
        self.X = self.X[mask]
        self.y = self.y[mask]

        if len(self.X) == 0:
            raise ValueError(f"No data found for the years specified for mode '{self.mode}'.")

    def _create_temporal_features(self):
        """
        Generates cyclical day-of-year features for the CNN model.
        This is called after the data split to ensure indices match.
        """
        print(f"[{self.mode}] Creating temporal features for CNN...")
        day_of_year = self.y.index.dayofyear
        norm_denom = self.config.get('data_processing', {}).get('day_of_year_normalize_denominator', 365.0)
        day_of_year_rad = ((day_of_year - 1) / norm_denom) * 2 * np.pi
        
        self.temporal_features = pd.DataFrame({
            'day_of_year_sin': np.sin(day_of_year_rad),
            'day_of_year_cos': np.cos(day_of_year_rad)
        }, index=self.y.index)

    def _normalize_features(self):
        # ... (this method is unchanged from our last version) ...
        """Applies normalization based on the model type."""
        if self.model_type == 'ffnn':
            self._normalize_ffnn()
        elif self.model_type == 'cnn':
            self._normalize_cnn()

    def _normalize_ffnn(self):
        # ... (this method is unchanged from our last version) ...
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
        # ... (this method is unchanged from our last version) ...
        """Calculates/loads per-channel mean/std for 4D image-like data."""
        scaler_path = PROCESSED_DATA_DIR / f"{self.setup_name}_cnn_scaler.npz"
        
        if self.mode == 'train':
            print(f"[{self.mode}] Fitting new CNN scaler (per-channel mean/std)...")
            mean = np.mean(self.X, axis=(0, 2, 3), keepdims=True)
            std = np.std(self.X, axis=(0, 2, 3), keepdims=True)
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
        Returns a tuple of 2 tensors for FFNN and 3 for CNN.
        """
        if self.model_type == 'ffnn':
            features = self.X.iloc[idx].values
            labels = self.y.iloc[idx].values
            
            feature_tensor = torch.tensor(features, dtype=torch.float32)
            label_tensor = torch.tensor(labels, dtype=torch.float32)
            return feature_tensor, label_tensor
            
        elif self.model_type == 'cnn':
            spatial_features = self.X[idx]
            temporal_features = self.temporal_features.iloc[idx].values
            labels = self.y.iloc[idx].values
            
            spatial_tensor = torch.tensor(spatial_features, dtype=torch.float32)
            temporal_tensor = torch.tensor(temporal_features, dtype=torch.float32)
            label_tensor = torch.tensor(labels, dtype=torch.float32)
            return spatial_tensor, temporal_tensor, label_tensor