# src/fencast/models.py

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict

class DynamicFFNN(nn.Module):
    """
    A dynamically generated Feed-Forward Neural Network.
    
    The number and size of hidden layers are determined by the `hidden_layers` list.
    """
    def __init__(self, input_size: int, output_size: int, hidden_layers: List[int], dropout_rate: float, activation_fn: nn.Module):
        super().__init__()
        
        layers = []
        current_input_size = input_size
        
        # Create hidden layers dynamically
        for layer_size in hidden_layers:
            layers.append(nn.Linear(current_input_size, layer_size))
            layers.append(activation_fn)
            layers.append(nn.Dropout(dropout_rate))
            current_input_size = layer_size
            
        # Add the final output layer
        layers.append(nn.Linear(current_input_size, output_size))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        return self.network(x)


class DynamicCNN(nn.Module):
    """
    A dynamically generated Convolutional Neural Network with a two-stream input.

    This model processes spatial weather data through a CNN body and combines the
    result with non-spatial temporal data before feeding it to a dense regression head.
    """
    def __init__(self, config: Dict, params: Dict):
        super().__init__()
        self.config = config
        self.params = params

        # 1. Dynamically calculate input dimensions from the config
        # ----------------------------------------------------------------------
        input_channels = len(self.config['era5_var_names']) * len(self.config['feature_level'])
        
        lat_min, lat_max = self.config['feature_region']['lat_min'], self.config['feature_region']['lat_max']
        lon_min, lon_max = self.config['feature_region']['lon_min'], self.config['feature_region']['lon_max']
        resolution = 0.25 # ERA5 resolution
        input_height = int((lat_max - lat_min) / resolution) + 1
        input_width = int((lon_max - lon_min) / resolution) + 1
        
        activation_fn = getattr(nn, self.params['activation_name'])()
        
        # 2. Build the CNN Body for processing spatial data
        # ----------------------------------------------------------------------
        cnn_layers = []
        in_channels = input_channels
        out_channels_list = self.params['out_channels']
        kernel_size = self.params['kernel_size']
        
        for out_channels in out_channels_list:
            cnn_layers.append(nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size // 2) # Preserves dimensions with stride 1
            ))
            cnn_layers.append(activation_fn)
            # Add pooling for downsampling after each convolutional layer
            cnn_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
            
        self.cnn_body = nn.Sequential(*cnn_layers)

        # 3. Calculate the flattened feature size using a dummy forward pass
        # ----------------------------------------------------------------------
        with torch.no_grad():
            dummy_input = torch.randn(1, input_channels, input_height, input_width)
            dummy_output = self.cnn_body(dummy_input)
            self.flattened_size = int(np.prod(dummy_output.shape))

        # 4. Build the Regression Head for processing combined features
        # ----------------------------------------------------------------------
        dense_layers_list = self.params.get('dense_layers', [256, 128]) # Use tuned layers or a default
        dropout_rate = self.params['dropout_rate']
        output_size = self.config['target_size']
        num_temporal_features = 2 # day_of_year_sin, day_of_year_cos
        
        dense_layers = []
        # The first dense layer takes the flattened CNN output + temporal features
        current_input_size = self.flattened_size + num_temporal_features
        
        for layer_size in dense_layers_list:
            dense_layers.append(nn.Linear(current_input_size, layer_size))
            dense_layers.append(activation_fn)
            dense_layers.append(nn.Dropout(dropout_rate))
            current_input_size = layer_size
            
        dense_layers.append(nn.Linear(current_input_size, output_size))
        self.regression_head = nn.Sequential(*dense_layers)

    def forward(self, x_spatial: torch.Tensor, x_temporal: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the two-stream architecture.

        Args:
            x_spatial (torch.Tensor): The 4D tensor of weather data (batch, channels, height, width).
            x_temporal (torch.Tensor): The 2D tensor of temporal features (batch, num_features).

        Returns:
            torch.Tensor: The final model predictions.
        """
        # 1. Process spatial data through the CNN body
        x_spatial_features = self.cnn_body(x_spatial)
        
        # 2. Flatten the spatial features
        x_spatial_flat = torch.flatten(x_spatial_features, 1)
        
        # 3. Concatenate flattened spatial features with temporal features
        combined_features = torch.cat([x_spatial_flat, x_temporal], dim=1)
        
        # 4. Process the combined vector through the regression head
        predictions = self.regression_head(combined_features)
        
        return predictions