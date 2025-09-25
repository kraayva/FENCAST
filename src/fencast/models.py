# src/fencast/models.py

import torch
import torch.nn as nn

class SimpleFFNN(nn.Module):
    """
    A simple feed-forward neural network for the FENCAST project.

    The architecture consists of two hidden layers with ReLU activations,
    Dropout for regularization, and a final Sigmoid activation to ensure
    the output is between 0 and 1.
    """
    def __init__(self, input_size: int, output_size: int):
        """
        Initializes the layers of the network.

        Args:
            input_size (int): The number of input features (e.g., 20297 = 20295 weather + 2 temporal).
            output_size (int): The number of output values (e.g., 38 NUTS-2 regions).
        """
        super().__init__()
        
        self.network = nn.Sequential(
            # 1st Hidden Layer Block
            nn.Linear(input_size, 1024),  # Could be configurable
            nn.ReLU(),
            nn.Dropout(p=0.3),  # Could be configurable

            # 2nd Hidden Layer Block
            nn.Linear(1024, 512),  # Could be configurable
            nn.ReLU(),
            nn.Dropout(p=0.3),  # Could be configurable

            # Output Layer Block
            nn.Linear(512, output_size),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
# In src/fencast/models.py, add this new class

class DynamicFFNN(nn.Module):
    """
    A feed-forward neural network with a dynamically configurable architecture.
    """
    def __init__(self, input_size: int, output_size: int, hidden_layers: list, 
                 dropout_rate: float, activation_fn: nn.Module = nn.ReLU()):
        """
        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output values.
            hidden_layers (list): A list of integers, where each integer is the number of neurons in a hidden layer.
            dropout_rate (float): The dropout probability.
            activation_fn (nn.Module): The activation function to use (e.g., nn.ReLU() or nn.ELU()).
        """
        super().__init__()
        
        layers = []
        in_features = input_size
        
        # Create hidden layers dynamically
        for h_features in hidden_layers:
            layers.append(nn.Linear(in_features, h_features))
            layers.append(activation_fn)
            layers.append(nn.Dropout(dropout_rate))
            in_features = h_features # The output of this layer is the input to the next
            
        # Add the final output layer
        layers.append(nn.Linear(in_features, output_size))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


if __name__ == '__main__':
    # This block is for testing the model architecture.
    # --- Test Configuration (hardcoded for model testing only) ---
    INPUT_FEATURES = 20297 
    # Define the number of NUTS-2 regions to predict
    OUTPUT_FEATURES = 38
    # Define a sample batch size for testing
    BATCH_SIZE = 64

    # --- Model Initialization ---
    print("Initializing model...")
    model = SimpleFFNN(input_size=INPUT_FEATURES, output_size=OUTPUT_FEATURES)
    print("Model Architecture:")
    print(model)

    # --- Test Forward Pass ---
    print("\nTesting a forward pass with a dummy tensor...")
    # Create a random tensor with the shape of a batch of input data
    dummy_input = torch.randn(BATCH_SIZE, INPUT_FEATURES)
    print(f"Shape of dummy input: {dummy_input.shape}")
    
    # Pass the dummy data through the model
    output = model(dummy_input)
    print(f"Shape of model output: {output.shape}")

    # --- Verification ---
    # Check if the output shape is correct
    assert output.shape == (BATCH_SIZE, OUTPUT_FEATURES)
    # Check if output values are between 0 and 1 (due to Sigmoid)
    assert torch.all(output >= 0) and torch.all(output <= 1)
    
    print("\nModel test passed successfully! ✅")