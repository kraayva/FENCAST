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
            input_size (int): The number of input features (e.g., 20295).
            output_size (int): The number of output values (e.g., 38 NUTS-2 regions).
        """
        super().__init__()
        
        self.network = nn.Sequential(
            # 1st Hidden Layer Block
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            # 2nd Hidden Layer Block
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            # Output Layer Block
            nn.Linear(512, output_size),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, output_size).
        """
        return self.network(x)


if __name__ == '__main__':
    # This block is for testing the model architecture.
    # You can run this file directly with `python src/fencast/models.py`
    
    # --- Configuration ---
    # Define the number of features based on our data processing
    INPUT_FEATURES = 20295 
    # Define the number of NUTS-2 regions to predict
    OUTPUT_FEATURES = 38
    # Define a sample batch size
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