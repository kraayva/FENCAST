# scripts/run_training.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Import our custom modules
from fencast.utils.paths import load_config
from fencast.dataset import FencastDataset
from fencast.models import SimpleFFNN

def run_training():
    """
    Main function to orchestrate the model training and validation process.
    """
    # 1. SETUP
    # =================================================================================
    print("--- 1. Setting up experiment ---")
    
    # Load the configuration for the specific run
    config = load_config("datapp_de")
    
    # Hyperparameters
    LEARNING_RATE = 0.001
    BATCH_SIZE = 64
    EPOCHS = 10 # Start with a small number to see if it works
    
    # Define input and output sizes from the config or data
    # This is a bit manual now, but could be automated later
    INPUT_SIZE = 20295
    OUTPUT_SIZE = 38
    
    # Set the device (use GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. DATA LOADING
    # =================================================================================
    print("\n--- 2. Loading data ---")
    
    # Create Dataset instances for training and validation
    # The scaler is fit on the training set and applied to the validation set
    train_dataset = FencastDataset(config=config, mode='train')
    validation_dataset = FencastDataset(config=config, mode='validation')
    
    # Create DataLoader instances
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True # Shuffle training data each epoch
    )
    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False # No need to shuffle validation data
    )

    # 3. MODEL, LOSS, and OPTIMIZER
    # =================================================================================
    print("\n--- 3. Initializing model, loss, and optimizer ---")

    model = SimpleFFNN(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE).to(device)
    
    # Loss Function: Mean Squared Error for regression
    criterion = nn.MSELoss()
    
    # Optimizer: Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. TRAINING LOOP
    # =================================================================================
    print("\n--- 4. Starting training ---")
    
    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train() # Set the model to training mode
        train_losses = []
        for batch_idx, (features, labels) in enumerate(train_loader):
            # Move data to the selected device
            features = features.to(device)
            labels = labels.to(device)
            
            # Forward pass: get model predictions
            outputs = model(features)
            
            # Calculate the loss
            loss = criterion(outputs, labels)
            train_losses.append(loss.item())
            
            # Backward pass and optimization
            optimizer.zero_grad() # Clear gradients from previous step
            loss.backward() # Compute gradients
            optimizer.step() # Update model weights

        avg_train_loss = np.mean(train_losses)

        # --- Validation Phase ---
        model.eval() # Set the model to evaluation mode (disables dropout)
        validation_losses = []
        with torch.no_grad(): # Disable gradient calculation for validation
            for features, labels in validation_loader:
                features = features.to(device)
                labels = labels.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                validation_losses.append(loss.item())

        avg_validation_loss = np.mean(validation_losses)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.6f}, Validation Loss: {avg_validation_loss:.6f}")

    print("\n--- Training finished ---")
    
    # (Optional) Save the trained model
    # torch.save(model.state_dict(), "fencast_model.pth")
    # print("Model saved to fencast_model.pth")


if __name__ == '__main__':
    run_training()