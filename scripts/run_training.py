# scripts/run_training.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Import our custom modules
from fencast.utils.paths import load_config
from fencast.dataset import FencastDataset
from fencast.models import SimpleFFNN
from fencast.utils.paths import PROJECT_ROOT

def run_training(LEARNING_RATE: float = 0.0001, 
                 BATCH_SIZE: int = 64, 
                 EPOCHS: int = 10, 
                 INPUT_SIZE: int = 20295, 
                 OUTPUT_SIZE: int = 38):
    """
    Main function to orchestrate the model training and validation process.
    """
    # 1. SETUP
    # =================================================================================
    print("--- 1. Setting up experiment ---")
    
    # Load the configuration for the specific run
    config = load_config("datapp_de")
            
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

    best_validation_loss = float('inf') # Initialize with a very high value
    model_save_path = PROJECT_ROOT / f"model/{config['setup_name']}_best_model.pth"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)  # Create model directory if it doesn't exist

    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        train_losses = []
        for batch_idx, (features, labels) in enumerate(train_loader):
            # ... (training pass logic is the same) ...
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        avg_train_loss = np.mean(train_losses)

        # --- Validation Phase ---
        model.eval()
        validation_losses = []
        with torch.no_grad():
            for features, labels in validation_loader:
                # ... (validation pass logic is the same) ...
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                validation_losses.append(loss.item())

        avg_validation_loss = np.mean(validation_losses)

        # --- Save the best model checkpoint ---
        if avg_validation_loss < best_validation_loss:
            best_validation_loss = avg_validation_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.6f}, Validation Loss: {avg_validation_loss:.6f} (saved)")
        else:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.6f}, Validation Loss: {avg_validation_loss:.6f}")

    print("\n--- Training finished ---")
    print(f"Best model validation loss: {best_validation_loss:.6f}")
    print(f"Saved to: {model_save_path}")


if __name__ == '__main__':
    run_training(INPUT_SIZE=20295, OUTPUT_SIZE=37)