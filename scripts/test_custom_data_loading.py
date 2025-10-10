#!/usr/bin/env python3
"""
Quick validation script to test data loading with custom years for cross validation.
"""

import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fencast.utils.paths import load_config
from fencast.dataset import FencastDataset
from fencast.training import ModelTrainer


def test_custom_data_loading():
    """Test custom data loading functionality for cross validation."""
    
    print("Testing custom data loading for cross validation...")
    
    # Load config
    config = load_config('datapp_de')
    
    # Define test years
    train_years = [1980, 1981, 1982, 1983, 1984]
    val_years = [1985, 1986]
    
    print(f"Train years: {train_years}")
    print(f"Val years: {val_years}")
    
    try:
        # Test creating datasets with custom years
        print("\nTesting FencastDataset with custom years...")
        train_dataset = FencastDataset(config=config, mode='train', model_type='cnn', 
                                     apply_normalization=False, custom_years=train_years)
        val_dataset = FencastDataset(config=config, mode='validation', model_type='cnn', 
                                   apply_normalization=False, custom_years=val_years)
        
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Val dataset size: {len(val_dataset)}")
        
        # Test ModelTrainer custom data loaders
        print("\nTesting ModelTrainer custom data loaders...")
        dummy_params = {
            'lr': 0.001,
            'activation_name': 'ELU',
            'dropout_rate': 0.1,
            'n_conv_layers': 3,
            'filters': 32,
            'kernel_size': 3
        }
        trainer = ModelTrainer(config, 'cnn', dummy_params)
        
        train_loader, val_loader = trainer.create_custom_data_loaders(train_years, val_years)
        
        print(f"Train loader batches: {len(train_loader)}")
        print(f"Val loader batches: {len(val_loader)}")
        
        # Test getting a sample batch
        print("\nTesting batch extraction...")
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        
        print(f"Train batch - Spatial: {train_batch[0].shape}, Temporal: {train_batch[1].shape}, Labels: {train_batch[2].shape}")
        print(f"Val batch - Spatial: {val_batch[0].shape}, Temporal: {val_batch[1].shape}, Labels: {val_batch[2].shape}")
        
        print("\n✅ All tests passed! Custom data loading is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_custom_data_loading()
    sys.exit(0 if success else 1)