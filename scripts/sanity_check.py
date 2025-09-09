"""
Sanity check script to verify dataset and model setup.
"""
import torch
import yaml
from pathlib import Path

from core.data.dataset import GTZANDataset
from core.models.cnn import AudioCNN


def main():
    print("Running sanity checks...")
    
    # Load config
    config_path = Path('core/config.yaml')
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    print("✓ Config loaded")
    
    # Test dataset
    try:
        train_dataset = GTZANDataset(cfg['data']['root'], 'train', cfg)
        print(f"✓ Train dataset loaded: {len(train_dataset)} samples")
        
        # Test getting a sample
        spec, label = train_dataset[0]
        print(f"✓ Sample shape: {spec.shape}, label: {label}")
        
        # Test data loader
        from torch.utils.data import DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg['data']['batch_size_train'],
            shuffle=True,
            num_workers=0  # Use 0 for debugging
        )
        
        batch_spec, batch_labels = next(iter(train_loader))
        print(f"✓ Batch shape: {batch_spec.shape}, labels: {batch_labels.shape}")
        
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        return
    
    # Test model
    try:
        model = AudioCNN(
            n_classes=cfg['model']['num_classes'],
            dropout=cfg['model']['dropout']
        )
        print(f"✓ Model created: {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test forward pass
        with torch.no_grad():
            output = model(batch_spec)
            print(f"✓ Forward pass: input {batch_spec.shape} -> output {output.shape}")
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return
    
    print("\nAll sanity checks passed! ✓")


if __name__ == '__main__':
    main()
