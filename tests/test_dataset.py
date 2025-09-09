"""
Tests for dataset functionality.
"""
import pytest
import torch
import yaml
from pathlib import Path

from core.data.dataset import GTZANDataset


@pytest.fixture
def config():
    """Load test config."""
    config_path = Path('core/config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@pytest.mark.slow
def test_dataset_loading(config):
    """Test dataset loading and basic functionality."""
    # Skip if data not available
    data_path = Path(config['data']['root'])
    if not data_path.exists():
        pytest.skip("GTZAN dataset not available")
    
    # Create dataset
    dataset = GTZANDataset(config['data']['root'], 'train', config)
    
    # Test basic properties
    assert len(dataset) > 0
    assert len(dataset.genres) == 10
    assert len(dataset.label_to_id) == 10
    
    # Test getting a sample
    spec, label = dataset[0]
    
    # Check tensor shapes and types
    assert isinstance(spec, torch.Tensor)
    assert isinstance(label, int)
    assert spec.shape[0] == 1  # Single channel
    assert spec.shape[1] == config['data']['n_mels']  # Mel bins
    assert label >= 0 and label < 10  # Valid label range
    
    # Test data loader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    
    batch_spec, batch_labels = next(iter(loader))
    assert batch_spec.shape[0] == 2  # Batch size
    assert batch_labels.shape[0] == 2
    assert batch_spec.shape[1:] == spec.shape[1:]  # Same feature shape


def test_dataset_splits(config):
    """Test that train/val/test splits are disjoint."""
    # Skip if data not available
    data_path = Path(config['data']['root'])
    if not data_path.exists():
        pytest.skip("GTZAN dataset not available")
    
    train_dataset = GTZANDataset(config['data']['root'], 'train', config)
    val_dataset = GTZANDataset(config['data']['root'], 'val', config)
    test_dataset = GTZANDataset(config['data']['root'], 'test', config)
    
    # Check that splits are disjoint
    train_indices = set(train_dataset.split_indices)
    val_indices = set(val_dataset.split_indices)
    test_indices = set(test_dataset.split_indices)
    
    assert len(train_indices & val_indices) == 0
    assert len(train_indices & test_indices) == 0
    assert len(val_indices & test_indices) == 0
