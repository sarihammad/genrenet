"""
Tests for model forward pass.
"""
import pytest
import torch

from core.models.cnn import AudioCNN


def test_model_forward():
    """Test model forward pass with random input."""
    model = AudioCNN(n_classes=10, dropout=0.3)
    
    # Create random input: [batch_size, channels, n_mels, time]
    batch_size = 4
    n_mels = 128
    time_frames = 200
    x = torch.randn(batch_size, 1, n_mels, time_frames)
    
    # Forward pass
    output = model(x)
    
    # Check output shape
    assert output.shape == (batch_size, 10)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_model_parameters():
    """Test model parameter count."""
    model = AudioCNN(n_classes=10, dropout=0.3)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Should have reasonable number of parameters
    assert total_params > 100000  # At least 100k parameters
    assert total_params < 10000000  # Less than 10M parameters
    assert total_params == trainable_params  # All parameters should be trainable


def test_model_different_input_sizes():
    """Test model with different input time dimensions."""
    model = AudioCNN(n_classes=10, dropout=0.3)
    
    # Test with different time dimensions
    time_dims = [100, 200, 300, 500]
    batch_size = 2
    n_mels = 128
    
    for time_dim in time_dims:
        x = torch.randn(batch_size, 1, n_mels, time_dim)
        output = model(x)
        
        # Output should always be [batch_size, n_classes]
        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()
