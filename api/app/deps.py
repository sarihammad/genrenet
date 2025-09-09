"""
Dependency injection for FastAPI app.
"""
import torch
import json
import yaml
from pathlib import Path
from typing import Dict, Any
from functools import lru_cache

from core.models.cnn import AudioCNN


@lru_cache()
def get_device() -> torch.device:
    """Get device for inference."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@lru_cache()
def get_model() -> AudioCNN:
    """Load trained model."""
    device = get_device()
    
    # Load config
    config_path = Path('saved_models/config_snapshot.json')
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    
    # Create model
    model = AudioCNN(
        n_classes=cfg['model']['num_classes'],
        dropout=cfg['model']['dropout']
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = Path('saved_models/best_model.pt')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    return model


@lru_cache()
def get_label_map() -> Dict[int, str]:
    """Load label map."""
    label_map_path = Path('saved_models/label_map.json')
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    
    return {int(k): v for k, v in label_map.items()}


@lru_cache()
def get_config() -> Dict[str, Any]:
    """Load config."""
    config_path = Path('saved_models/config_snapshot.json')
    with open(config_path, 'r') as f:
        return json.load(f)
