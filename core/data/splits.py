"""
Data splitting utilities for GTZAN dataset.
"""
import torch
from torchaudio.datasets import GTZAN
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict
import numpy as np


def build_indices(dataset: GTZAN, seed: int = 42) -> Dict[str, List[int]]:
    """Build stratified train/val/test indices per genre."""
    # Get all labels and file indices
    labels = []
    indices = []
    
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(label)
        indices.append(i)
    
    labels = np.array(labels)
    indices = np.array(indices)
    
    # Create stratified splits per genre
    train_indices = []
    val_indices = []
    test_indices = []
    
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        label_mask = labels == label
        label_indices = indices[label_mask]
        
        # 80/10/10 split
        train_idx, temp_idx = train_test_split(
            label_indices, test_size=0.2, random_state=seed, stratify=None
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=seed, stratify=None
        )
        
        train_indices.extend(train_idx)
        val_indices.extend(val_idx)
        test_indices.extend(test_idx)
    
    return {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }
