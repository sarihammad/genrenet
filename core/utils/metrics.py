"""
Metrics utilities for evaluation.
"""
import torch
import numpy as np
from typing import List, Tuple


def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate accuracy."""
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    return correct / targets.size(0)


def top_k_accuracy(outputs: torch.Tensor, targets: torch.Tensor, k: int = 3) -> float:
    """Calculate top-k accuracy."""
    _, top_k_preds = torch.topk(outputs, k, dim=1)
    correct = 0
    for i, target in enumerate(targets):
        if target.item() in top_k_preds[i]:
            correct += 1
    return correct / targets.size(0)


def per_class_accuracy(outputs: torch.Tensor, targets: torch.Tensor, num_classes: int) -> List[float]:
    """Calculate per-class accuracy."""
    _, predicted = torch.max(outputs, 1)
    per_class_correct = torch.zeros(num_classes)
    per_class_total = torch.zeros(num_classes)
    
    for i in range(targets.size(0)):
        label = targets[i]
        per_class_correct[label] += (predicted[i] == label).item()
        per_class_total[label] += 1
    
    per_class_acc = []
    for i in range(num_classes):
        if per_class_total[i] > 0:
            per_class_acc.append(per_class_correct[i] / per_class_total[i])
        else:
            per_class_acc.append(0.0)
    
    return per_class_acc
