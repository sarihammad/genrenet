"""
I/O utilities for saving and loading data.
"""
import json
import yaml
import csv
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import classification_report


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save data as JSON file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Dict[str, Any]:
    """Load data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_confmat_csv(confusion_matrix: np.ndarray, labels: List[str], filepath: str) -> None:
    """Save confusion matrix as CSV."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header row
        writer.writerow([''] + labels)
        # Data rows
        for i, label in enumerate(labels):
            writer.writerow([label] + confusion_matrix[i].tolist())


def save_class_report(y_true: List[int], y_pred: List[int], labels: List[str], filepath: str) -> None:
    """Save classification report as CSV."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'precision', 'recall', 'f1-score', 'support'])
        
        for label in labels:
            if label in report:
                writer.writerow([
                    label,
                    report[label]['precision'],
                    report[label]['recall'],
                    report[label]['f1-score'],
                    report[label]['support']
                ])
        
        # Add averages
        writer.writerow([
            'macro avg',
            report['macro avg']['precision'],
            report['macro avg']['recall'],
            report['macro avg']['f1-score'],
            report['macro avg']['support']
        ])
        writer.writerow([
            'weighted avg',
            report['weighted avg']['precision'],
            report['weighted avg']['recall'],
            report['weighted avg']['f1-score'],
            report['weighted avg']['support']
        ])
