"""
Evaluation script for GTZAN genre classification.
"""
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import json

from ..utils.seed import set_seed
from ..utils.metrics import accuracy, top_k_accuracy, per_class_accuracy
from ..utils.io import save_json, save_confmat_csv, save_class_report
from ..utils.plots import plot_confusion_matrix
from ..data.dataset import GTZANDataset
from ..models.cnn import AudioCNN


def main():
    parser = argparse.ArgumentParser(description='Evaluate GTZAN genre classifier')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Set seed
    set_seed(cfg['seed'])
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test dataset
    test_dataset = GTZANDataset(cfg['data']['root'], 'test', cfg)
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg['data']['batch_size_eval'],
        shuffle=False,
        num_workers=cfg['data']['num_workers'],
        pin_memory=True
    )
    
    # Create model
    model = AudioCNN(
        n_classes=cfg['model']['num_classes'],
        dropout=cfg['model']['dropout']
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Load label map
    ckpt_dir = Path(args.ckpt).parent
    with open(ckpt_dir / 'label_map.json', 'r') as f:
        label_map = json.load(f)
    
    # Convert label map to list for indexing
    id_to_label = {int(k): v for k, v in label_map.items()}
    labels = [id_to_label[i] for i in range(len(id_to_label))]
    
    # Evaluation
    all_predictions = []
    all_targets = []
    all_outputs = []
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    
    criterion = nn.CrossEntropyLoss()
    
    print("Evaluating on test set...")
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            # Get predictions
            _, predicted = torch.max(output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_outputs.extend(output.cpu().numpy())
            
            acc = accuracy(output, target)
            total_loss += loss.item()
            total_acc += acc
            num_batches += 1
    
    # Calculate metrics
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    
    # Top-k accuracy
    all_outputs = torch.tensor(all_outputs)
    all_targets = torch.tensor(all_targets)
    top3_acc = top_k_accuracy(all_outputs, all_targets, k=3)
    top5_acc = top_k_accuracy(all_outputs, all_targets, k=5)
    
    # Per-class accuracy
    per_class_acc = per_class_accuracy(all_outputs, all_targets, cfg['model']['num_classes'])
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    # Classification report
    class_report = classification_report(
        all_targets, all_predictions, 
        target_names=labels, 
        output_dict=True
    )
    
    # Print results
    print(f"\nTest Results:")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {avg_acc:.4f}")
    print(f"Top-3 Accuracy: {top3_acc:.4f}")
    print(f"Top-5 Accuracy: {top5_acc:.4f}")
    
    print(f"\nPer-class Accuracy:")
    for i, (label, acc) in enumerate(zip(labels, per_class_acc)):
        print(f"{label}: {acc:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(all_targets, all_predictions, target_names=labels))
    
    # Save results
    results_dir = ckpt_dir / 'eval_results'
    results_dir.mkdir(exist_ok=True)
    
    # Save confusion matrix
    save_confmat_csv(cm, labels, results_dir / 'confusion_matrix.csv')
    plot_confusion_matrix(cm, labels, results_dir / 'confusion_matrix.png')
    
    # Save classification report
    save_class_report(all_targets, all_predictions, labels, results_dir / 'classification_report.csv')
    
    # Save summary metrics
    summary = {
        'test_loss': avg_loss,
        'test_accuracy': avg_acc,
        'top3_accuracy': top3_acc,
        'top5_accuracy': top5_acc,
        'per_class_accuracy': {label: acc for label, acc in zip(labels, per_class_acc)},
        'classification_report': class_report
    }
    save_json(summary, results_dir / 'summary.json')
    
    print(f"\nResults saved to: {results_dir}")


if __name__ == '__main__':
    main()
