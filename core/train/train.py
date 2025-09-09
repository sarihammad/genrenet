"""
Training script for GTZAN genre classification.
"""
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import os
from pathlib import Path
from tqdm import tqdm
import json

from ..utils.seed import set_seed
from ..utils.metrics import accuracy
from ..utils.io import save_json
from ..data.dataset import GTZANDataset
from ..models.cnn import AudioCNN


def resolve_device(device: str) -> torch.device:
    """Resolve device from config."""
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


def create_optimizer(model: nn.Module, cfg: dict) -> optim.Optimizer:
    """Create optimizer based on config."""
    if cfg['train']['optimizer'] == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=cfg['train']['lr'],
            weight_decay=cfg['train']['weight_decay']
        )
    elif cfg['train']['optimizer'] == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=cfg['train']['lr'],
            weight_decay=cfg['train']['weight_decay'],
            momentum=cfg['train']['momentum']
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg['train']['optimizer']}")


def create_scheduler(optimizer: optim.Optimizer, cfg: dict, num_epochs: int) -> optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler."""
    if cfg['train']['scheduler'] == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif cfg['train']['scheduler'] == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs//3, gamma=0.1)
    else:
        return optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer,
                criterion: nn.Module, device: torch.device, scaler: GradScaler = None) -> tuple:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with autocast():
                output = model(data)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        acc = accuracy(output, target)
        total_loss += loss.item()
        total_acc += acc
        num_batches += 1
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{acc:.4f}'
        })
    
    return total_loss / num_batches, total_acc / num_batches


def validate_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
                  device: torch.device) -> tuple:
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            acc = accuracy(output, target)
            
            total_loss += loss.item()
            total_acc += acc
            num_batches += 1
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{acc:.4f}'
            })
    
    return total_loss / num_batches, total_acc / num_batches


def main():
    parser = argparse.ArgumentParser(description='Train GTZAN genre classifier')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Set seed
    set_seed(cfg['seed'])
    
    # Setup device
    device = resolve_device(cfg['device'])
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(cfg['train']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create datasets
    train_dataset = GTZANDataset(cfg['data']['root'], 'train', cfg)
    val_dataset = GTZANDataset(cfg['data']['root'], 'val', cfg)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['data']['batch_size_train'],
        shuffle=True,
        num_workers=cfg['data']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
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
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, cfg)
    scheduler = create_scheduler(optimizer, cfg, cfg['train']['epochs'])
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # AMP scaler
    scaler = GradScaler() if cfg['train']['amp'] and device.type == 'cuda' else None
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(cfg['train']['epochs']):
        print(f"\nEpoch {epoch+1}/{cfg['train']['epochs']}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save model state dict
            torch.save(model.state_dict(), save_dir / cfg['train']['save_name'])
            
            # Save label map
            label_map = {i: genre for i, genre in enumerate(train_dataset.genres)}
            save_json(label_map, save_dir / 'label_map.json')
            
            # Save config snapshot
            save_json(cfg, save_dir / 'config_snapshot.json')
            
            print(f"New best model saved! Val Acc: {val_acc:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= cfg['train']['early_stop_patience']:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")


if __name__ == '__main__':
    main()
