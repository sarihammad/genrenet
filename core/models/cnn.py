"""
CNN model for audio genre classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioCNN(nn.Module):
    """CNN model for audio genre classification using log-mel spectrograms."""
    
    def __init__(self, n_classes: int = 10, dropout: float = 0.3):
        super(AudioCNN, self).__init__()
        
        # Convolutional blocks
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Adaptive pooling to handle variable time dimensions
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Classifier
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, n_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Input: [B, 1, n_mels, T]
        
        # Conv block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        # Conv block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # Conv block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Conv block 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        # Adaptive pooling: [B, 256, 1, 1]
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten: [B, 256]
        
        # Classifier
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)  # Logits: [B, n_classes]
        
        return x
