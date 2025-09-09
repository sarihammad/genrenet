"""
GTZAN Dataset implementation with augmentation.
"""
import torch
from torch.utils.data import Dataset
import torchaudio.transforms as T
import librosa
import numpy as np
from typing import Dict, Any, Tuple
from .augment import pad_or_trim, random_time_stretch, random_pitch_shift, maybe_specaugment
from .splits import build_indices
from torchaudio.datasets import GTZAN


class GTZANDataset(Dataset):
    """GTZAN Dataset with log-mel spectrogram preprocessing."""
    
    def __init__(self, root: str, split: str, cfg: Dict[str, Any]):
        self.root = root
        self.split = split
        self.cfg = cfg
        
        # Initialize GTZAN dataset
        self.dataset = GTZAN(root=root, download=True)
        
        # Build split indices
        self.indices = build_indices(self.dataset, seed=cfg['seed'])
        self.split_indices = self.indices[split]
        
        # Audio transforms
        self.sample_rate = cfg['data']['sample_rate']
        self.duration_sec = cfg['data']['duration_sec']
        self.n_mels = cfg['data']['n_mels']
        self.n_fft = cfg['data']['n_fft']
        self.hop_length = cfg['data']['hop_length']
        
        # Mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        # Amplitude to DB transform
        self.to_db = T.AmplitudeToDB()
        
        # Resample transform
        self.resample = T.Resample(22050, self.sample_rate)
        
        # Genre labels
        self.genres = [
            'blues', 'classical', 'country', 'disco', 'hiphop',
            'jazz', 'metal', 'pop', 'reggae', 'rock'
        ]
        self.label_to_id = {genre: i for i, genre in enumerate(self.genres)}
        
    def __len__(self) -> int:
        return len(self.split_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Get actual dataset index
        dataset_idx = self.split_indices[idx]
        waveform, label = self.dataset[dataset_idx]
        
        # Convert to mono and resample
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        waveform = self.resample(waveform)
        
        # Convert to numpy for augmentation
        y = waveform.squeeze().numpy()
        
        # Apply waveform augmentation for training
        if self.split == 'train' and self.cfg['data']['augment']:
            y = random_time_stretch(y, p=0.3)
            y = random_pitch_shift(y, self.sample_rate, p=0.3)
        
        # Pad or trim to target duration
        target_len = int(self.duration_sec * self.sample_rate)
        y = pad_or_trim(y, target_len)
        
        # Convert back to tensor
        waveform = torch.from_numpy(y).unsqueeze(0)
        
        # Compute mel spectrogram
        mel_spec = self.mel_transform(waveform)
        log_mel_spec = self.to_db(mel_spec)
        
        # Per-example z-normalization
        log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-8)
        
        # Apply SpecAugment for training
        if self.split == 'train' and self.cfg['data']['specaugment']['p'] > 0:
            specaug_cfg = self.cfg['data']['specaugment']
            log_mel_spec = maybe_specaugment(
                log_mel_spec,
                freq_mask_param=specaug_cfg['freq_mask_param'],
                time_mask_param=specaug_cfg['time_mask_param'],
                p=specaug_cfg['p']
            )
        
        # Convert label to integer
        label_id = self.label_to_id[label]
        
        return log_mel_spec, label_id
