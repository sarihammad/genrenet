"""
Inference utilities for API.
"""
import torch
import torchaudio.transforms as T
import librosa
import numpy as np
from typing import List, Tuple
from .schemas import GenrePrediction
from core.data.augment import pad_or_trim


def preprocess_audio(audio_bytes: bytes, sample_rate: int = 22050, duration_sec: int = 30) -> torch.Tensor:
    """Preprocess audio bytes for inference."""
    # Load audio from bytes
    y, sr = librosa.load(audio_bytes, sr=sample_rate, mono=True)
    
    # Pad or trim to target duration
    target_len = int(duration_sec * sample_rate)
    y = pad_or_trim(y, target_len)
    
    # Convert to tensor
    waveform = torch.from_numpy(y).unsqueeze(0)
    
    # Compute mel spectrogram
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=128
    )
    
    to_db = T.AmplitudeToDB()
    
    mel_spec = mel_transform(waveform)
    log_mel_spec = to_db(mel_spec)
    
    # Per-example z-normalization
    log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-8)
    
    return log_mel_spec


def predict(model: torch.nn.Module, spec: torch.Tensor, label_map: dict, 
           device: torch.device, topk: int = 3) -> List[GenrePrediction]:
    """Run inference and return top-k predictions."""
    spec = spec.to(device)
    
    with torch.no_grad():
        output = model(spec)
        probabilities = torch.softmax(output, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, topk, dim=1)
        
        top_probs = top_probs.squeeze().cpu().numpy()
        top_indices = top_indices.squeeze().cpu().numpy()
    
    # Create predictions
    predictions = []
    for idx, prob in zip(top_indices, top_probs):
        genre = label_map[idx]
        predictions.append(GenrePrediction(label=genre, score=float(prob)))
    
    return predictions
