"""
Waveform & SpecAugment utils.
"""
import random
import numpy as np
import librosa
import torch
import torchaudio.transforms as T


def pad_or_trim(y: np.ndarray, target_len: int) -> np.ndarray:
    """Pad or trim audio to target length."""
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y


def random_time_stretch(y: np.ndarray, low: float = 0.9, high: float = 1.15, p: float = 0.5) -> np.ndarray:
    """Apply random time stretching."""
    if random.random() < p:
        rate = random.uniform(low, high)
        return librosa.effects.time_stretch(y, rate=rate)
    return y


def random_pitch_shift(y: np.ndarray, sr: int, low: float = -2.0, high: float = 2.0, p: float = 0.5) -> np.ndarray:
    """Apply random pitch shifting."""
    if random.random() < p:
        n_steps = random.uniform(low, high)
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    return y


def maybe_specaugment(spec: torch.Tensor, freq_mask_param: int = 12, time_mask_param: int = 24, p: float = 0.3) -> torch.Tensor:
    """Apply SpecAugment with probability p."""
    if random.random() < p:
        spec = T.FrequencyMasking(freq_mask_param=freq_mask_param)(spec)
    if random.random() < p:
        spec = T.TimeMasking(time_mask_param=time_mask_param)(spec)
    return spec
