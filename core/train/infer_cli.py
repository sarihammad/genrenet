"""
CLI inference script for GTZAN genre classification.
"""
import argparse
import torch
import torchaudio.transforms as T
import librosa
import numpy as np
import json
from pathlib import Path

from ..models.cnn import AudioCNN
from ..data.augment import pad_or_trim


def load_audio(filepath: str, sample_rate: int = 22050, duration_sec: int = 30) -> torch.Tensor:
    """Load and preprocess audio file."""
    # Load audio
    y, sr = librosa.load(filepath, sr=sample_rate, mono=True)
    
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


def main():
    parser = argparse.ArgumentParser(description='Infer genre from audio file')
    parser.add_argument('--wav', type=str, required=True, help='Path to audio file')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--topk', type=int, default=3, help='Number of top predictions')
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load label map
    ckpt_dir = Path(args.ckpt).parent
    with open(ckpt_dir / 'label_map.json', 'r') as f:
        label_map = json.load(f)
    
    # Convert label map to list for indexing
    id_to_label = {int(k): v for k, v in label_map.items()}
    
    # Create model
    model = AudioCNN(n_classes=len(label_map), dropout=0.3).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Load and preprocess audio
    print(f"Loading audio: {args.wav}")
    spec = load_audio(args.wav).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(spec)
        probabilities = torch.softmax(output, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, args.topk, dim=1)
        
        top_probs = top_probs.squeeze().cpu().numpy()
        top_indices = top_indices.squeeze().cpu().numpy()
    
    # Print results
    print(f"\nTop-{args.topk} Genre Predictions:")
    for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
        genre = id_to_label[idx]
        print(f"{i+1}. {genre}: {prob:.4f}")


if __name__ == '__main__':
    main()
