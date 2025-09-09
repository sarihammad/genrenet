"""
Script to download GTZAN dataset.
"""
import argparse
from pathlib import Path
from torchaudio.datasets import GTZAN


def main():
    parser = argparse.ArgumentParser(description='Download GTZAN dataset')
    parser.add_argument('--out', type=str, default='data', help='Output directory')
    args = parser.parse_args()
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading GTZAN dataset...")
    print("Note: GTZAN dataset is for research purposes only.")
    print("Please ensure you comply with the dataset license terms.")
    
    # Initialize dataset to trigger download
    dataset = GTZAN(root=str(out_dir), download=True)
    
    print(f"Dataset downloaded to: {out_dir}")
    print(f"Total samples: {len(dataset)}")
    
    # Print genre distribution
    genres = {}
    for i in range(len(dataset)):
        _, label = dataset[i]
        genres[label] = genres.get(label, 0) + 1
    
    print("\nGenre distribution:")
    for genre, count in sorted(genres.items()):
        print(f"  {genre}: {count}")


if __name__ == '__main__':
    main()
