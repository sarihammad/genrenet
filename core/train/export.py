"""
Export script for packaging trained model and metadata.
"""
import argparse
import yaml
import shutil
from pathlib import Path
import json


def main():
    parser = argparse.ArgumentParser(description='Export trained model package')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--out', type=str, required=True, help='Output directory')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Create output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Get checkpoint directory
    ckpt_dir = Path(args.ckpt).parent
    
    # Copy model checkpoint
    shutil.copy2(args.ckpt, out_dir / 'best_model.pt')
    
    # Copy label map
    shutil.copy2(ckpt_dir / 'label_map.json', out_dir / 'label_map.json')
    
    # Copy config snapshot
    shutil.copy2(ckpt_dir / 'config_snapshot.json', out_dir / 'config_snapshot.json')
    
    # Copy evaluation results if they exist
    eval_dir = ckpt_dir / 'eval_results'
    if eval_dir.exists():
        shutil.copytree(eval_dir, out_dir / 'eval_results', dirs_exist_ok=True)
    
    # Create package info
    package_info = {
        'model_name': 'gtzan_genre_cnn',
        'version': '0.1.0',
        'checkpoint': 'best_model.pt',
        'label_map': 'label_map.json',
        'config': 'config_snapshot.json',
        'num_classes': cfg['model']['num_classes'],
        'genres': [
            'blues', 'classical', 'country', 'disco', 'hiphop',
            'jazz', 'metal', 'pop', 'reggae', 'rock'
        ]
    }
    
    with open(out_dir / 'package_info.json', 'w') as f:
        json.dump(package_info, f, indent=2)
    
    print(f"Model package exported to: {out_dir}")
    print("Contents:")
    for item in out_dir.iterdir():
        print(f"  - {item.name}")


if __name__ == '__main__':
    main()
