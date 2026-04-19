"""Feature pre-extraction and caching script for frozen DINOv2 backbone."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import CUB200Dataset
from models import DINOv2Extractor


def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    L2-normalize feature vectors.
    
    Args:
        features (np.ndarray): Feature array, shape (n, d)
        
    Returns:
        np.ndarray: L2-normalized features, shape (n, d)
    """
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)  # Avoid division by zero
    return features / norms


def extract_features(
    data_root: str,
    split: str,
    model_variant: str = 'vitb14',
    batch_size: int = 128,
    output_dir: str = './cache',
    device: str = None,
) -> None:
    """
    Extract and cache features from frozen DINOv2 backbone.
    
    Args:
        data_root (str): Root directory of CUB-200 dataset.
        split (str): One of 'train', 'val', 'test'.
        model_variant (str): DINOv2 variant ('vits14', 'vitb14', 'vitl14').
        batch_size (int): Batch size for DataLoader.
        output_dir (str): Directory to save .npz cache files.
        device (str): Device to use ('cuda', 'cpu'). Auto-detected if None.
    """
    start_time = time.time()

    # Auto-detect device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"\nLoading {split} dataset from {data_root}...")
    dataset = CUB200Dataset(root_dir=data_root, split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    print(f"Dataset size: {len(dataset)}")

    # Load frozen model
    print(f"\nInitializing DINOv2 ({model_variant})...")
    model = DINOv2Extractor(model_variant=model_variant)
    model = model.to(device)
    model.eval()

    # Extract features
    print(f"\nExtracting features...")
    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc=split)):
            images = images.to(device)
            
            # Forward pass
            features = model(images)  # shape: (batch_size, output_dim)
            
            # Accumulate
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    # Stack arrays
    features_array = np.vstack(all_features)  # shape: (n_samples, output_dim)
    labels_array = np.concatenate(all_labels)  # shape: (n_samples,)

    # L2-normalize features (default protocol)
    features_array = normalize_features(features_array)

    # Compute statistics
    mean_l2_norm = np.linalg.norm(features_array, axis=1).mean()
    elapsed = time.time() - start_time

    # Save to disk
    output_file = output_path / f"{split}_features.npz"
    np.savez(output_file, features=features_array, labels=labels_array)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Split: {split}")
    print(f"Features shape: {features_array.shape}")
    print(f"Labels shape: {labels_array.shape}")
    print(f"Mean feature L2 norm: {mean_l2_norm:.4f}")
    print(f"Saved to: {output_file}")
    print(f"Time elapsed: {elapsed:.2f}s")
    print(f"{'='*60}\n")


def main():
    """Parse arguments and run feature extraction."""
    parser = argparse.ArgumentParser(
        description="Extract and cache features from frozen DINOv2 backbone."
    )
    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Root directory of CUB-200 dataset.'
    )
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'val', 'test'],
        required=True,
        help='Dataset split to extract.'
    )
    parser.add_argument(
        '--model_variant',
        type=str,
        choices=['vits14', 'vitb14', 'vitl14'],
        default='vitb14',
        help='DINOv2 model variant (default: vitb14).'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size for DataLoader (default: 128).'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./cache',
        help='Directory to save .npz cache files (default: ./cache).'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default=None,
        help='Device to use (auto-detected if not specified).'
    )

    args = parser.parse_args()

    extract_features(
        data_root=args.data_root,
        split=args.split,
        model_variant=args.model_variant,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == '__main__':
    main()
