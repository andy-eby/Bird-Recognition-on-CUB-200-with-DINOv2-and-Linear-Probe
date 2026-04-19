"""CUB-200-2011 dataset loader."""

import os
from pathlib import Path
from typing import Tuple, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import get_train_transforms, get_eval_transforms


class CUB200Dataset(Dataset):
    """
    CUB-200-2011 Fine-Grained Bird Classification Dataset.
    
    Dataset structure:
    - images.txt: image_id → relative_path
    - image_class_labels.txt: image_id → class_id (1-indexed)
    - train_test_split.txt: image_id → is_training (1 = train, 0 = test)
    - classes.txt: class_id → human_readable_name
    """

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[object] = None,
    ):
        """
        Args:
            root_dir (str): Root directory of CUB-200-2011 dataset.
            split (str): One of 'train', 'val', or 'test'.
            transform (callable, optional): Image transform to apply.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform or (get_train_transforms() if split == 'train' else get_eval_transforms())

        # Parse all index files
        self._parse_index_files()
        
        # Build split indices
        self._build_splits()

    def _parse_index_files(self):
        """Parse CUB-200 index files in order."""
        # 1. Parse images.txt: image_id → relative_path
        images_file = self.root_dir / 'images.txt'
        self.image_paths = {}  # image_id → relative_path
        
        with open(images_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                image_id = int(parts[0])
                relative_path = parts[1]
                self.image_paths[image_id] = relative_path

        # 2. Parse image_class_labels.txt: image_id → class_id (1-indexed)
        class_labels_file = self.root_dir / 'image_class_labels.txt'
        self.image_labels = {}  # image_id → class_id (0-indexed after conversion)
        
        with open(class_labels_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                image_id = int(parts[0])
                class_id = int(parts[1]) - 1  # Convert to 0-indexed
                self.image_labels[image_id] = class_id

        # 3. Parse train_test_split.txt: image_id → is_training (1 = train, 0 = test)
        split_file = self.root_dir / 'train_test_split.txt'
        self.is_training = {}  # image_id → is_training_flag
        
        with open(split_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                image_id = int(parts[0])
                is_training_flag = int(parts[1])
                self.is_training[image_id] = is_training_flag

        # 4. Parse classes.txt: class_id → human_readable_name
        classes_file = self.root_dir / 'classes.txt'
        self.class_names = {}  # class_id (0-indexed) → class_name
        
        with open(classes_file, 'r') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                class_id = int(parts[0]) - 1  # Convert to 0-indexed
                class_name = parts[1]
                self.class_names[class_id] = class_name

        # Store as ordered list
        self.class_names = [self.class_names[i] for i in range(len(self.class_names))]

    def _build_splits(self):
        """
        Build train/val/test splits.
        
        Strategy:
        - Use official train/test split
        - Carve validation set from training: last 10% of each class (stratified)
        - Keep test set untouched
        """
        # Separate training and test image_ids
        train_image_ids = [img_id for img_id, is_train in self.is_training.items() if is_train == 1]
        test_image_ids = [img_id for img_id, is_train in self.is_training.items() if is_train == 0]

        # Group training images by class for stratified split
        train_by_class = {}
        for img_id in train_image_ids:
            class_id = self.image_labels[img_id]
            if class_id not in train_by_class:
                train_by_class[class_id] = []
            train_by_class[class_id].append(img_id)

        # Sort each class's images for reproducible stratified split
        for class_id in train_by_class:
            train_by_class[class_id].sort()

        # Carve out 10% of each class for validation (from the end)
        val_image_ids = []
        final_train_image_ids = []

        for class_id in sorted(train_by_class.keys()):
            class_images = train_by_class[class_id]
            num_val = max(1, len(class_images) // 10)  # At least 1 per class
            
            final_train_image_ids.extend(class_images[:-num_val])
            val_image_ids.extend(class_images[-num_val:])

        # Assign splits
        if self.split == 'train':
            self.image_ids = sorted(final_train_image_ids)
        elif self.split == 'val':
            self.image_ids = sorted(val_image_ids)
        elif self.split == 'test':
            self.image_ids = sorted(test_image_ids)
        else:
            raise ValueError(f"Unknown split: {self.split}")

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get image and label.
        
        Args:
            idx (int): Index in split.
            
        Returns:
            Tuple[torch.Tensor, int]: (image_tensor, label)
        """
        image_id = self.image_ids[idx]
        
        # Load image
        image_path = self.root_dir / 'images' / self.image_paths[image_id]
        image = Image.open(image_path).convert('RGB')
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.image_labels[image_id]
        
        return image, label
