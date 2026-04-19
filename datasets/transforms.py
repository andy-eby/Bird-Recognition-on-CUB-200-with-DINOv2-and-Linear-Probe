"""Data augmentation and preprocessing pipelines."""

import torch
from torchvision import transforms


def get_train_transforms():
    """
    Training augmentation pipeline.
    
    Includes:
    - RandomResizedCrop(224) with scale (0.5, 1.0)
    - RandomHorizontalFlip
    - ToTensor
    - Normalize with ImageNet statistics
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(
            size=224,
            scale=(0.5, 1.0),
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_eval_transforms():
    """
    Evaluation augmentation pipeline (validation + test).
    
    Includes:
    - Resize(256)
    - CenterCrop(224)
    - ToTensor
    - Normalize with ImageNet statistics
    """
    return transforms.Compose([
        transforms.Resize(
            size=256,
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
