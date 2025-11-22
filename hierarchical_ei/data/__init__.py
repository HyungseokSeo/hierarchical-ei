"""
Data loading utilities for Hierarchical Emotional Intelligence
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Dict

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .fer_dataset import FER2013Dataset
from .transforms import get_train_transforms, get_val_transforms


def get_fer2013_loaders(
    data_dir: str = './data/fer2013',
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 48,
    augment: bool = True,
    include_valence_arousal: bool = True,
    balanced_sampling: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get FER2013 data loaders
    
    Args:
        data_dir: Directory containing fer2013.csv
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        image_size: Size to resize images to
        augment: Whether to use data augmentation
        include_valence_arousal: Include V-A values
        balanced_sampling: Use balanced sampling for training
        
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Create transforms
    train_transform = get_train_transforms(image_size, augment=augment)
    val_transform = get_val_transforms(image_size)
    
    # Create datasets
    train_dataset = FER2013Dataset(
        data_dir,
        split='train',
        transform=train_transform,
        include_valence_arousal=include_valence_arousal
    )
    
    val_dataset = FER2013Dataset(
        data_dir,
        split='val',
        transform=val_transform,
        include_valence_arousal=include_valence_arousal
    )
    
    test_dataset = FER2013Dataset(
        data_dir,
        split='test',
        transform=val_transform,
        include_valence_arousal=include_valence_arousal
    )
    
    # Print dataset statistics
    print(f"\nDataset Statistics:")
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")
    
    print(f"\nEmotion distribution (train):")
    for emotion, count in train_dataset.get_emotion_distribution().items():
        print(f"  {emotion}: {count}")
    
    # Create sampler for balanced training if requested
    train_sampler = None
    if balanced_sampling:
        class_weights = train_dataset.get_class_weights()
        sample_weights = class_weights[train_dataset.labels]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader


def get_dataloaders(
    dataset_name: str = 'fer2013',
    batch_size: int = 32,
    data_dir: Optional[str] = None,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Generic function to get data loaders for any supported dataset
    
    Args:
        dataset_name: Name of dataset ('fer2013', 'affectnet', etc.)
        batch_size: Batch size
        data_dir: Data directory (uses default if None)
        **kwargs: Additional dataset-specific arguments
        
    Returns:
        train_loader, val_loader
    """
    
    if data_dir is None:
        data_dir = f'./data/{dataset_name}'
    
    if dataset_name.lower() == 'fer2013':
        train_loader, val_loader, test_loader = get_fer2013_loaders(
            data_dir=data_dir,
            batch_size=batch_size,
            **kwargs
        )
        return train_loader, val_loader
        
    # Add more datasets here
    # elif dataset_name.lower() == 'affectnet':
    #     return get_affectnet_loaders(data_dir, batch_size, **kwargs)
    
    else:
        raise ValueError(f"Dataset {dataset_name} not supported. "
                        f"Supported datasets: ['fer2013']")


# Convenience function for testing
def test_data_loading():
    """Test function to verify data loading works"""
    
    print("Testing FER2013 data loading...")
    
    try:
        # Try to load with small batch
        train_loader, val_loader = get_dataloaders(
            'fer2013',
            batch_size=4,
            num_workers=0  # Use 0 for testing
        )
        
        # Get one batch
        batch = next(iter(train_loader))
        
        print(f"\nBatch contents:")
        print(f"  Images shape: {batch['image'].shape}")
        print(f"  Labels shape: {batch['label'].shape}")
        
        if 'valence' in batch:
            print(f"  Valence shape: {batch['valence'].shape}")
            print(f"  Valence values: {batch['valence']}")
            
        if 'arousal' in batch:
            print(f"  Arousal shape: {batch['arousal'].shape}")
            print(f"  Arousal values: {batch['arousal']}")
            
        print("\nData loading test successful!")
        return True
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return False


if __name__ == "__main__":
    test_data_loading()