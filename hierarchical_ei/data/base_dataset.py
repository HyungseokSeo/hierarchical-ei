"""
Base Dataset Class for Emotion Recognition
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, Tuple, List
import numpy as np
from PIL import Image
from abc import ABC, abstractmethod


class EmotionDataset(Dataset, ABC):
    """
    Base class for emotion recognition datasets
    """
    
    # Standard emotion labels across datasets
    EMOTION_LABELS = {
        'angry': 0,
        'disgust': 1,
        'fear': 2,
        'happy': 3,
        'sad': 4,
        'surprise': 5,
        'neutral': 6,
        'contempt': 7  # Optional, not in all datasets
    }
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform=None,
        include_valence_arousal: bool = False
    ):
        """
        Args:
            root_dir: Root directory of dataset
            split: 'train', 'val', or 'test'
            transform: Optional transform to apply to images
            include_valence_arousal: Whether to include V-A values if available
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.include_valence_arousal = include_valence_arousal
        
        # To be filled by subclasses
        self.data = []
        self.labels = []
        self.valence = []
        self.arousal = []
        
    @abstractmethod
    def load_data(self):
        """Load dataset-specific data"""
        pass
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset
        
        Returns:
            Dictionary containing:
                - image: Tensor of shape (C, H, W)
                - label: Integer emotion label
                - valence: Float value (-1 to 1) if available
                - arousal: Float value (-1 to 1) if available
                - idx: Original index in dataset
        """
        sample = {
            'idx': idx,
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }
        
        # Load and process image
        image = self.load_image(idx)
        transform = transforms.Compose([
            transforms.ToPILImage(),  # If starting from numpy array
            transforms.Resize((48, 48)),
            transforms.ToTensor(),  # This MUST come before Normalize
            transforms.Normalize(mean=[0.5], std=[0.5])  # This expects a tensor
        ])
        if self.transform:
            image = self.transform(image)
        sample['image'] = image
        
        # Add valence and arousal if available
        if self.include_valence_arousal:
            if len(self.valence) > idx:
                sample['valence'] = torch.tensor(self.valence[idx], dtype=torch.float32)
            if len(self.arousal) > idx:
                sample['arousal'] = torch.tensor(self.arousal[idx], dtype=torch.float32)
                
        return sample
        
    @abstractmethod
    def load_image(self, idx: int) -> Image.Image:
        """Load image at given index"""
        pass
        
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling imbalanced datasets
        """
        class_counts = np.bincount(self.labels)
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        return torch.tensor(class_weights, dtype=torch.float32)