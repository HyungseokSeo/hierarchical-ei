"""
Data augmentation and preprocessing transforms for emotion recognition
"""

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
import numpy as np
from typing import Optional, Tuple


class EmotionAugmentation:
    """
    Custom augmentation pipeline for facial expression images
    Preserves emotional features while adding variety
    """
    
    def __init__(
        self,
        image_size: int = 48,
        augment_prob: float = 0.5,
        preserve_emotion: bool = True
    ):
        self.image_size = image_size
        self.augment_prob = augment_prob
        self.preserve_emotion = preserve_emotion
        
    def __call__(self, image):
        """Apply augmentations while preserving emotional features"""
        
        if random.random() > self.augment_prob:
            return image
            
        # Mild rotation (emotions are rotation-sensitive)
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            image = TF.rotate(image, angle)
            
        # Slight translation
        if random.random() > 0.5:
            translate_x = random.randint(-5, 5)
            translate_y = random.randint(-5, 5)
            image = TF.affine(image, angle=0, translate=(translate_x, translate_y), 
                            scale=1.0, shear=0)
            
        # Brightness and contrast (preserve facial features)
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            image = TF.adjust_brightness(image, brightness_factor)
            
        if random.random() > 0.5:
            contrast_factor = random.uniform(0.8, 1.2)
            image = TF.adjust_contrast(image, contrast_factor)
            
        # Horizontal flip (emotions can be symmetric)
        if random.random() > 0.5 and not self.preserve_emotion:
            image = TF.hflip(image)
            
        return image


def get_train_transforms(
    image_size: int = 48,
    normalize: bool = True,
    augment: bool = True
) -> T.Compose:
   
    transforms = []
    
    # Resize if needed
    transforms.append(T.Resize((image_size, image_size)))
    
    # Data augmentation
    if augment:
        transforms.append(EmotionAugmentation(
            image_size=image_size,
            augment_prob=0.5,
            preserve_emotion=True
        ))
        
    # Convert to tensor
    transforms.append(T.ToTensor())

    # Random erasing (simulate occlusions)
    if augment:
        transforms.append(T.RandomErasing(
            p=0.1,
            scale=(0.02, 0.1),
            ratio=(0.3, 3.3),
            value=0
        ))  
        
    # Normalize
    if normalize:
        # ImageNet normalization for pretrained models
        transforms.append(T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ))
        
    return T.Compose(transforms)


def get_val_transforms(
    image_size: int = 48,
    normalize: bool = True
) -> T.Compose:
    """
    Get validation/test transforms (no augmentation)
    
    Args:
        image_size: Target image size
        normalize: Whether to normalize images
    """
    transforms = []
    
    # Resize
    transforms.append(T.Resize((image_size, image_size)))
    
    # Convert to tensor
    transforms.append(T.ToTensor())
    
    # Normalize
    if normalize:
        transforms.append(T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ))
        
    return T.Compose(transforms)


class TemporalTransform:
    """
    Transform for video/sequence data
    Maintains temporal consistency across frames
    """
    
    def __init__(
        self,
        sequence_length: int = 16,
        image_size: int = 48,
        sample_rate: int = 1
    ):
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.sample_rate = sample_rate
        
        self.spatial_transform = get_val_transforms(image_size)
        
    def __call__(self, video_frames):
        """
        Args:
            video_frames: List of PIL Images or numpy array
            
        Returns:
            Tensor of shape (T, C, H, W)
        """
        # Sample frames
        if len(video_frames) > self.sequence_length:
            indices = np.linspace(0, len(video_frames)-1, 
                                self.sequence_length, dtype=int)
            frames = [video_frames[i] for i in indices]
        else:
            frames = video_frames
            
        # Apply spatial transform to each frame
        transformed = []
        for frame in frames:
            transformed.append(self.spatial_transform(frame))
            
        # Stack into tensor
        return torch.stack(transformed)