"""
FER2013 Dataset Implementation
"""

import os
import csv
import numpy as np
from PIL import Image
import torch
from typing import Optional
from .base_dataset import EmotionDataset


class FER2013Dataset(EmotionDataset):
    """
    FER2013 (Facial Expression Recognition 2013) Dataset
    
    The dataset consists of 48x48 pixel grayscale images of faces.
    7 emotion categories: angry, disgust, fear, happy, sad, surprise, neutral
    
    Dataset structure:
    - fer2013.csv with columns: emotion, pixels, Usage
    - 28,709 training samples
    - 3,589 validation samples
    - 3,589 test samples
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform=None,
        include_valence_arousal: bool = False,
        csv_file: str = 'fer2013.csv'
    ):
        """
        Args:
            root_dir: Directory containing fer2013.csv
            split: 'train', 'val', or 'test'
            transform: Optional transform to apply
            include_valence_arousal: Include estimated V-A values
            csv_file: Name of the CSV file
        """
        super().__init__(root_dir, split, transform, include_valence_arousal)
        
        self.csv_path = os.path.join(root_dir, csv_file)
        
        # FER2013 specific emotion mapping
        self.fer_emotion_map = {
            0: 'angry',
            1: 'disgust', 
            2: 'fear',
            3: 'happy',
            4: 'sad',
            5: 'surprise',
            6: 'neutral'
        }
        
        # Estimated valence-arousal values for each emotion
        # Based on Russell's circumplex model
        self.emotion_to_va = {
            'angry': (-0.8, 0.8),     # Negative valence, high arousal
            'disgust': (-0.9, 0.2),    # Very negative valence, low arousal
            'fear': (-0.7, 0.7),       # Negative valence, high arousal
            'happy': (0.9, 0.6),       # Positive valence, moderate arousal
            'sad': (-0.8, -0.5),       # Negative valence, low arousal
            'surprise': (0.1, 0.8),    # Neutral valence, high arousal
            'neutral': (0.0, 0.0)      # Neutral valence, neutral arousal
        }
        
        self.load_data()
        
    def load_data(self):
        """
        Load FER2013 data from CSV file
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"FER2013 CSV not found at {self.csv_path}. "
                "Please download from: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge"
            )
            
        # Map split names to FER2013 Usage values
        split_map = {
            'train': 'Training',
            'val': 'PublicTest',
            'test': 'PrivateTest'
        }
        
        if self.split not in split_map:
            raise ValueError(f"Split must be one of {list(split_map.keys())}")
            
        usage_value = split_map[self.split]
        
        # Read CSV and filter by split
        with open(self.csv_path, 'r') as f:
            csv_reader = csv.DictReader(f)
            
            for row in csv_reader:
                if row['Usage'] == usage_value:
                    # Parse emotion label
                    emotion_idx = int(row['emotion'])
                    self.labels.append(emotion_idx)
                    
                    # Parse pixels
                    pixels = np.array(row['pixels'].split(), dtype=np.uint8)
                    pixels = pixels.reshape(48, 48)
                    self.data.append(pixels)
                    
                    # Add valence-arousal if requested
                    if self.include_valence_arousal:
                        emotion_name = self.fer_emotion_map[emotion_idx]
                        va = self.emotion_to_va[emotion_name]
                        self.valence.append(va[0])
                        self.arousal.append(va[1])
                        
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        
        if self.include_valence_arousal:
            self.valence = np.array(self.valence)
            self.arousal = np.array(self.arousal)
            
        print(f"Loaded {len(self.data)} samples for {self.split} split")
        
    def load_image(self, idx: int) -> Image.Image:
        """
        Load and convert grayscale array to PIL Image
        """
        img_array = self.data[idx]
        
        # Convert to PIL Image
        image = Image.fromarray(img_array, mode='L')
        
        # Convert to RGB (3 channels) for model compatibility
        image = image.convert('RGB')
        
        return image
        
    def get_emotion_distribution(self) -> dict:
        """
        Get distribution of emotions in the dataset
        """
        unique, counts = np.unique(self.labels, return_counts=True)
        distribution = {}
        
        for label, count in zip(unique, counts):
            emotion_name = self.fer_emotion_map[label]
            distribution[emotion_name] = count
            
        return distribution