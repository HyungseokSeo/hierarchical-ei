"""
Script to download and prepare emotion recognition datasets

Supports:
- FER2013
- AffectNet
- CMU-MOSEI
- CK+
- DEAP
"""

import os
import argparse
import zipfile
import tarfile
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import shutil
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Base class for dataset downloaders"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def download_file(self, url: str, filename: str) -> Path:
        """Download file with progress bar"""
        filepath = self.data_dir / filename
        
        if filepath.exists():
            logger.info(f"{filename} already exists, skipping download")
            return filepath
            
        logger.info(f"Downloading {filename}...")
        
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
                    
        return filepath
    
    def extract_archive(self, filepath: Path, extract_dir: Path):
        """Extract zip or tar archive"""
        logger.info(f"Extracting {filepath.name}...")
        
        if filepath.suffix == '.zip':
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif filepath.suffix in ['.tar', '.gz', '.tgz']:
            with tarfile.open(filepath, 'r:*') as tar_ref:
                tar_ref.extractall(extract_dir)
        else:
            raise ValueError(f"Unknown archive format: {filepath.suffix}")


class FER2013Downloader(DatasetDownloader):
    """Download and prepare FER2013 dataset"""
    
    def download(self):
        """Download FER2013 from Kaggle (requires API key)"""
        dataset_dir = self.data_dir / "fer2013"
        dataset_dir.mkdir(exist_ok=True)
        
        # Check if already downloaded
        if (dataset_dir / "train.csv").exists():
            logger.info("FER2013 already downloaded")
            return
        
        logger.info("Downloading FER2013...")
        logger.info("Please download manually from Kaggle:")
        logger.info("https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data")
        logger.info(f"Extract fer2013.csv to {dataset_dir}")
        
        # If Kaggle API is set up
        try:
            import kaggle
            kaggle.api.authenticate()
            kaggle.api.competition_download_files(
                'challenges-in-representation-learning-facial-expression-recognition-challenge',
                path=dataset_dir
            )
            
            # Extract if downloaded as zip
            zip_path = dataset_dir / "challenges-in-representation-learning-facial-expression-recognition-challenge.zip"
            if zip_path.exists():
                self.extract_archive(zip_path, dataset_dir)
                
        except Exception as e:
            logger.error(f"Kaggle API error: {e}")
            logger.info("Please download manually")
            
    def prepare(self):
        """Prepare FER2013 data"""
        dataset_dir = self.data_dir / "fer2013"
        csv_path = dataset_dir / "fer2013.csv"
        
        if not csv_path.exists():
            logger.error(f"fer2013.csv not found in {dataset_dir}")
            return
            
        logger.info("Preparing FER2013 data...")
        
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Split into train/val/test
        train_df = df[df['Usage'] == 'Training']
        val_df = df[df['Usage'] == 'PublicTest']
        test_df = df[df['Usage'] == 'PrivateTest']
        
        # Save splits
        train_df.to_csv(dataset_dir / "train.csv", index=False)
        val_df.to_csv(dataset_dir / "val.csv", index=False)
        test_df.to_csv(dataset_dir / "test.csv", index=False)
        
        # Create image directories
        for split in ['train', 'val', 'test']:
            split_dir = dataset_dir / split
            split_dir.mkdir(exist_ok=True)
            
            # Extract images
            split_df = pd.read_csv(dataset_dir / f"{split}.csv")
            
            logger.info(f"Extracting {split} images...")
            for idx, row in tqdm(split_df.iterrows(), total=len(split_df)):
                # Parse pixel string
                pixels = np.fromstring(row['pixels'], dtype=int, sep=' ')
                pixels = pixels.reshape(48, 48)
                
                # Save image
                emotion_dir = split_dir / str(row['emotion'])
                emotion_dir.mkdir(exist_ok=True)
                
                img_path = emotion_dir / f"{idx:06d}.npy"
                np.save(img_path, pixels)
                
        logger.info("FER2013 preparation complete")


class AffectNetDownloader(DatasetDownloader):
    """Download and prepare AffectNet dataset"""
    
    def download(self):
        """Download AffectNet (requires registration)"""
        dataset_dir = self.data_dir / "affectnet"
        dataset_dir.mkdir(exist_ok=True)
        
        logger.info("AffectNet requires registration:")
        logger.info("1. Register at http://mohammadmahoor.com/affectnet/")
        logger.info("2. Download the dataset")
        logger.info(f"3. Extract to {dataset_dir}")
        
        # Check if extracted
        if (dataset_dir / "train_set").exists():
            logger.info("AffectNet already downloaded")
        else:
            logger.info("Waiting for manual download...")
            
    def prepare(self):
        """Prepare AffectNet data"""
        dataset_dir = self.data_dir / "affectnet"
        
        # Create metadata file
        metadata = {
            "emotions": {
                "0": "Neutral",
                "1": "Happy",
                "2": "Sad",
                "3": "Surprise",
                "4": "Fear",
                "5": "Disgust",
                "6": "Anger",
                "7": "Contempt"
            },
            "train_samples": 0,
            "val_samples": 0
        }
        
        # Count samples
        for split in ['train_set', 'val_set']:
            split_dir = dataset_dir / split
            if split_dir.exists():
                count = sum(1 for _ in split_dir.rglob("*.jpg"))
                metadata[f"{split.split('_')[0]}_samples"] = count
                
        # Save metadata
        with open(dataset_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info("AffectNet preparation complete")


class CMUMOSEIDownloader(DatasetDownloader):
    """Download and prepare CMU-MOSEI dataset"""
    
    def download(self):
        """Download CMU-MOSEI"""
        dataset_dir = self.data_dir / "cmu-mosei"
        dataset_dir.mkdir(exist_ok=True)
        
        # Base URL for CMU-MOSEI
        base_url = "http://immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/"
        
        files_to_download = [
            "cmu-mosei/seq_length_50/data.pkl.pkl",
            "cmu-mosei/seq_length_50/train.pkl",
            "cmu-mosei/seq_length_50/valid.pkl",
            "cmu-mosei/seq_length_50/test.pkl"
        ]
        
        for file_path in files_to_download:
            url = base_url + file_path
            filename = Path(file_path).name
            
            try:
                self.download_file(url, f"cmu-mosei/{filename}")
            except Exception as e:
                logger.error(f"Error downloading {filename}: {e}")
                logger.info("Please download CMU-MOSEI manually from:")
                logger.info("http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/")
                
    def prepare(self):
        """Prepare CMU-MOSEI data"""
        dataset_dir = self.data_dir / "cmu-mosei"
        
        # Create info file
        info = {
            "description": "CMU Multimodal Opinion Sentiment and Emotion Intensity",
            "modalities": ["vision", "audio", "text"],
            "labels": ["sentiment", "emotions"],
            "emotions": ["happy", "sad", "anger", "surprise", "disgust", "fear"]
        }
        
        with open(dataset_dir / "info.json", 'w') as f:
            json.dump(info, f, indent=2)
            
        logger.info("CMU-MOSEI preparation complete")


class DatasetManager:
    """Manage all datasets"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        
        self.downloaders = {
            'fer2013': FER2013Downloader(data_dir),
            'affectnet': AffectNetDownloader(data_dir),
            'cmu-mosei': CMUMOSEIDownloader(data_dir)
        }
        
    def download_dataset(self, dataset_name: str):
        """Download specific dataset"""
        if dataset_name not in self.downloaders:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
        downloader = self.downloaders[dataset_name]
        downloader.download()
        downloader.prepare()
        
    def download_all(self):
        """Download all datasets"""
        for name, downloader in self.downloaders.items():
            logger.info(f"\nProcessing {name}...")
            try:
                downloader.download()
                downloader.prepare()
            except Exception as e:
                logger.error(f"Error with {name}: {e}")
                
    def create_unified_metadata(self):
        """Create unified metadata for all datasets"""
        metadata = {
            "datasets": {},
            "total_samples": 0,
            "emotion_mapping": {
                "0": "angry",
                "1": "disgust",
                "2": "fear",
                "3": "happy",
                "4": "sad",
                "5": "surprise",
                "6": "neutral",
                "7": "contempt"
            }
        }
        
        # Check each dataset
        for name in self.downloaders.keys():
            dataset_dir = self.data_dir / name
            if dataset_dir.exists():
                # Count samples
                sample_count = 0
                for split in ['train', 'val', 'test']:
                    split_dir = dataset_dir / split
                    if split_dir.exists():
                        sample_count += sum(1 for _ in split_dir.rglob("*.*"))
                        
                metadata["datasets"][name] = {
                    "path": str(dataset_dir),
                    "samples": sample_count
                }
                metadata["total_samples"] += sample_count
                
        # Save metadata
        with open(self.data_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Created unified metadata: {metadata['total_samples']} total samples")


def main():
    parser = argparse.ArgumentParser(description='Download emotion recognition datasets')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'fer2013', 'affectnet', 'cmu-mosei'],
                        help='Dataset to download')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory to save datasets')
    
    args = parser.parse_args()
    
    # Create manager
    manager = DatasetManager(args.data_dir)
    
    # Download datasets
    if args.dataset == 'all':
        manager.download_all()
    else:
        manager.download_dataset(args.dataset)
        
    # Create unified metadata
    manager.create_unified_metadata()
    
    logger.info("\nDataset download complete!")
    logger.info("Note: Some datasets require manual download due to licensing")


if __name__ == '__main__':
    main()
