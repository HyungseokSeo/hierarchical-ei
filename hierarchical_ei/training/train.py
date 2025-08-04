"""
Training script for Hierarchical Emotional Intelligence Model

This script implements the two-phase training:
1. JEPA pre-training for each hierarchical level
2. Active inference fine-tuning
"""

import os
import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from hierarchical_ei_model import (
    HierarchicalEmotionalIntelligence, 
    HierarchicalConfig
)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EmotionDataset(Dataset):
    """Mock dataset for demonstration - replace with actual dataset"""
    
    def __init__(self, data_dir: str, split: str = 'train', 
                 sequence_length: int = 300, transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Mock implementation - replace with actual data loading
        self.num_samples = 1000 if split == 'train' else 100
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Mock data generation - replace with actual data loading
        visual = torch.randn(self.sequence_length, 3, 64, 64)
        audio = torch.randn(self.sequence_length, 128)
        context = torch.randn(128)
        
        # Mock labels
        emotions = torch.randint(0, 8, (self.sequence_length,))
        valence = torch.randn(self.sequence_length)
        arousal = torch.randn(self.sequence_length)
        
        sample = {
            'visual': visual,
            'audio': audio,
            'context': context,
            'emotions': emotions,
            'valence': valence,
            'arousal': arousal
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample


class Trainer:
    """Trainer class for Hierarchical EI model"""
    
    def __init__(self, config_path: str, checkpoint_dir: str = 'checkpoints'):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config_dict = yaml.safe_load(f)
        
        # Setup directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize model
        self.model_config = HierarchicalConfig(**self.config_dict.get('model', {}))
        self.model = HierarchicalEmotionalIntelligence(self.model_config)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")
        
        # Training configuration
        self.train_config = self.config_dict.get('training', {})
        self.batch_size = self.train_config.get('batch_size', 32)
        self.num_epochs = self.train_config.get('num_epochs', 100)
        self.learning_rate = self.train_config.get('learning_rate', 1e-4)
        self.warmup_steps = self.train_config.get('warmup_steps', 10000)
        
        # Initialize optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2
        )
        
        # Setup data loaders
        self.setup_data_loaders()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # TensorBoard
        self.writer = SummaryWriter(f'runs/hierarchical_ei_{datetime.now():%Y%m%d_%H%M%S}')
        
    def setup_data_loaders(self):
        """Setup training and validation data loaders"""
        
        # Training dataset
        train_dataset = EmotionDataset(
            data_dir=self.train_config.get('data_dir', 'data'),
            split='train',
            sequence_length=self.train_config.get('sequence_length', 300)
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Validation dataset
        val_dataset = EmotionDataset(
            data_dir=self.train_config.get('data_dir', 'data'),
            split='val',
            sequence_length=self.train_config.get('sequence_length', 300)
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
    def train_epoch_jepa(self, phase: str = 'all'):
        """Train one epoch with JEPA objectives"""
        
        self.model.train()
        epoch_losses = {}
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.epoch} - JEPA {phase}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            visual = batch['visual'].to(self.device)
            audio = batch['audio'].to(self.device)
            context = batch['context'].to(self.device)
            
            # Forward pass
            outputs = self.model(visual, audio, context, return_all=True)
            
            # Compute JEPA losses based on phase
            losses = {}
            
            if phase in ['all', 'level1']:
                if 'level1_predictions' in outputs:
                    losses['jepa1'] = F.mse_loss(
                        outputs['level1_predictions'],
                        outputs['level1_targets'].detach()
                    )
            
            if phase in ['all', 'level2']:
                if 'level2_predictions' in outputs:
                    losses['jepa2'] = F.mse_loss(
                        outputs['level2_predictions'],
                        outputs['level2_targets'].detach()
                    )
            
            if phase in ['all', 'level3']:
                if 'level3_z3_prediction' in outputs:
                    # Simple prediction loss for Level 3
                    losses['jepa3'] = F.mse_loss(
                        outputs['level3_z3_prediction'],
                        outputs['level3_z3'][:, -1].detach()
                    )
            
            # Total loss
            if losses:
                total_loss = sum(losses.values())
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                # Update learning rate
                if self.global_step < self.warmup_steps:
                    # Linear warmup
                    lr = self.learning_rate * (self.global_step / self.warmup_steps)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    self.scheduler.step()
                
                # Log losses
                for k, v in losses.items():
                    if k not in epoch_losses:
                        epoch_losses[k] = []
                    epoch_losses[k].append(v.item())
                    self.writer.add_scalar(f'train_jepa/{k}', v.item(), self.global_step)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': total_loss.item(),
                    'lr': self.optimizer.param_groups[0]['lr']
                })
                
                self.global_step += 1
        
        # Return average losses
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        return avg_losses
    
    def train_epoch_active_inference(self):
        """Train one epoch with active inference objectives"""
        
        self.model.train()
        epoch_losses = {}
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.epoch} - Active Inference')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            visual = batch['visual'].to(self.device)
            audio = batch['audio'].to(self.device)
            context = batch['context'].to(self.device)
            emotions = batch['emotions'].to(self.device)
            
            # Forward pass
            outputs = self.model(visual, audio, context, return_all=True)
            
            # Compute all losses
            targets = {'emotions': emotions}
            losses = self.model.compute_loss(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Log losses
            for k, v in losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = []
                epoch_losses[k].append(v.item())
                self.writer.add_scalar(f'train_ai/{k}', v.item(), self.global_step)
            
            # Log additional metrics
            if 'level2_emotion_probs' in outputs:
                # Emotion recognition accuracy
                pred_emotions = outputs['level2_emotion_probs'].argmax(dim=-1)
                accuracy = (pred_emotions == emotions).float().mean()
                self.writer.add_scalar('train_ai/emotion_accuracy', accuracy, self.global_step)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': losses['total'].item(),
                'free_energy': losses.get('free_energy', 0).item()
            })
            
            self.global_step += 1
        
        # Return average losses
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        return avg_losses
    
    @torch.no_grad()
    def validate(self):
        """Validate model"""
        
        self.model.eval()
        val_losses = {}
        all_predictions = []
        all_targets = []
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            # Move batch to device
            visual = batch['visual'].to(self.device)
            audio = batch['audio'].to(self.device)
            context = batch['context'].to(self.device)
            emotions = batch['emotions'].to(self.device)
            
            # Forward pass
            outputs = self.model(visual, audio, context, return_all=True)
            
            # Compute losses
            targets = {'emotions': emotions}
            losses = self.model.compute_loss(outputs, targets)
            
            # Accumulate losses
            for k, v in losses.items():
                if k not in val_losses:
                    val_losses[k] = []
                val_losses[k].append(v.item())
            
            # Collect predictions for metrics
            if 'level2_emotion_probs' in outputs:
                pred_emotions = outputs['level2_emotion_probs'].argmax(dim=-1)
                all_predictions.extend(pred_emotions.cpu().numpy().flatten())
                all_targets.extend(emotions.cpu().numpy().flatten())
        
        # Compute average losses
        avg_losses = {k: np.mean(v) for k, v in val_losses.items()}
        
        # Compute metrics
        if all_predictions:
            from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
            
            accuracy = accuracy_score(all_targets, all_predictions)
            f1 = f1_score(all_targets, all_predictions, average='weighted')
            
            avg_losses['accuracy'] = accuracy
            avg_losses['f1_score'] = f1
            
            logger.info(f"Validation - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return avg_losses
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config_dict
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{self.epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        logger.info(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train(self, resume_from: Optional[str] = None):
        """Main training loop"""
        
        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)
        
        logger.info("Starting training...")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Phase 1: JEPA Pre-training
        jepa_epochs = self.train_config.get('jepa_epochs', 50)
        
        for phase_epoch in range(jepa_epochs):
            self.epoch = phase_epoch
            logger.info(f"\n{'='*50}")
            logger.info(f"JEPA Pre-training - Epoch {self.epoch}/{jepa_epochs}")
            
            # Train with JEPA objectives
            train_losses = self.train_epoch_jepa(phase='all')
            logger.info(f"Train losses: {train_losses}")
            
            # Validate
            if (self.epoch + 1) % 5 == 0:
                val_losses = self.validate()
                logger.info(f"Val losses: {val_losses}")
                
                # Save checkpoint
                if val_losses['total'] < self.best_val_loss:
                    self.best_val_loss = val_losses['total']
                    self.save_checkpoint(is_best=True)
        
        # Phase 2: Active Inference Fine-tuning
        ai_epochs = self.train_config.get('ai_epochs', 50)
        
        for phase_epoch in range(ai_epochs):
            self.epoch = jepa_epochs + phase_epoch
            logger.info(f"\n{'='*50}")
            logger.info(f"Active Inference Fine-tuning - Epoch {self.epoch}/{jepa_epochs + ai_epochs}")
            
            # Train with full objectives
            train_losses = self.train_epoch_active_inference()
            logger.info(f"Train losses: {train_losses}")
            
            # Validate
            val_losses = self.validate()
            logger.info(f"Val losses: {val_losses}")
            
            # Save checkpoint
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.save_checkpoint(is_best=True)
            elif (self.epoch + 1) % 10 == 0:
                self.save_checkpoint(is_best=False)
        
        logger.info("Training completed!")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train Hierarchical EI Model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    
    args = parser.parse_args()
    
    # Set GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    
    # Initialize trainer
    trainer = Trainer(args.config, args.checkpoint_dir)
    
    # Start training
    trainer.train(resume_from=args.resume)


if __name__ == '__main__':
    main()
