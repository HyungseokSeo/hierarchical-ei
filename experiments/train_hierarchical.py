"""
Training Script for Hierarchical Emotional Intelligence Model
IEEE TAC Submission - JEPA + Active Inference Framework

Author: Hyungseok Seo
Date: 2025
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import wandb

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from hierarchical_ei.models.hierarchical_ei import HierarchicalConfig
from hierarchical_ei.metrics.ercs import ERCS
from hierarchical_ei.metrics.cedi import CEDI


class HierarchicalTrainer:
    """
    Trainer for Hierarchical Emotional Intelligence Model
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: HierarchicalConfig,
        device: str = 'cuda',
        experiment_name: Optional[str] = None
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Create experiment name
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"hierarchical_ei_{timestamp}"
        self.experiment_name = experiment_name
        
        # Setup directories
        self.checkpoint_dir = Path('experiments/results/checkpoints') / experiment_name
        self.log_dir = Path('experiments/results/logs') / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics
        self.ercs = ERCS()
        self.cedi = CEDI()
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=50,
            T_mult=2,
            eta_min=1e-6
        )
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        
        # Tensorboard writer
        self.writer = SummaryWriter(self.log_dir)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_accuracy = 0.0
        
    def compute_hierarchical_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-level hierarchical loss
        
        Components:
        1. Classification loss (emotion predictions)
        2. Free energy regularization
        3. Valence/Arousal regression loss
        4. Hierarchical consistency loss (via ERCS)
        """
        losses = {}
        
        # 1. Classification loss
        if 'emotion_logits' in outputs and 'labels' in targets:
            classification_loss = self.ce_loss(
                outputs['emotion_logits'],
                targets['labels']
            )
            losses['classification'] = classification_loss.item()
        else:
            classification_loss = 0.0
            
        # 2. Free energy regularization
        if 'free_energy' in outputs:
            # Minimize free energy
            free_energy_loss = outputs['free_energy'].mean()
            losses['free_energy'] = free_energy_loss.item()
        else:
            free_energy_loss = 0.0
            
        # 3. Valence/Arousal regression
        valence_loss = 0.0
        arousal_loss = 0.0
        
        if 'valence' in outputs and 'valence' in targets:
            valence_loss = self.mse_loss(
                outputs['valence'],
                targets['valence']
            )
            losses['valence'] = valence_loss.item()
            
        if 'arousal' in outputs and 'arousal' in targets:
            arousal_loss = self.mse_loss(
                outputs['arousal'],
                targets['arousal']
            )
            losses['arousal'] = arousal_loss.item()
            
        # 4. Hierarchical consistency (ERCS as auxiliary loss)
        ercs_results = self.ercs(outputs)
        consistency_loss = 1.0 - ercs_results['ercs_score'].mean()
        losses['consistency'] = consistency_loss.item()
        
        # Combine losses with weights
        total_loss = (
            classification_loss * 1.0 +
            free_energy_loss * self.config.free_energy_beta +
            valence_loss * 0.5 +
            arousal_loss * 0.5 +
            consistency_loss * 0.1
        )
        
        losses['total'] = total_loss.item()
        
        return total_loss, losses
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch
        """
        self.model.train()
        epoch_losses = []
        epoch_metrics = {'accuracy': [], 'ercs': [], 'cedi': []}
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Prepare targets
            targets = {'labels': labels}
            
            # Add valence/arousal if available
            if 'valence' in batch:
                targets['valence'] = batch['valence'].to(self.device)
            if 'arousal' in batch:
                targets['arousal'] = batch['arousal'].to(self.device)
                
            # Forward pass
            outputs = self.model(images)
            
            # Compute loss
            loss, loss_components = self.compute_hierarchical_loss(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                # Accuracy
                if 'emotion_logits' in outputs:
                    preds = outputs['emotion_logits'].argmax(dim=-1)
                    accuracy = (preds == labels).float().mean().item()
                    epoch_metrics['accuracy'].append(accuracy)
                    
                # ERCS score
                ercs_results = self.ercs(outputs)
                epoch_metrics['ercs'].append(ercs_results['ercs_score'].mean().item())
                
                # CEDI score  
                cedi_results = self.cedi(outputs, outputs.get('free_energy'))
                epoch_metrics['cedi'].append(cedi_results['cedi_score'].mean().item())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{accuracy:.3f}",
                'ercs': f"{epoch_metrics['ercs'][-1]:.3f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            epoch_losses.append(loss_components)
            
            # Log to tensorboard
            if self.global_step % 10 == 0:
                for key, value in loss_components.items():
                    self.writer.add_scalar(f'train/loss_{key}', value, self.global_step)
                self.writer.add_scalar('train/accuracy', accuracy, self.global_step)
                self.writer.add_scalar('train/ercs', epoch_metrics['ercs'][-1], self.global_step)
                self.writer.add_scalar('train/cedi', epoch_metrics['cedi'][-1], self.global_step)
                
            self.global_step += 1
            
        # Compute epoch averages
        avg_loss = np.mean([l['total'] for l in epoch_losses])
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        
        return {'loss': avg_loss, **avg_metrics}
        
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validation loop
        """
        self.model.eval()
        
        val_losses = []
        all_preds = []
        all_labels = []
        val_metrics = {'ercs': [], 'cedi': []}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                targets = {'labels': labels}
                if 'valence' in batch:
                    targets['valence'] = batch['valence'].to(self.device)
                if 'arousal' in batch:
                    targets['arousal'] = batch['arousal'].to(self.device)
                    
                # Forward pass
                outputs = self.model(images)
                
                # Compute loss
                loss, loss_components = self.compute_hierarchical_loss(outputs, targets)
                val_losses.append(loss_components)
                
                # Predictions
                if 'emotion_logits' in outputs:
                    preds = outputs['emotion_logits'].argmax(dim=-1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                # Metrics
                ercs_results = self.ercs(outputs)
                val_metrics['ercs'].append(ercs_results['ercs_score'].mean().item())
                
                cedi_results = self.cedi(outputs, outputs.get('free_energy'))
                val_metrics['cedi'].append(cedi_results['cedi_score'].mean().item())
        
        # Calculate validation metrics
        avg_loss = np.mean([l['total'] for l in val_losses])
        
        if all_preds:
            accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        else:
            accuracy = 0.0
            
        avg_metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'ercs': np.mean(val_metrics['ercs']),
            'cedi': np.mean(val_metrics['cedi'])
        }
        
        # Log to tensorboard
        for key, value in avg_metrics.items():
            self.writer.add_scalar(f'val/{key}', value, self.current_epoch)
            
        return avg_metrics
        
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """
        Save model checkpoint
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best model with accuracy: {metrics['accuracy']:.4f}")
            
        # Keep only last 5 checkpoints
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        if len(checkpoints) > 5:
            for old_checkpoint in checkpoints[:-5]:
                old_checkpoint.unlink()
                
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100
    ):
        """
        Main training loop
        """
        print(f"Starting training: {self.experiment_name}")
        print(f"Device: {self.device}")
        print(f"Total epochs: {num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            print(f"\nEpoch {epoch} - Train: Loss={train_metrics['loss']:.4f}, "
                  f"Acc={train_metrics['accuracy']:.4f}, "
                  f"ERCS={train_metrics['ercs']:.4f}, "
                  f"CEDI={train_metrics['cedi']:.4f}")
            
            # Validation
            val_metrics = self.validate(val_loader)
            print(f"Epoch {epoch} - Val: Loss={val_metrics['loss']:.4f}, "
                  f"Acc={val_metrics['accuracy']:.4f}, "
                  f"ERCS={val_metrics['ercs']:.4f}, "
                  f"CEDI={val_metrics['cedi']:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Save checkpoint
            is_best = val_metrics['accuracy'] > self.best_accuracy
            if is_best:
                self.best_accuracy = val_metrics['accuracy']
            self.save_checkpoint(val_metrics, is_best)
            
            # Log to wandb if available
            if wandb.run is not None:
                wandb.log({
                    'train/loss': train_metrics['loss'],
                    'train/accuracy': train_metrics['accuracy'],
                    'val/loss': val_metrics['loss'],
                    'val/accuracy': val_metrics['accuracy'],
                    'val/ercs': val_metrics['ercs'],
                    'val/cedi': val_metrics['cedi'],
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch': epoch
                })
                
        print(f"\nTraining complete! Best accuracy: {self.best_accuracy:.4f}")
        self.writer.close()
        return self.best_accuracy


def main():
    parser = argparse.ArgumentParser(description='Train Hierarchical Emotional Intelligence Model')
    parser.add_argument('--config', type=str, default='experiments/configs/ieee_tac.yaml',
                       help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default='fer2013',
                       choices=['fer2013', 'affectnet', 'rafdb'],
                       help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name')
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = HierarchicalConfig(**config_dict)
    else:
        config = HierarchicalConfig()
        
    # Override config with command line arguments
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    
    # Initialize wandb if requested
    if args.wandb:
        wandb.init(
            project='hierarchical-ei',
            name=args.experiment_name,
            config=config.__dict__
        )
    
    # TODO: Import your actual model class here
    # from hierarchical_ei.models.hierarchical_ei import HierarchicalEmotionalIntelligence
    # model = HierarchicalEmotionalIntelligence(config)
    
    # For now, create a placeholder
    print("Note: You need to import and instantiate your actual model class")
    print("The training infrastructure is ready!")
    
    # TODO: Create data loaders
    # from hierarchical_ei.data import get_dataloaders
    # train_loader, val_loader = get_dataloaders(args.dataset, config.batch_size)
    
    # Initialize trainer
    # trainer = HierarchicalTrainer(model, config, args.device, args.experiment_name)
    
    # Train model
    # best_accuracy = trainer.train(train_loader, val_loader, args.epochs)
    
    print("\nTraining script structure created successfully!")
    print("Next steps:")
    print("1. Complete your model implementation in hierarchical_ei/models/")
    print("2. Create data loaders in hierarchical_ei/data/")
    print("3. Uncomment the model instantiation and training lines")


if __name__ == '__main__':
    main()