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

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

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

# TODO: Create data loaders
from hierarchical_ei.data import get_dataloaders
from hierarchical_ei.training.trainer import HierarchicalTrainer
from hierarchical_ei.models.hierarchical_ei import HierarchicalConfig
from hierarchical_ei.metrics.ercs import ERCS
from hierarchical_ei.metrics.cedi import CEDI

from hierarchical_ei.models.hierarchical_model import create_hierarchical_model

def main():
    parser = argparse.ArgumentParser(description='Train Hierarchical Emotional Intelligence Model')
    parser.add_argument('--config', type=str, default='experiments/configs/ieee_tac.yaml',
                       help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default='fer2013',
                       choices=['fer2013', 'affectnet', 'rafdb'],
                       help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='./data')

    # Parse arguments - THIS CREATES 'args'
    args = parser.parse_args() 
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='hierarchical_ei')
    parser.add_argument('--pretrained', action='store_true')
    
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01) 
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--num_workers', type=int, default=4)

    # Logging
    parser.add_argument('--use_wandb', action='store_true',
                    help='Use Weights & Biases logging')
    parser.add_argument('--project_name', type=str, default='hierarchical-ei')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
   
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name')
                       
    args = parser.parse_args()    
    
    # Define device BEFORE using it
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(args.device)
    print(f"Using device: {device}")
       
    # Create model
    print("Creating model...")
    model = create_hierarchical_model(pretrained=args.pretrained)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created: {total_params:,} total parameters, {trainable_params:,} trainable")
    
     # Load data
    print("Loading dataset...")
    train_loader, val_loader = get_dataloaders(
        args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"Dataset loaded: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Create trainer
    trainer = HierarchicalTrainer(
        model=model,
        device=device,
        config=vars(args),
        use_wandb=args.use_wandb
    )
   
    # Training loop
    print("\nStarting training...")
    best_val_acc = 0
    best_ercs = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*50}")
        
        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)
        
        # Validate
        val_metrics = trainer.validate(val_loader)
        
        # Update scheduler : learning_rate
        trainer.scheduler.step()
        
        # Print metrics
        print(f"\nEpoch {epoch} Summary:")
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%, "
              f"ERCS: {train_metrics['ercs']:.3f}, CEDI: {train_metrics['cedi']:.3f}")
        print(f"Val - Loss: {val_metrics['val_loss']:.4f}, Acc: {val_metrics['val_accuracy']:.2f}%, "
              f"ERCS: {val_metrics['val_ercs']:.3f}, CEDI: {val_metrics['val_cedi']:.3f}")
        
        # Save best model
        if val_metrics['val_accuracy'] > best_val_acc:
            best_val_acc = val_metrics['val_accuracy']
            best_ercs = val_metrics['val_ercs']
            
            save_path = os.path.join(args.save_dir, f'best_model.pth')
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_acc': best_val_acc,
                'val_ercs': best_ercs,
                'args': args
            }, save_path)
            print(f"✓ Saved best model (Acc: {best_val_acc:.2f}%, ERCS: {best_ercs:.3f})")
        
    # Log to wandb
    if args.use_wandb:
        wandb.log({
            **train_metrics,
            **val_metrics,
            'epoch': epoch,
            'learning_rate': trainer.scheduler.get_last_lr()[0]
        })
   
    print(f"\n{'='*50}")
    print(f"Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best ERCS Score: {best_ercs:.3f}")
    print(f"{'='*50}")
    
    if args.use_wandb:
        wandb.finish()    
        
# Compute hierarchical loss 
def compute_loss(outputs, targets):
    
    emotion_loss = F.cross_entropy(outputs['emotion_logits'], targets['label'])
    
    # Add prediction error regularization
    pred_error_loss = (outputs['prediction_error_l1'].pow(2).mean() + 
                       outputs['prediction_error_l2'].pow(2).mean())
    
    total_loss = emotion_loss + 0.1 * pred_error_loss
    
    return total_loss, {
        'emotion_loss': emotion_loss.item(),
        'pred_error_loss': pred_error_loss.item()
    }        
  
if __name__ == '__main__':
    main() 
    