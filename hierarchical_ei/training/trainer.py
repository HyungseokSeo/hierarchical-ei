# hierarchical_ei/training/trainer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
# from torch.amp.autocast('cuda', args)
from tqdm import tqdm
import wandb
from typing import Dict, Optional
import numpy as np

from hierarchical_ei.metrics.emotional_metrics import ERCS, CEDI

class HierarchicalTrainer:
    """Training loop with ERCS and CEDI metrics"""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: Dict,
        use_wandb: bool = False
    ):
        self.model = model
        self.device = device
        self.config = config
        self.use_wandb = use_wandb
        
        # Metrics
        self.ercs = ERCS(n_bins=10)
        self.cedi = CEDI(n_classes=7)
        
        # Optimizer with level-specific learning rates
        self.optimizer = AdamW([
            {'params': model.level1.parameters(), 'lr': 1e-3},
            {'params': model.level2.parameters(), 'lr': 5e-4},
            {'params': model.level3.parameters(), 'lr': 1e-4}
        ], weight_decay=0.01)
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Mixed precision training
        self.scaler = GradScaler() if device.type == 'cuda' else None
        
        # Loss weights
        self.loss_weights = {
            'emotion': 1.0,
            'jepa': 0.1,
            'prediction_error': 0.1,
            'temporal_consistency': 0.05
        }
        
    def compute_hierarchical_loss(self, outputs: Dict, targets: Dict) -> Dict:
        """Compute multi-component loss"""
        losses = {}
        
        # Emotion classification loss
        emotion_loss = F.cross_entropy(
            outputs['emotion_logits'], 
            targets['label'],
            label_smoothing=0.1
        )
        losses['emotion'] = emotion_loss
        
        # Prediction error (active inference)
        if 'prediction_error_l1' in outputs:
            pred_error = (outputs['prediction_error_l1'].pow(2).mean() + 
                         outputs['prediction_error_l2'].pow(2).mean())
            losses['prediction_error'] = pred_error
            
        # Pattern loss if available
        if 'pattern_logits' in outputs and 'pattern' in targets:
            pattern_loss = F.cross_entropy(outputs['pattern_logits'], targets['pattern'])
            losses['pattern'] = pattern_loss
            
        # Mood regression if available
        if 'mood_values' in outputs and 'valence' in targets:
            mood_loss = F.mse_loss(
                outputs['mood_values'], 
                torch.stack([targets['valence'], targets['arousal']], dim=1)
            )
            losses['mood'] = mood_loss
            
        # Weighted sum
        total_loss = sum(self.loss_weights.get(k, 1.0) * v 
                        for k, v in losses.items())
        losses['total'] = total_loss
        
        return losses
        
    def train_epoch(self, train_loader, epoch: int) -> Dict:
        """Train for one epoch"""
        self.model.train()
        self.ercs.reset()
        self.cedi.reset()
        
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Prepare targets dict
            targets = {
                'label': labels,
                'valence': batch.get('valence', torch.zeros_like(labels)).to(self.device),
                'arousal': batch.get('arousal', torch.zeros_like(labels)).to(self.device)
            }
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            if self.scaler:
                with autocast():
                    outputs = self.model(images)
                    losses = self.compute_hierarchical_loss(outputs, targets)
                    loss = losses['total']
                    
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                losses = self.compute_hierarchical_loss(outputs, targets)
                loss = losses['total']
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
            # Update JEPA target encoders
            if hasattr(self.model, 'update_target_encoders'):
                self.model.update_target_encoders()
                
            # Update metrics
            self.ercs.update(outputs['emotion_logits'], labels)
            self.cedi.update(outputs['emotion_logits'], labels)
            
            # Calculate accuracy
            _, predicted = outputs['emotion_logits'].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'acc': f'{100.*correct/total:.1f}%'
            })
            
            # Log to wandb
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/emotion_loss': losses.get('emotion', 0).item(),
                    'train/prediction_error': losses.get('prediction_error', 0).item(),
                    'train/batch_acc': 100.*predicted.eq(labels).sum().item()/labels.size(0),
                    'train/learning_rate': self.scheduler.get_last_lr()[0]
                })
                
        # Compute epoch metrics
        ercs_metrics = self.ercs.compute()
        cedi_metrics = self.cedi.compute()
        
        epoch_metrics = {
            'loss': total_loss / len(train_loader),
            'accuracy': 100. * correct / total,
            'ercs': ercs_metrics['ercs'],
            'ece': ercs_metrics['ece'],
            'cedi': cedi_metrics['cedi_overall']
        }
        
        return epoch_metrics
        
    def validate(self, val_loader) -> Dict:
        """Validation loop"""
        self.model.eval()
        self.ercs.reset()
        self.cedi.reset()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                targets = {'label': labels}
                
                outputs = self.model(images)
                losses = self.compute_hierarchical_loss(outputs, targets)
                
                self.ercs.update(outputs['emotion_logits'], labels)
                self.cedi.update(outputs['emotion_logits'], labels)
                
                _, predicted = outputs['emotion_logits'].max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                total_loss += losses['total'].item()
                
        # Compute validation metrics
        ercs_metrics = self.ercs.compute()
        cedi_metrics = self.cedi.compute()
        
        val_metrics = {
            'val_loss': total_loss / len(val_loader),
            'val_accuracy': 100. * correct / total,
            'val_ercs': ercs_metrics['ercs'],
            'val_ece': ercs_metrics['ece'],
            'val_mce': ercs_metrics['mce'],
            'val_cedi': cedi_metrics['cedi_overall'],
            'val_cedi_pairs': cedi_metrics['cedi_pairs']
        }
        
        return val_metrics