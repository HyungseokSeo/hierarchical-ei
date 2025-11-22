# hierarchical_ei/models/hierarchical_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from typing import Dict, Tuple, Optional, List
from einops import rearrange, repeat

from hierarchical_ei.models.hierarchical_model import create_hierarchical_model

# Replace the placeholder model creation with:
model = create_hierarchical_model(pretrained=False)
model = model.to(device)

# The model will output a dictionary, so update loss computation:
def compute_loss(outputs, targets):
    """Compute hierarchical loss"""
    emotion_loss = F.cross_entropy(outputs['emotion_logits'], targets['label'])
    
    # Add prediction error regularization
    pred_error_loss = (outputs['prediction_error_l1'].pow(2).mean() + 
                       outputs['prediction_error_l2'].pow(2).mean())
    
    total_loss = emotion_loss + 0.1 * pred_error_loss
    
    return total_loss, {
        'emotion_loss': emotion_loss.item(),
        'pred_error_loss': pred_error_loss.item()
    }

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal information"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class JEPAEncoder(nn.Module):
    """JEPA Encoder for learning representations without reconstruction"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.encoder(x)


class JEPAPredictor(nn.Module):
    """JEPA Predictor for temporal prediction in latent space"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, z_context, mask=None):
        return self.predictor(z_context)

class JEPATargetEncoder(nn.Module):
    """Target encoder with exponential moving average (EMA) updates"""
    def __init__(self, base_encoder: JEPAEncoder, momentum: float = 0.996):
        super().__init__()
        self.encoder = copy.deepcopy(base_encoder)
        self.momentum = momentum
        
        # Freeze target encoder - updated via EMA only
        for param in self.encoder.parameters():
            param.requires_grad = False
            
    @torch.no_grad()
    def update(self, base_encoder):
        """Update target encoder with EMA"""
        for param_base, param_target in zip(base_encoder.parameters(), 
                                           self.encoder.parameters()):
            param_target.data = (self.momentum * param_target.data + 
                                (1 - self.momentum) * param_base.data)
    
    def forward(self, x):
        return self.encoder(x)

class JEPAMasking:
    """Block masking for JEPA training"""
    def __init__(self, mask_ratio: float = 0.75):
        self.mask_ratio = mask_ratio
        
    def __call__(self, x, block_size: int = 4):
        """Create random block masks"""
        # Handle both 2D and 3D inputs
        if len(x.shape) == 2:
            # For 2D input (batch, features), add time dimension
            B, D = x.shape
            mask = torch.rand(B, 1) > self.mask_ratio  # Simple random masking
            return mask.to(x.device)
        
        elif len(x.shape) == 3:
            # Original 3D code
            B, T, D = x.shape
            mask = torch.ones(B, T)
            
            # Number of blocks to mask
            num_masked = int(T * self.mask_ratio / block_size)
            
            for b in range(B):
                # Random block positions
                positions = torch.randperm(T // block_size)[:num_masked]
                for pos in positions:
                    start = pos * block_size
                    end = min(start + block_size, T)
                    mask[b, start:end] = 0
                    
            return mask.bool().to(x.device)
        
        else:
            raise ValueError(f"Expected 2D or 3D input, got shape {x.shape}")
                 
class JEPA(nn.Module):
    """Complete JEPA implementation"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        
        # Context encoder
        self.context_encoder = JEPAEncoder(input_dim, hidden_dim, output_dim)
        
        # Target encoder (EMA updated)
        self.target_encoder = JEPATargetEncoder(
            JEPAEncoder(input_dim, hidden_dim, output_dim)
        )
        
        # Predictor (only applied to context)
        self.predictor = JEPAPredictor(output_dim, hidden_dim, output_dim)
        
        # Masking
        self.masking = JEPAMasking(mask_ratio=0.75)
        
    def forward(self, x):
        # Handle 2D input
        if len(x.shape) == 2:
            # For single images, skip masking or use simple approach
            z_context = self.context_encoder(x)
            z_pred = self.predictor(z_context)
        
            with torch.no_grad():
                z_target = self.target_encoder(x)
        
            return z_pred, z_target
        
        # Create masks
        mask = self.masking(x)
        
        # Encode context (visible patches)
        z_context = self.context_encoder(x[mask])
        
        # Predict target from context
        z_pred = self.predictor(z_context)
        
        # Encode target (masked patches) - no gradient
        with torch.no_grad():
            z_target = self.target_encoder(x[~mask])
            
        return z_pred, z_target
        
    def update_target_encoder(self):
        """Call after each training step"""
        self.target_encoder.update(self.context_encoder)
        
    def jepa_loss(z_pred, z_target):
        """
        JEPA loss: predict target representations from context
        Uses cosine similarity or L2 distance in latent space
        """
        # Normalize representations
        z_pred = F.normalize(z_pred, dim=-1, p=2)
        z_target = F.normalize(z_target, dim=-1, p=2)
    
        # Cosine similarity loss
        loss = 2 - 2 * (z_pred * z_target).sum(dim=-1)
    
        return loss.mean()
   
class Level1MicroExpression(nn.Module):
    """ Level 1: Micro-expression encoder (10-500ms)"""
    def __init__(self, image_size: int = 48, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.jepa = JEPA(128*16, 512, hidden_dim)

	# Lightweight CNN for facial features
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Temporal Convolutional Network with dilations
        self.tcn = nn.ModuleList([
            nn.Conv1d(128*16, hidden_dim, kernel_size=3, dilation=1, padding=1),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2, padding=2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=4, padding=4),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=8, padding=8)
        ])
        
        # JEPA components
        self.jepa_encoder = JEPAEncoder(128*16, 512, hidden_dim)
        self.jepa_predictor = JEPAPredictor(hidden_dim, 512, hidden_dim)
        
        # Audio feature fusion (if available)
        self.audio_encoder = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.fusion_attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        
    def forward(self, images, audio_features=None):
        batch_size = images.shape[0]
        
        # Extract CNN features
        x = self.conv_encoder(images)
        x = x.view(batch_size, -1)
        
        # JEPA encoding
        z = self.jepa_encoder(x)
        z_pred, z_target = self.jepa(x)
        return z_pred
        
        # If we have temporal sequence, apply TCN
        if len(z.shape) == 3:  # (batch, time, features)
            z = z.transpose(1, 2)  # (batch, features, time)
            for tcn_layer in self.tcn:
                z = F.relu(tcn_layer(z))
            z = z.transpose(1, 2)
        
        # Fuse audio if available
        if audio_features is not None:
            audio_z = self.audio_encoder(audio_features)
            z, _ = self.fusion_attention(z.unsqueeze(1), 
                                       audio_z.unsqueeze(1), 
                                       audio_z.unsqueeze(1))
            z = z.squeeze(1)
            
        return z
    
    # Add JEPA loss to training	
    def jepa_loss(z_pred, z_target):
        """
        JEPA loss: predict target representations from context
     	Uses cosine similarity or L2 distance in latent space
        """
        # Normalize representations
        z_pred = F.normalize(z_pred, dim=-1, p=2)
        z_target = F.normalize(z_target, dim=-1, p=2)
    
    	# Cosine similarity loss
        loss = 2 - 2 * (z_pred * z_target).sum(dim=-1)
    
        return loss.mean()
        
class Level2EmotionalState(nn.Module):
    """Level 2: Emotional state transitions (1s-5min)"""
    def __init__(self, input_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Project from Level 1
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Transformer encoder for state modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True  # Fix for the warning
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len=300)
        
        # Causal transformer for prediction
        self.causal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=4
        )
        
        # JEPA components
        self.jepa_encoder = JEPAEncoder(hidden_dim, 1024, hidden_dim)
        self.jepa_predictor = JEPAPredictor(hidden_dim, 1024, hidden_dim)
        
        # Emotion classification head
        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 7)  # 7 basic emotions
        )
        
        # State transition predictor
        self.transition_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 7 * 7)  # Transition matrix
        )
        
    def forward(self, level1_features, context=None):
        # Project Level 1 features
        x = self.input_projection(level1_features)
        
        # Add positional encoding if sequence
        if len(x.shape) == 3:
            x = self.positional_encoding(x)
        else:
            x = x.unsqueeze(1)
        
        # Encode current state
        encoded = self.transformer_encoder(x)
        
        # JEPA encoding
        z = self.jepa_encoder(encoded.mean(dim=1) if len(encoded.shape) == 3 else encoded)
        
        # Predict next state with causal mask
        if len(x.shape) == 3:
            seq_len = x.shape[1]
            causal_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
            predicted = self.causal_transformer(x, mask=causal_mask.to(x.device))
        else:
            predicted = x
            
        # Emotion classification
        emotion_logits = self.emotion_classifier(z)
        
        return z, emotion_logits, predicted


class Level3AffectivePattern(nn.Module):
    """Level 3: Long-term affective patterns (5min-days)"""
    def __init__(self, input_dim: int = 512, hidden_dim: int = 768):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Project from Level 2
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Hierarchical transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=12,
            dim_feedforward=3072,
            dropout=0.1,
            batch_first=True
        )
        self.hierarchical_transformer = nn.TransformerEncoder(encoder_layer, num_layers=8)
        
        # Memory-augmented component
        self.memory_slots = 100
        self.memory_dim = hidden_dim
        self.memory_bank = nn.Parameter(torch.randn(self.memory_slots, self.memory_dim))
        
        self.memory_attention = nn.MultiheadAttention(
            hidden_dim, 
            num_heads=8,
            batch_first=True
        )
        
        # JEPA components
        self.jepa_encoder = JEPAEncoder(hidden_dim, 1536, hidden_dim)
        self.jepa_predictor = JEPAPredictor(hidden_dim, 1536, hidden_dim)
        
        # Pattern classifier
        self.pattern_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 8)  # 8 affective patterns
        )
        
        # Mood predictor (valence-arousal)
        self.mood_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # Valence and arousal
        )
        
    def forward(self, level2_features):
        # Project Level 2 features
        x = self.input_projection(level2_features)
        
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        # Apply hierarchical transformer
        encoded = self.hierarchical_transformer(x)
        
        # Memory attention
        memory_expanded = self.memory_bank.unsqueeze(0).expand(x.shape[0], -1, -1)
        attended_memory, _ = self.memory_attention(encoded, memory_expanded, memory_expanded)
        
        # Combine with encoded features
        combined = encoded + attended_memory
        
        # JEPA encoding
        z = self.jepa_encoder(combined.mean(dim=1))
        
        # Pattern classification
        pattern_logits = self.pattern_classifier(z)
        
        # Mood prediction
        mood_values = self.mood_predictor(z)
        
        return z, pattern_logits, mood_values


class HierarchicalEmotionalIntelligence(nn.Module):
    """Complete hierarchical model with active inference"""
    def __init__(self, config=None):
        super().__init__()
        
        # Three levels
        self.level1 = Level1MicroExpression(image_size=48, hidden_dim=256)
        self.level2 = Level2EmotionalState(input_dim=256, hidden_dim=512)
        self.level3 = Level3AffectivePattern(input_dim=512, hidden_dim=768)
        
        # Precision weighting for active inference
        self.precision_weights = nn.ParameterDict({
            'level1': nn.Parameter(torch.ones(1)),
            'level2': nn.Parameter(torch.ones(1)),
            'level3': nn.Parameter(torch.ones(1))
        })
        
        # Top-down prediction pathways
        self.topdown_3to2 = nn.Linear(768, 512)
        self.topdown_2to1 = nn.Linear(512, 256)
        
    def compute_free_energy(self, predictions, targets, level):
        """Compute free energy for active inference"""
        # Complexity term: KL divergence from prior
        complexity = 0.0
        
        # Accuracy term: prediction error
        accuracy = F.mse_loss(predictions, targets)
        
        # Weight by precision
        precision = F.softplus(self.precision_weights[f'level{level}'])
        
        free_energy = precision * accuracy + (1 - precision) * complexity
        return free_energy
        
    def forward(self, images, audio_features=None, return_all_levels=False):
        """
        Forward pass through hierarchy
        
        Args:
            images: Input images (batch, 3, H, W)
            audio_features: Optional audio features
            return_all_levels: Whether to return features from all levels
        """
        # Level 1: Micro-expressions
        z1 = self.level1(images, audio_features)
        
        # Level 2: Emotional states  
        z2, emotion_logits, state_pred = self.level2(z1)
        
        # Level 3: Affective patterns
        z3, pattern_logits, mood_values = self.level3(z2)
        
        # Top-down predictions
        z3_to_z2 = self.topdown_3to2(z3)
        z2_to_z1 = self.topdown_2to1(z2)
        
        # Compute prediction errors (bottom-up signals)
        if len(z2.shape) == 2:
            error_2 = z2 - z3_to_z2
            error_1 = z1 - z2_to_z1
        else:
            error_2 = z2.mean(dim=1) - z3_to_z2
            error_1 = z1.mean(dim=1) - z2_to_z1
        
        # Primary output is emotion classification
        output = {
            'emotion_logits': emotion_logits,
            'pattern_logits': pattern_logits,
            'mood_values': mood_values,
            'prediction_error_l1': error_1,
            'prediction_error_l2': error_2
        }
        
        if return_all_levels:
            output.update({
                'level1_features': z1,
                'level2_features': z2,
                'level3_features': z3
            })
            
        return output


# Create the model
def create_hierarchical_model(pretrained=False):
    """Factory function to create the model"""
    model = HierarchicalEmotionalIntelligence()
    
    if pretrained:
        # Load pretrained weights if available
        pass
        
    return model 