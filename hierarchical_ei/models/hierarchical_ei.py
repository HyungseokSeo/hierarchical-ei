"""
Hierarchical Emotional Intelligence Model
A Unified JEPA-Active Inference Framework

Author: Hyungseok Seo
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import math


@dataclass
class HierarchicalConfig:
    """Configuration for Hierarchical EI model"""
    # Level 1: Micro-expressions
    level1_dim: int = 256
    level1_encoder_dims: List[int] = (3, 64, 128, 256)
    level1_window_size: int = 15  # frames
    
    # Level 2: Emotional states  
    level2_dim: int = 512
    level2_heads: int = 8
    level2_layers: int = 6
    level2_window_size: int = 150  # ~5 seconds at 30fps
    
    # Level 3: Affective patterns
    level3_dim: int = 1024
    level3_memory_size: int = 100
    level3_memory_dim: int = 512
    
    # Active inference
    free_energy_beta: float = 0.1
    precision_lr: float = 0.01
    expected_free_energy_horizon: int = 10
    
    # Training
    learning_rate: float = 1e-4
    batch_size: int = 32
    warmup_steps: int = 10000


class JEPAEncoder(nn.Module):
    """JEPA Encoder for hierarchical level"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class JEPAPredictor(nn.Module):
    """JEPA Predictor with attention mechanism"""
    
    def __init__(self, dim: int, window_size: int):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        
        # Attention components
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        
    def forward(self, z_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_history: [batch, window_size, dim]
        Returns:
            z_pred: [batch, dim]
        """
        B, W, D = z_history.shape
        
        # Self-attention over history
        Q = self.q_proj(z_history[:, -1:, :])  # [B, 1, D]
        K = self.k_proj(z_history)  # [B, W, D]
        V = self.v_proj(z_history)  # [B, W, D]
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        context = torch.matmul(attn_weights, V).squeeze(1)  # [B, D]
        
        # Predict next state
        z_pred = self.predictor(context)
        
        return z_pred


class Level1MicroExpression(nn.Module):
    """Level 1: Micro-expression dynamics (10-500ms)"""
    
    def __init__(self, config: HierarchicalConfig):
        super().__init__()
        self.config = config
        
        # Multi-modal encoders
        self.visual_encoder = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.visual_layers = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.audio_encoder = nn.Conv1d(1, 64, kernel_size=3, stride=2, padding=1)
        self.audio_layers = nn.Sequential(
            nn.Conv1d(64, 128, 3, 2, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 3, 2, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(512, config.level1_dim),
            nn.LayerNorm(config.level1_dim),
            nn.GELU()
        )
        
        # JEPA components
        self.jepa_encoder = JEPAEncoder(config.level1_dim, config.level1_dim * 2, config.level1_dim)
        self.jepa_predictor = JEPAPredictor(config.level1_dim, config.level1_window_size)
        
        # Precision estimation
        self.precision_net = nn.Sequential(
            nn.Linear(config.level1_dim * 2, config.level1_dim),
            nn.ReLU(),
            nn.Linear(config.level1_dim, config.level1_dim),
            nn.Softplus()
        )
        
    def forward(self, visual: torch.Tensor, audio: torch.Tensor, 
                return_predictions: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            visual: [batch, time, 3, H, W]
            audio: [batch, time, audio_dim]
        """
        B, T = visual.shape[:2]
        
        # Encode each modality
        visual_feats = []
        audio_feats = []
        
        for t in range(T):
            # Visual encoding
            v = self.visual_encoder(visual[:, t])
            v = self.visual_layers(v).squeeze(-1).squeeze(-1)
            visual_feats.append(v)
            
            # Audio encoding
            a = self.audio_encoder(audio[:, t].unsqueeze(1))
            a = self.audio_layers(a).squeeze(-1)
            audio_feats.append(a)
        
        visual_feats = torch.stack(visual_feats, dim=1)  # [B, T, 256]
        audio_feats = torch.stack(audio_feats, dim=1)    # [B, T, 256]
        
        # Multi-modal fusion
        fused = torch.cat([visual_feats, audio_feats], dim=-1)
        z1 = self.fusion(fused)  # [B, T, level1_dim]
        
        # JEPA encoding
        z1_encoded = []
        for t in range(T):
            z1_encoded.append(self.jepa_encoder(z1[:, t]))
        z1_encoded = torch.stack(z1_encoded, dim=1)
        
        outputs = {'z1': z1_encoded}
        
        # Predictions for JEPA loss
        if return_predictions and T > self.config.level1_window_size:
            predictions = []
            targets = []
            
            for t in range(self.config.level1_window_size, T-1):
                z_history = z1_encoded[:, t-self.config.level1_window_size:t]
                z_pred = self.jepa_predictor(z_history)
                predictions.append(z_pred)
                targets.append(z1_encoded[:, t+1])
            
            outputs['predictions'] = torch.stack(predictions, dim=1)
            outputs['targets'] = torch.stack(targets, dim=1)
        
        # Estimate precision
        if T > 1:
            z_concat = torch.cat([z1_encoded[:, :-1], z1_encoded[:, 1:]], dim=-1)
            precision = self.precision_net(z_concat)
            outputs['precision1'] = precision
        
        return outputs


class Level2EmotionalStates(nn.Module):
    """Level 2: Emotional state transitions (1s-5min)"""
    
    def __init__(self, config: HierarchicalConfig):
        super().__init__()
        self.config = config
        
        # Transformer encoder for temporal modeling
        self.input_projection = nn.Linear(config.level1_dim, config.level2_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.level2_dim,
            nhead=config.level2_heads,
            dim_feedforward=config.level2_dim * 4,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.level2_layers)
        
        # Context integration
        self.context_dim = 128
        self.context_encoder = nn.Sequential(
            nn.Linear(self.context_dim, config.level2_dim),
            nn.LayerNorm(config.level2_dim),
            nn.GELU()
        )
        
        # JEPA components
        self.jepa_encoder = JEPAEncoder(config.level2_dim, config.level2_dim * 2, config.level2_dim)
        self.jepa_predictor = JEPAPredictor(config.level2_dim, config.level2_window_size // config.level1_window_size)
        
        # Emotion classification head
        self.num_emotions = 8  # 7 basic + neutral
        self.emotion_head = nn.Linear(config.level2_dim, self.num_emotions)
        
        # Gaussian parameters for transitions
        self.transition_mean = nn.Linear(config.level2_dim, config.level2_dim)
        self.transition_logvar = nn.Linear(config.level2_dim, config.level2_dim)
        
    def forward(self, z1: torch.Tensor, context: Optional[torch.Tensor] = None,
                return_predictions: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            z1: [batch, time, level1_dim] - Level 1 embeddings
            context: [batch, context_dim] - Optional context
        """
        B, T, _ = z1.shape
        
        # Project to Level 2 dimension
        z2 = self.input_projection(z1)
        
        # Add context if available
        if context is not None:
            context_embed = self.context_encoder(context).unsqueeze(1)
            z2 = z2 + context_embed
        
        # Transformer encoding with causal mask
        mask = torch.triu(torch.ones(T, T) * float('-inf'), diagonal=1).to(z1.device)
        z2_transformed = self.transformer(z2.transpose(0, 1), mask=mask).transpose(0, 1)
        
        # JEPA encoding
        z2_encoded = []
        for t in range(T):
            z2_encoded.append(self.jepa_encoder(z2_transformed[:, t]))
        z2_encoded = torch.stack(z2_encoded, dim=1)
        
        outputs = {'z2': z2_encoded}
        
        # Emotion classification
        emotion_logits = self.emotion_head(z2_encoded)
        outputs['emotion_logits'] = emotion_logits
        outputs['emotion_probs'] = F.softmax(emotion_logits, dim=-1)
        
        # Transition modeling
        if T > 1:
            mu = self.transition_mean(z2_encoded[:, :-1])
            logvar = self.transition_logvar(z2_encoded[:, :-1])
            outputs['transition_mu'] = mu
            outputs['transition_logvar'] = logvar
            outputs['transition_dist'] = Normal(mu, torch.exp(0.5 * logvar))
        
        # JEPA predictions
        if return_predictions and T > self.config.level2_window_size // self.config.level1_window_size:
            window = self.config.level2_window_size // self.config.level1_window_size
            predictions = []
            targets = []
            
            for t in range(window, T-1):
                z_history = z2_encoded[:, t-window:t]
                z_pred = self.jepa_predictor(z_history)
                predictions.append(z_pred)
                targets.append(z2_encoded[:, t+1])
            
            if predictions:
                outputs['predictions'] = torch.stack(predictions, dim=1)
                outputs['targets'] = torch.stack(targets, dim=1)
        
        return outputs


class Level3AffectivePatterns(nn.Module):
    """Level 3: Long-term affective patterns (5min-days)"""
    
    def __init__(self, config: HierarchicalConfig):
        super().__init__()
        self.config = config
        
        # Memory-augmented architecture
        self.memory_size = config.level3_memory_size
        self.memory_dim = config.level3_memory_dim
        
        # Initialize memory
        self.register_buffer('memory', torch.randn(1, self.memory_size, self.memory_dim))
        
        # Input projection
        self.input_projection = nn.Linear(config.level2_dim, config.level3_dim)
        
        # Memory attention
        self.memory_query = nn.Linear(config.level3_dim, self.memory_dim)
        self.memory_key = nn.Linear(self.memory_dim, self.memory_dim)
        self.memory_value = nn.Linear(self.memory_dim, self.memory_dim)
        
        # Pattern encoder
        self.pattern_encoder = nn.LSTM(
            input_size=config.level3_dim + self.memory_dim,
            hidden_size=config.level3_dim,
            num_layers=2,
            batch_first=True
        )
        
        # JEPA components
        self.jepa_encoder = JEPAEncoder(config.level3_dim, config.level3_dim * 2, config.level3_dim)
        self.jepa_predictor = nn.Sequential(
            nn.Linear(config.level3_dim, config.level3_dim * 2),
            nn.GELU(),
            nn.Linear(config.level3_dim * 2, config.level3_dim)
        )
        
        # Affective pattern heads
        self.mood_head = nn.Linear(config.level3_dim, 3)  # Positive, Neutral, Negative
        self.personality_head = nn.Linear(config.level3_dim, 5)  # Big Five traits
        
    def forward(self, z2: torch.Tensor, update_memory: bool = True) -> Dict[str, torch.Tensor]:
        """
        Args:
            z2: [batch, time, level2_dim] - Level 2 embeddings
        """
        B, T, _ = z2.shape
        
        # Project to Level 3
        z3 = self.input_projection(z2)
        
        # Expand memory for batch
        memory = self.memory.expand(B, -1, -1)
        
        # Process with memory attention
        z3_with_memory = []
        
        for t in range(T):
            # Query memory
            query = self.memory_query(z3[:, t:t+1])  # [B, 1, memory_dim]
            keys = self.memory_key(memory)  # [B, memory_size, memory_dim]
            values = self.memory_value(memory)  # [B, memory_size, memory_dim]
            
            # Attention over memory
            attn_scores = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(self.memory_dim)
            attn_weights = F.softmax(attn_scores, dim=-1)
            memory_read = torch.matmul(attn_weights, values).squeeze(1)  # [B, memory_dim]
            
            # Concatenate with current state
            z3_t = torch.cat([z3[:, t], memory_read], dim=-1)
            z3_with_memory.append(z3_t)
            
            # Update memory (moving average)
            if update_memory and self.training:
                memory = 0.95 * memory + 0.05 * z3[:, t:t+1].unsqueeze(1)
        
        z3_with_memory = torch.stack(z3_with_memory, dim=1)
        
        # LSTM processing
        z3_lstm, (h_n, c_n) = self.pattern_encoder(z3_with_memory)
        
        # JEPA encoding
        z3_encoded = []
        for t in range(T):
            z3_encoded.append(self.jepa_encoder(z3_lstm[:, t]))
        z3_encoded = torch.stack(z3_encoded, dim=1)
        
        outputs = {'z3': z3_encoded}
        
        # Pattern predictions
        outputs['mood_logits'] = self.mood_head(z3_encoded)
        outputs['personality_scores'] = torch.sigmoid(self.personality_head(z3_encoded))
        
        # Long-term prediction
        z3_final = z3_encoded[:, -1]
        z3_pred = self.jepa_predictor(z3_final)
        outputs['z3_prediction'] = z3_pred
        
        return outputs


class ActiveInferenceModule(nn.Module):
    """Active Inference for hierarchical control"""
    
    def __init__(self, config: HierarchicalConfig):
        super().__init__()
        self.config = config
        
        # Generative models for each level
        self.g1 = nn.Sequential(
            nn.Linear(config.level2_dim, config.level1_dim * 2),
            nn.GELU(),
            nn.Linear(config.level1_dim * 2, config.level1_dim)
        )
        
        self.g2 = nn.Sequential(
            nn.Linear(config.level3_dim, config.level2_dim * 2),
            nn.GELU(),
            nn.Linear(config.level2_dim * 2, config.level2_dim)
        )
        
        # Precision matrices (learnable)
        self.log_pi1 = nn.Parameter(torch.zeros(config.level1_dim))
        self.log_pi2 = nn.Parameter(torch.zeros(config.level2_dim))
        self.log_pi3 = nn.Parameter(torch.zeros(config.level3_dim))
        
        # Action selection network
        self.action_dim = 32
        self.action_net = nn.Sequential(
            nn.Linear(config.level1_dim + config.level2_dim + config.level3_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim)
        )
        
    def compute_free_energy(self, z: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute variational free energy for each level"""
        
        free_energy = {}
        
        # Level 1 free energy
        if 'z1' in z and 'z2' in z:
            z1_pred = self.g1(z['z2'])
            pi1 = torch.exp(self.log_pi1)
            error1 = pi1 * (z['z1'] - z1_pred)
            free_energy['F1'] = 0.5 * torch.sum(error1 ** 2, dim=-1)
        
        # Level 2 free energy  
        if 'z2' in z and 'z3' in z:
            z2_pred = self.g2(z['z3'])
            pi2 = torch.exp(self.log_pi2)
            error2 = pi2 * (z['z2'] - z2_pred)
            free_energy['F2'] = 0.5 * torch.sum(error2 ** 2, dim=-1)
        
        # Level 3 free energy (prior)
        if 'z3' in z:
            pi3 = torch.exp(self.log_pi3)
            free_energy['F3'] = 0.5 * torch.sum((pi3 * z['z3']) ** 2, dim=-1)
        
        # Total free energy
        free_energy['F_total'] = sum(free_energy.values())
        
        return free_energy
    
    def select_action(self, z: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Select action to minimize expected free energy"""
        
        # Concatenate states from all levels
        states = []
        if 'z1' in z:
            states.append(z['z1'])
        if 'z2' in z:
            states.append(z['z2'])
        if 'z3' in z:
            states.append(z['z3'])
        
        if not states:
            raise ValueError("No states provided for action selection")
        
        state_concat = torch.cat(states, dim=-1)
        
        # Get action logits
        action_logits = self.action_net(state_concat)
        
        # Sample action (with temperature for exploration)
        temperature = 1.0
        action_probs = F.softmax(action_logits / temperature, dim=-1)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        
        return action, action_logits


class HierarchicalEmotionalIntelligence(nn.Module):
    """Complete Hierarchical EI Model"""
    
    def __init__(self, config: HierarchicalConfig):
        super().__init__()
        self.config = config
        
        # Hierarchical levels
        self.level1 = Level1MicroExpression(config)
        self.level2 = Level2EmotionalStates(config)
        self.level3 = Level3AffectivePatterns(config)
        
        # Active inference
        self.active_inference = ActiveInferenceModule(config)
        
        # Emotion regulation head
        self.regulation_net = nn.Sequential(
            nn.Linear(config.level1_dim + config.level2_dim + config.level3_dim, 512),
            nn.ReLU(),
            nn.Linear(512, config.level2_dim)
        )
        
    def forward(self, visual: torch.Tensor, audio: torch.Tensor, 
                context: Optional[torch.Tensor] = None,
                return_all: bool = True) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass through hierarchy
        
        Args:
            visual: [batch, time, 3, H, W]
            audio: [batch, time, audio_dim]
            context: [batch, context_dim]
            return_all: Return all intermediate outputs
        """
        
        outputs = {}
        
        # Level 1: Micro-expressions
        level1_out = self.level1(visual, audio, return_predictions=return_all)
        outputs.update({f'level1_{k}': v for k, v in level1_out.items()})
        
        # Level 2: Emotional states
        level2_out = self.level2(level1_out['z1'], context, return_predictions=return_all)
        outputs.update({f'level2_{k}': v for k, v in level2_out.items()})
        
        # Level 3: Affective patterns
        level3_out = self.level3(level2_out['z2'])
        outputs.update({f'level3_{k}': v for k, v in level3_out.items()})
        
        # Active inference
        z_dict = {
            'z1': level1_out['z1'][:, -1],
            'z2': level2_out['z2'][:, -1],
            'z3': level3_out['z3'][:, -1]
        }
        
        free_energy = self.active_inference.compute_free_energy(z_dict)
        outputs.update({f'ai_{k}': v for k, v in free_energy.items()})
        
        action, action_logits = self.active_inference.select_action(z_dict)
        outputs['action'] = action
        outputs['action_logits'] = action_logits
        
        # Emotion regulation
        state_concat = torch.cat([z_dict['z1'], z_dict['z2'], z_dict['z3']], dim=-1)
        regulation_signal = self.regulation_net(state_concat)
        outputs['regulation'] = regulation_signal
        
        return outputs
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], 
                     targets: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """Compute hierarchical losses"""
        
        losses = {}
        
        # JEPA losses for each level
        if 'level1_predictions' in outputs and 'level1_targets' in outputs:
            jepa_loss1 = F.mse_loss(outputs['level1_predictions'], 
                                    outputs['level1_targets'].detach())
            losses['jepa1'] = jepa_loss1
        
        if 'level2_predictions' in outputs and 'level2_targets' in outputs:
            jepa_loss2 = F.mse_loss(outputs['level2_predictions'], 
                                    outputs['level2_targets'].detach())
            losses['jepa2'] = jepa_loss2
        
        # Emotion classification loss
        if targets and 'emotions' in targets and 'level2_emotion_logits' in outputs:
            emotion_loss = F.cross_entropy(
                outputs['level2_emotion_logits'].reshape(-1, 8),
                targets['emotions'].reshape(-1)
            )
            losses['emotion'] = emotion_loss
        
        # Active inference loss (free energy)
        if 'ai_F_total' in outputs:
            losses['free_energy'] = outputs['ai_F_total'].mean()
        
        # Regularization losses
        if 'level1_precision1' in outputs:
            # Encourage reasonable precision values
            precision_reg = torch.mean((outputs['level1_precision1'] - 1.0) ** 2)
            losses['precision_reg'] = 0.01 * precision_reg
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses
    
    @torch.no_grad()
    def generate_emotional_trajectory(self, initial_state: torch.Tensor, 
                                      steps: int = 100) -> List[torch.Tensor]:
        """Generate future emotional trajectory"""
        
        trajectory = []
        current_state = initial_state
        
        for _ in range(steps):
            # Use Level 2 transition model
            mu = self.level2.transition_mean(current_state)
            logvar = self.level2.transition_logvar(current_state)
            dist = Normal(mu, torch.exp(0.5 * logvar))
            
            # Sample next state
            next_state = dist.sample()
            trajectory.append(next_state)
            
            current_state = next_state
        
        return trajectory


# Example usage
if __name__ == "__main__":
    # Initialize model
    config = HierarchicalConfig()
    model = HierarchicalEmotionalIntelligence(config)
    
    # Dummy inputs
    batch_size = 2
    time_steps = 300  # 10 seconds at 30fps
    visual = torch.randn(batch_size, time_steps, 3, 64, 64)
    audio = torch.randn(batch_size, time_steps, 128)
    context = torch.randn(batch_size, 128)
    
    # Forward pass
    outputs = model(visual, audio, context)
    
    # Print output shapes
    print("Model outputs:")
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
    
    # Compute losses
    targets = {'emotions': torch.randint(0, 8, (batch_size, time_steps))}
    losses = model.compute_loss(outputs, targets)
    
    print("\nLosses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
