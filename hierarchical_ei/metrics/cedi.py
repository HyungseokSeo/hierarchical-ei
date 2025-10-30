"""
Contextual Emotional Dynamics Index (CEDI)
Measures the model's ability to capture emotional transitions and context
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class CEDI:
    """
    Contextual Emotional Dynamics Index
    
    Evaluates the model's understanding of emotional dynamics
    including valence/arousal trajectories and contextual adaptation.
    """
    
    def __init__(
        self,
        alpha: float = 0.5,  # Weight for valence vs arousal
        context_window: int = 10
    ):
        self.alpha = alpha
        self.context_window = context_window
        
    def compute_emotional_derivatives(
        self,
        free_energy: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute valence (dF/dt) and arousal (d²F/dt²) from free energy
        
        Args:
            free_energy: Free energy trajectory [batch, time]
            
        Returns:
            valence: First derivative (dF/dt)
            arousal: Second derivative (d²F/dt²)
        """
        batch_size, seq_len = free_energy.shape
        
        if seq_len < 3:
            # Not enough points for second derivative
            valence = torch.zeros_like(free_energy)
            arousal = torch.zeros_like(free_energy)
            return valence, arousal
            
        # Compute first derivative (valence)
        # Using central differences
        valence = torch.zeros_like(free_energy)
        valence[:, 1:-1] = (free_energy[:, 2:] - free_energy[:, :-2]) / 2
        valence[:, 0] = free_energy[:, 1] - free_energy[:, 0]
        valence[:, -1] = free_energy[:, -1] - free_energy[:, -2]
        
        # Compute second derivative (arousal)
        arousal = torch.zeros_like(free_energy)
        arousal[:, 1:-1] = (valence[:, 2:] - valence[:, :-2]) / 2
        arousal[:, 0] = valence[:, 1] - valence[:, 0]
        arousal[:, -1] = valence[:, -1] - valence[:, -2]
        
        # Take absolute value for arousal (magnitude of change)
        arousal = torch.abs(arousal)
        
        return valence, arousal
        
    def compute_context_adaptation(
        self,
        predictions: torch.Tensor,
        context_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Measure how well predictions adapt to context changes
        """
        batch_size, seq_len, _ = predictions.shape
        
        adaptation_scores = []
        
        for i in range(self.context_window, seq_len):
            # Get context window
            context_window = context_features[:, i-self.context_window:i, :]
            current_pred = predictions[:, i, :]
            
            # Compute attention weights based on context
            context_mean = context_window.mean(dim=1)
            attention = F.softmax(
                torch.matmul(current_pred.unsqueeze(1), context_mean.unsqueeze(-1)).squeeze(-1),
                dim=-1
            )
            
            # Measure adaptation as entropy of attention
            # Higher entropy = better adaptation to diverse context
            entropy = -torch.sum(attention * torch.log(attention + 1e-8), dim=-1)
            adaptation_scores.append(entropy)
            
        if adaptation_scores:
            return torch.stack(adaptation_scores, dim=1).mean(dim=1)
        else:
            return torch.zeros(batch_size, device=predictions.device)
    
    def __call__(
        self,
        model_outputs: Dict[str, torch.Tensor],
        free_energy: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute CEDI score
        
        Args:
            model_outputs: Dictionary containing:
                - 'valence': Predicted valence trajectory
                - 'arousal': Predicted arousal trajectory
                - 'predictions': Emotion predictions
                - 'context_features': Contextual representations
            free_energy: Optional free energy trajectory for computing derivatives
            
        Returns:
            Dictionary containing:
                - 'cedi_score': Overall CEDI score
                - 'valence_accuracy': Valence prediction accuracy
                - 'arousal_accuracy': Arousal prediction accuracy
                - 'context_adaptation': Context adaptation score
        """
        results = {}
        
        # If free energy provided, compute true dynamics
        if free_energy is not None:
            true_valence, true_arousal = self.compute_emotional_derivatives(free_energy)
            
            # Compare with predicted dynamics
            if 'valence' in model_outputs:
                valence_error = F.mse_loss(
                    model_outputs['valence'], 
                    true_valence,
                    reduction='none'
                ).mean(dim=-1)
                results['valence_accuracy'] = 1.0 - torch.tanh(valence_error)
                
            if 'arousal' in model_outputs:
                arousal_error = F.mse_loss(
                    model_outputs['arousal'],
                    true_arousal,
                    reduction='none'
                ).mean(dim=-1)
                results['arousal_accuracy'] = 1.0 - torch.tanh(arousal_error)
        
        # Compute context adaptation
        if 'predictions' in model_outputs and 'context_features' in model_outputs:
            results['context_adaptation'] = self.compute_context_adaptation(
                model_outputs['predictions'],
                model_outputs['context_features']
            )
        
        # Combine into overall CEDI score
        cedi_components = []
        
        if 'valence_accuracy' in results:
            cedi_components.append(results['valence_accuracy'] * self.alpha)
        if 'arousal_accuracy' in results:
            cedi_components.append(results['arousal_accuracy'] * (1 - self.alpha))
        if 'context_adaptation' in results:
            cedi_components.append(results['context_adaptation'] * 0.3)  # Additional weight
            
        if cedi_components:
            results['cedi_score'] = torch.stack(cedi_components).mean(dim=0)
        else:
            results['cedi_score'] = torch.zeros(1)
            
        return results