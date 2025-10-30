"""
Emotional Response Coherence Score (ERCS)
Measures consistency of emotional predictions across hierarchical levels
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
import numpy as np


class ERCS:
    """
    Emotional Response Coherence Score
    
    Evaluates the consistency between different hierarchical levels
    of emotional understanding.
    """
    
    def __init__(
        self,
        level_weights: Optional[List[float]] = None,
        temperature: float = 1.0
    ):
        self.level_weights = level_weights or [0.2, 0.3, 0.5]  # Level 1, 2, 3
        self.temperature = temperature
        
    def compute_level_coherence(
        self,
        level1_features: torch.Tensor,
        level2_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute coherence between two hierarchical levels
        """
        # Normalize features
        level1_norm = F.normalize(level1_features, p=2, dim=-1)
        level2_norm = F.normalize(level2_features, p=2, dim=-1)
        
        # Compute cosine similarity
        if level1_norm.shape != level2_norm.shape:
            # Align dimensions via projection if needed
            proj_dim = min(level1_norm.shape[-1], level2_norm.shape[-1])
            level1_norm = level1_norm[..., :proj_dim]
            level2_norm = level2_norm[..., :proj_dim]
            
        coherence = torch.sum(level1_norm * level2_norm, dim=-1)
        return coherence
    
    def compute_temporal_consistency(
        self,
        predictions: torch.Tensor,
        window_size: int = 5
    ) -> torch.Tensor:
        """
        Measure temporal consistency of predictions
        """
        batch_size, seq_len, num_classes = predictions.shape
        
        if seq_len <= window_size:
            return torch.ones(batch_size, device=predictions.device)
            
        consistency_scores = []
        
        for i in range(seq_len - window_size):
            window = predictions[:, i:i+window_size, :]
            # Compute variance within window
            window_std = torch.std(window, dim=1)
            # Average variance across classes
            consistency = 1.0 - torch.mean(window_std, dim=-1)
            consistency_scores.append(consistency)
            
        return torch.stack(consistency_scores, dim=1).mean(dim=1)
    
    def __call__(
        self,
        hierarchical_outputs: Dict[str, torch.Tensor],
        targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute ERCS score
        
        Args:
            hierarchical_outputs: Dictionary containing outputs from each level
                - 'level1': Micro-expression features
                - 'level2': Emotional state features  
                - 'level3': Affective pattern features
                - 'predictions': Final emotion predictions
            targets: Ground truth labels (optional)
            
        Returns:
            Dictionary containing:
                - 'ercs_score': Overall ERCS score
                - 'level_coherence': Coherence between levels
                - 'temporal_consistency': Temporal consistency score
        """
        results = {}
        
        # Compute coherence between levels
        coherence_scores = []
        
        if 'level1' in hierarchical_outputs and 'level2' in hierarchical_outputs:
            coh_12 = self.compute_level_coherence(
                hierarchical_outputs['level1'],
                hierarchical_outputs['level2']
            )
            coherence_scores.append(coh_12 * self.level_weights[0])
            
        if 'level2' in hierarchical_outputs and 'level3' in hierarchical_outputs:
            coh_23 = self.compute_level_coherence(
                hierarchical_outputs['level2'],
                hierarchical_outputs['level3']
            )
            coherence_scores.append(coh_23 * self.level_weights[1])
            
        # Compute temporal consistency
        if 'predictions' in hierarchical_outputs:
            temporal_consistency = self.compute_temporal_consistency(
                hierarchical_outputs['predictions']
            )
            results['temporal_consistency'] = temporal_consistency
            
        # Combine scores
        if coherence_scores:
            results['level_coherence'] = torch.stack(coherence_scores).mean(dim=0)
            
        # Overall ERCS score
        ercs_components = []
        if 'level_coherence' in results:
            ercs_components.append(results['level_coherence'])
        if 'temporal_consistency' in results:
            ercs_components.append(results['temporal_consistency'] * self.level_weights[2])
            
        if ercs_components:
            results['ercs_score'] = torch.stack(ercs_components).mean(dim=0)
        else:
            results['ercs_score'] = torch.zeros(1)
            
        return results