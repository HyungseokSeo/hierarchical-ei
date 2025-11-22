import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Dict, List, Tuple

class ERCS(object):
    """Emotion Recognition Confidence Score
    Measures the calibration quality of confidence estimates
    """
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.reset()
        
    def reset(self):
        self.confidences = []
        self.accuracies = []
        self.bin_totals = []
        
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """Update ERCS with batch predictions"""
        probabilities = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(probabilities, 1)
        
        accuracies = predictions.eq(targets).float()
        
        self.confidences.extend(confidences.cpu().numpy())
        self.accuracies.extend(accuracies.cpu().numpy())
        
    def compute(self) -> Dict[str, float]:
        """Compute ERCS and Expected Calibration Error"""
        confidences = np.array(self.confidences)
        accuracies = np.array(self.accuracies)
        
        # Binning
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        mce = 0.0
        ercs_score = 0.0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                # Expected Calibration Error
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                # Maximum Calibration Error
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
                
                # ERCS component
                ercs_score += (1 - np.abs(avg_confidence_in_bin - accuracy_in_bin)) * prop_in_bin
                
        return {
            'ercs': ercs_score,
            'ece': ece,
            'mce': mce
        }


class CEDI(object):
    """Cross-Emotion Discrimination Index
    Measures discrimination ability between similar emotions
    """
    
    # Define emotion similarity groups
    EMOTION_GROUPS = {
        'negative_high': [0, 1, 2],  # angry, disgust, fear
        'positive': [3],              # happy
        'negative_low': [4],          # sad
        'surprise': [5],              # surprise
        'neutral': [6]                # neutral
    }
    
    # Confusion pairs to specifically track
    CONFUSION_PAIRS = [
        (2, 5),  # fear-surprise
        (0, 4),  # angry-sad
        (0, 1),  # angry-disgust
        (4, 6),  # sad-neutral
    ]
    
    def __init__(self, n_classes: int = 7):
        self.n_classes = n_classes
        self.reset()
        
    def reset(self):
        self.predictions = []
        self.targets = []
        
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """Update with batch predictions"""
        _, preds = torch.max(logits, 1)
        self.predictions.extend(preds.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        
    def compute(self) -> Dict[str, float]:
        """Compute CEDI scores"""
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Compute confusion matrix
        cm = confusion_matrix(targets, predictions, labels=range(self.n_classes))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Overall CEDI
        correct = np.diag(cm_normalized)
        cedi_overall = np.mean(correct)
        
        # Pair-specific CEDI
        pair_scores = {}
        for i, j in self.CONFUSION_PAIRS:
            if cm[i, i] + cm[i, j] > 0:
                discrimination_i = cm[i, i] / (cm[i, i] + cm[i, j] + 1e-7)
            else:
                discrimination_i = 0
                
            if cm[j, j] + cm[j, i] > 0:
                discrimination_j = cm[j, j] / (cm[j, j] + cm[j, i] + 1e-7)
            else:
                discrimination_j = 0
                
            pair_cedi = (discrimination_i + discrimination_j) / 2
            pair_scores[f'{i}-{j}'] = pair_cedi
            
        return {
            'cedi_overall': cedi_overall,
            'cedi_pairs': pair_scores,
            'confusion_matrix': cm.tolist()
        }