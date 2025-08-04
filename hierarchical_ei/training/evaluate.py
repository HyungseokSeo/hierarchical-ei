"""
Evaluation script for Hierarchical Emotional Intelligence Model

This script provides comprehensive evaluation including:
- Emotion recognition accuracy
- Temporal prediction performance
- Causal reasoning evaluation
- Emergent behavior analysis
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    mean_squared_error, mean_absolute_error
)
from tqdm import tqdm

from hierarchical_ei_model import HierarchicalEmotionalIntelligence, HierarchicalConfig
from train_hierarchical_ei import EmotionDataset
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


class Evaluator:
    """Comprehensive evaluation for Hierarchical EI Model"""
    
    def __init__(self, model_path: str, config_path: str, device: str = 'cuda'):
        # Load model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.load_model(model_path, config_path)
        
        # Emotion labels
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 
                              'Sad', 'Surprise', 'Neutral', 'Contempt']
        
        # Results storage
        self.results = {
            'emotion_recognition': {},
            'temporal_prediction': {},
            'causal_reasoning': {},
            'emergent_behaviors': {}
        }
        
    def load_model(self, model_path: str, config_path: str):
        """Load trained model"""
        # Load config
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Initialize model
        config = HierarchicalConfig(**config_dict.get('model', {}))
        self.model = HierarchicalEmotionalIntelligence(config)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Loaded model from {model_path}")
        
    @torch.no_grad()
    def evaluate_emotion_recognition(self, data_loader: DataLoader) -> Dict:
        """Evaluate emotion recognition performance"""
        
        all_predictions = []
        all_targets = []
        all_confidences = []
        
        for batch in tqdm(data_loader, desc="Evaluating emotion recognition"):
            visual = batch['visual'].to(self.device)
            audio = batch['audio'].to(self.device)
            context = batch['context'].to(self.device)
            emotions = batch['emotions'].numpy()
            
            # Forward pass
            outputs = self.model(visual, audio, context)
            
            if 'level2_emotion_probs' in outputs:
                probs = outputs['level2_emotion_probs'].cpu().numpy()
                preds = probs.argmax(axis=-1)
                
                # Flatten temporal dimension
                preds_flat = preds.reshape(-1)
                emotions_flat = emotions.reshape(-1)
                probs_flat = probs.reshape(-1, probs.shape[-1])
                
                all_predictions.extend(preds_flat)
                all_targets.extend(emotions_flat)
                all_confidences.extend(probs_flat.max(axis=-1))
        
        # Compute metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        f1_micro = f1_score(all_targets, all_predictions, average='micro')
        f1_macro = f1_score(all_targets, all_predictions, average='macro')
        f1_per_class = f1_score(all_targets, all_predictions, average=None)
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        # Average confidence
        avg_confidence = np.mean(all_confidences)
        
        results = {
            'accuracy': accuracy,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'f1_per_class': {self.emotion_labels[i]: f1 for i, f1 in enumerate(f1_per_class)},
            'confusion_matrix': cm.tolist(),
            'average_confidence': avg_confidence
        }
        
        self.results['emotion_recognition'] = results
        
        # Plot confusion matrix
        self._plot_confusion_matrix(cm, self.emotion_labels[:len(f1_per_class)])
        
        return results
    
    @torch.no_grad()
    def evaluate_temporal_prediction(self, data_loader: DataLoader) -> Dict:
        """Evaluate temporal prediction capabilities"""
        
        mse_scores = []
        mae_scores = []
        prediction_horizons = [1, 5, 10, 20]  # frames
        
        for batch in tqdm(data_loader, desc="Evaluating temporal prediction"):
            visual = batch['visual'].to(self.device)
            audio = batch['audio'].to(self.device)
            context = batch['context'].to(self.device)
            
            # Forward pass
            outputs = self.model(visual, audio, context, return_all=True)
            
            # Evaluate predictions at different levels
            for level in [1, 2]:
                if f'level{level}_predictions' in outputs and f'level{level}_targets' in outputs:
                    preds = outputs[f'level{level}_predictions'].cpu().numpy()
                    targets = outputs[f'level{level}_targets'].cpu().numpy()
                    
                    mse = mean_squared_error(targets.reshape(-1), preds.reshape(-1))
                    mae = mean_absolute_error(targets.reshape(-1), preds.reshape(-1))
                    
                    mse_scores.append(mse)
                    mae_scores.append(mae)
        
        # Generate trajectory predictions
        trajectory_results = self._evaluate_trajectory_generation(data_loader)
        
        results = {
            'mse': np.mean(mse_scores),
            'mae': np.mean(mae_scores),
            'trajectory_generation': trajectory_results
        }
        
        self.results['temporal_prediction'] = results
        return results
    
    def _evaluate_trajectory_generation(self, data_loader: DataLoader, num_steps: int = 50) -> Dict:
        """Evaluate emotional trajectory generation"""
        
        trajectory_errors = []
        
        for batch in data_loader:
            visual = batch['visual'].to(self.device)
            audio = batch['audio'].to(self.device)
            context = batch['context'].to(self.device)
            
            # Get initial state
            outputs = self.model(visual, audio, context)
            if 'level2_z2' in outputs:
                initial_state = outputs['level2_z2'][:, -1]
                
                # Generate trajectory
                trajectory = self.model.generate_emotional_trajectory(initial_state, steps=num_steps)
                
                # Compare with actual future states if available
                # This is a simplified evaluation - in practice, you'd compare with ground truth
                if len(trajectory) > 1:
                    # Compute smoothness of trajectory
                    trajectory_tensor = torch.stack(trajectory)
                    diffs = torch.diff(trajectory_tensor, dim=0)
                    smoothness = torch.mean(torch.norm(diffs, dim=-1)).item()
                    trajectory_errors.append(smoothness)
        
        return {
            'trajectory_smoothness': np.mean(trajectory_errors) if trajectory_errors else 0.0,
            'num_trajectories': len(trajectory_errors)
        }
    
    @torch.no_grad()
    def evaluate_causal_reasoning(self, data_loader: DataLoader) -> Dict:
        """Evaluate causal reasoning capabilities"""
        
        causal_scores = []
        
        # Create synthetic causal scenarios
        scenarios = [
            {
                'name': 'joy_to_sadness',
                'trigger': 'bad_news',
                'expected_transition': (3, 4)  # Happy to Sad
            },
            {
                'name': 'neutral_to_surprise',
                'trigger': 'unexpected_event',
                'expected_transition': (6, 5)  # Neutral to Surprise
            }
        ]
        
        for scenario in scenarios:
            correct_predictions = 0
            total_predictions = 0
            
            for batch in data_loader:
                visual = batch['visual'].to(self.device)
                audio = batch['audio'].to(self.device)
                context = batch['context'].to(self.device)
                
                # Forward pass
                outputs = self.model(visual, audio, context)
                
                # Analyze emotional transitions
                if 'level2_emotion_probs' in outputs:
                    emotion_probs = outputs['level2_emotion_probs']
                    
                    # Find transitions matching the scenario
                    for t in range(emotion_probs.shape[1] - 1):
                        current_emotion = emotion_probs[:, t].argmax(dim=-1)
                        next_emotion = emotion_probs[:, t + 1].argmax(dim=-1)
                        
                        # Check if transition matches expected pattern
                        mask = current_emotion == scenario['expected_transition'][0]
                        if mask.any():
                            predicted_next = next_emotion[mask]
                            correct = (predicted_next == scenario['expected_transition'][1]).sum().item()
                            correct_predictions += correct
                            total_predictions += mask.sum().item()
            
            if total_predictions > 0:
                accuracy = correct_predictions / total_predictions
                causal_scores.append({
                    'scenario': scenario['name'],
                    'accuracy': accuracy,
                    'support': total_predictions
                })
        
        results = {
            'scenarios': causal_scores,
            'average_accuracy': np.mean([s['accuracy'] for s in causal_scores]) if causal_scores else 0.0
        }
        
        self.results['causal_reasoning'] = results
        return results
    
    @torch.no_grad()
    def analyze_emergent_behaviors(self, data_loader: DataLoader) -> Dict:
        """Analyze emergent behaviors of the model"""
        
        behaviors = {
            'anticipatory_responses': [],
            'context_sensitivity': [],
            'emotional_contagion': [],
            'regulation_effectiveness': []
        }
        
        for batch in tqdm(data_loader, desc="Analyzing emergent behaviors"):
            visual = batch['visual'].to(self.device)
            audio = batch['audio'].to(self.device)
            context = batch['context'].to(self.device)
            
            # Forward pass
            outputs = self.model(visual, audio, context, return_all=True)
            
            # Analyze anticipatory responses
            if 'level1_precision1' in outputs and 'level2_emotion_probs' in outputs:
                precision = outputs['level1_precision1'].cpu().numpy()
                emotion_probs = outputs['level2_emotion_probs'].cpu().numpy()
                
                # High precision before emotional peaks indicates anticipation
                emotion_entropy = -np.sum(emotion_probs * np.log(emotion_probs + 1e-8), axis=-1)
                precision_mean = precision.mean(axis=-1)
                
                # Correlation between precision and future emotion certainty
                if len(precision_mean.shape) > 1 and precision_mean.shape[1] > 1:
                    for i in range(precision_mean.shape[0]):
                        if len(precision_mean[i]) > 10:
                            corr = np.corrcoef(precision_mean[i][:-5], emotion_entropy[i][5:])[0, 1]
                            behaviors['anticipatory_responses'].append(corr)
            
            # Analyze context sensitivity
            if 'level3_mood_logits' in outputs:
                mood_logits = outputs['level3_mood_logits'].cpu().numpy()
                # Variability in mood based on context indicates sensitivity
                mood_variability = np.std(mood_logits, axis=1).mean()
                behaviors['context_sensitivity'].append(mood_variability)
            
            # Analyze regulation effectiveness
            if 'regulation' in outputs and 'ai_F_total' in outputs:
                regulation_signal = outputs['regulation'].cpu().numpy()
                free_energy = outputs['ai_F_total'].cpu().numpy()
                
                # Effective regulation should reduce free energy
                reg_magnitude = np.linalg.norm(regulation_signal, axis=-1).mean()
                behaviors['regulation_effectiveness'].append({
                    'regulation_strength': reg_magnitude,
                    'free_energy': free_energy.mean()
                })
        
        # Aggregate results
        results = {
            'anticipatory_score': np.mean(behaviors['anticipatory_responses']) if behaviors['anticipatory_responses'] else 0.0,
            'context_sensitivity_score': np.mean(behaviors['context_sensitivity']) if behaviors['context_sensitivity'] else 0.0,
            'regulation_effectiveness': np.mean([b['regulation_strength'] for b in behaviors['regulation_effectiveness']]) if behaviors['regulation_effectiveness'] else 0.0
        }
        
        self.results['emergent_behaviors'] = results
        return results
    
    def _plot_confusion_matrix(self, cm: np.ndarray, labels: List[str]):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.title('Emotion Recognition Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300)
        plt.close()
    
    def visualize_hierarchical_attention(self, sample_input: Dict) -> None:
        """Visualize attention patterns across hierarchy"""
        
        visual = sample_input['visual'].unsqueeze(0).to(self.device)
        audio = sample_input['audio'].unsqueeze(0).to(self.device)
        context = sample_input['context'].unsqueeze(0).to(self.device)
        
        # Get attention weights
        outputs = self.model(visual, audio, context, return_all=True)
        
        # Plot attention heatmaps for each level
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Level 1: Micro-expression attention (if available)
        # Level 2: State transition attention
        # Level 3: Memory attention
        
        # This is a placeholder - actual implementation would extract
        # attention weights from the model
        
        plt.tight_layout()
        plt.savefig('hierarchical_attention.png', dpi=300)
        plt.close()
    
    def generate_report(self, output_path: str = 'evaluation_report.json'):
        """Generate comprehensive evaluation report"""
        
        # Add metadata
        self.results['metadata'] = {
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'evaluation_date': str(Path().absolute()),
            'device': str(self.device)
        }
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        if 'emotion_recognition' in self.results:
            er = self.results['emotion_recognition']
            print(f"\nEmotion Recognition:")
            print(f"  Accuracy: {er['accuracy']:.4f}")
            print(f"  F1 Score (Macro): {er['f1_macro']:.4f}")
            print(f"  Average Confidence: {er['average_confidence']:.4f}")
        
        if 'temporal_prediction' in self.results:
            tp = self.results['temporal_prediction']
            print(f"\nTemporal Prediction:")
            print(f"  MSE: {tp['mse']:.4f}")
            print(f"  MAE: {tp['mae']:.4f}")
        
        if 'causal_reasoning' in self.results:
            cr = self.results['causal_reasoning']
            print(f"\nCausal Reasoning:")
            print(f"  Average Accuracy: {cr['average_accuracy']:.4f}")
        
        if 'emergent_behaviors' in self.results:
            eb = self.results['emergent_behaviors']
            print(f"\nEmergent Behaviors:")
            print(f"  Anticipatory Score: {eb['anticipatory_score']:.4f}")
            print(f"  Context Sensitivity: {eb['context_sensitivity_score']:.4f}")
            print(f"  Regulation Effectiveness: {eb['regulation_effectiveness']:.4f}")
        
        print("\n" + "="*50)
        print(f"Full report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Hierarchical EI Model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to model configuration')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Path to evaluation data')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--output', type=str, default='evaluation_report.json',
                        help='Output path for evaluation report')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize evaluator
    evaluator = Evaluator(args.model, args.config)
    
    # Load evaluation data
    eval_dataset = EmotionDataset(args.data_dir, split='test')
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4)
    
    # Run evaluations
    logger.info("Starting evaluation...")
    
    # 1. Emotion Recognition
    logger.info("Evaluating emotion recognition...")
    evaluator.evaluate_emotion_recognition(eval_loader)
    
    # 2. Temporal Prediction
    logger.info("Evaluating temporal prediction...")
    evaluator.evaluate_temporal_prediction(eval_loader)
    
    # 3. Causal Reasoning
    logger.info("Evaluating causal reasoning...")
    evaluator.evaluate_causal_reasoning(eval_loader)
    
    # 4. Emergent Behaviors
    logger.info("Analyzing emergent behaviors...")
    evaluator.analyze_emergent_behaviors(eval_loader)
    
    # Generate report
    evaluator.generate_report(args.output)
    
    logger.info("Evaluation completed!")


if __name__ == '__main__':
    main() 
