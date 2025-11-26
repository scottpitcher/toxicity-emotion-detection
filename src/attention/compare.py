# compare.py
"""
Compare predictions and attributions between baseline and multi-task models.
Shows side-by-side visualizations and highlights differences.
"""

import torch
import numpy as np
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, List, Dict

from attribution import AttributionCalculator
from predictions import PredictionGetter


class ModelComparator:
    """Compare two models side-by-side."""
    
    def __init__(self, model1, model2, device="cpu", tokenizer_name="bert-base-uncased", 
                 model1_name="Baseline", model2_name="Multi-Task"):
        """
        Args:
            model1: First model (usually baseline)
            model2: Second model (usually multi-task)
            device: Device to run on
            tokenizer_name: HuggingFace tokenizer name
            model1_name: Display name for model1
            model2_name: Display name for model2
        """
        self.model1 = model1
        self.model2 = model2
        self.device = device
        self.model1_name = model1_name
        self.model2_name = model2_name
        
        self.attr1 = AttributionCalculator(model1, device, tokenizer_name)
        self.attr2 = AttributionCalculator(model2, device, tokenizer_name)
        
        self.pred1 = PredictionGetter(model1, device, tokenizer_name)
        self.pred2 = PredictionGetter(model2, device, tokenizer_name)
    
    def compare_predictions(self, text: str) -> Dict:
        """
        Compare predictions between two models.
        
        Args:
            text: Input text
            
        Returns:
            Comparison dictionary
        """
        preds1 = self.pred1.predict(text)
        preds2 = self.pred2.predict(text)
        
        comparison = {
            'input': text,
            'model1': preds1,
            'model2': preds2,
            'differences': {},
            'agreement': {}
        }
        
        for class_name in preds1.keys():
            diff = preds2[class_name] - preds1[class_name]
            comparison['differences'][class_name] = diff
            
            # Agreement
            pos1 = preds1[class_name] > 0.5
            pos2 = preds2[class_name] > 0.5
            comparison['agreement'][class_name] = 'AGREE' if pos1 == pos2 else 'DISAGREE'
        
        return comparison
    
    def compare_attributions(self, text: str, class_idx: int = 0) -> Tuple[List[str], Dict]:
        """
        Compare attributions for specific class.
        
        Args:
            text: Input text
            class_idx: Toxicity class to compare (0-5)
            
        Returns:
            tokens and attribution comparison
        """
        tokens1, attr1 = self.attr1.get_integrated_gradients(text, class_idx)
        tokens2, attr2 = self.attr2.get_integrated_gradients(text, class_idx)
        
        comparison = {
            'tokens': tokens1,
            'model1_attributions': attr1,
            'model2_attributions': attr2,
            'differences': attr2 - attr1
        }
        
        return comparison
    
    def visualize_comparison(self, text: str, output_path: str = None, 
                            figsize: Tuple[int, int] = (18, 12)):
        """
        Create side-by-side comparison visualization.
        
        Args:
            text: Input text
            output_path: Path to save figure
            figsize: Figure size
        """
        # Get data
        preds_comp = self.compare_predictions(text)
        attr_comp = self.compare_attributions(text, class_idx=0)
        
        tokens = attr_comp['tokens']
        attr1 = attr_comp['model1_attributions']
        attr2 = attr_comp['model2_attributions']
        diff = attr_comp['differences']
        
        preds1 = preds_comp['model1']
        preds2 = preds_comp['model2']
        agreement = preds_comp['agreement']
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # ===== Top Left: Model1 Predictions =====
        self._draw_predictions(axes[0, 0], preds1, f'{self.model1_name} Predictions')
        
        # ===== Top Right: Model2 Predictions =====
        self._draw_predictions(axes[0, 1], preds2, f'{self.model2_name} Predictions')
        
        # ===== Bottom Left: Model1 Attribution Heatmap =====
        self._draw_attribution_heatmap(axes[1, 0], tokens, attr1, 
                                      f'{self.model1_name} Token Attribution')
        
        # ===== Bottom Right: Model2 Attribution Heatmap =====
        self._draw_attribution_heatmap(axes[1, 1], tokens, attr2, 
                                      f'{self.model2_name} Token Attribution')
        
        # Overall title
        fig.suptitle(f'Model Comparison: {self.model1_name} vs {self.model2_name}\n'
                    f'Text: "{text}"',
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved comparison to {output_path}")
        else:
            plt.show()
        
        return fig
    
    def _draw_predictions(self, ax, predictions: Dict, title: str):
        """Draw prediction bar chart."""
        class_names = list(predictions.keys())
        scores = list(predictions.values())
        colors_bar = ['red' if s > 0.5 else 'orange' if s > 0.3 else 'green' 
                     for s in scores]
        
        bars = ax.barh(class_names, scores, color=colors_bar, edgecolor='black', linewidth=2)
        
        for bar, score in zip(bars, scores):
            ax.text(score + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{score:.3f}', va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlim(0, 1.0)
        ax.set_xlabel('Score', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.grid(axis='x', alpha=0.3)
    
    def _draw_attribution_heatmap(self, ax, tokens: List[str], attributions: np.ndarray, title: str):
        """Draw attribution heatmap for tokens."""
        # Reshape for visualization
        attr_reshaped = attributions.reshape(1, -1)
        
        # Normalize
        max_abs = np.abs(attributions).max()
        if max_abs > 0:
            attr_norm = attributions / max_abs
        else:
            attr_norm = attributions
        
        # Plot
        im = ax.imshow(attr_reshaped, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # Labels
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks([0])
        ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(['Attribution'], fontsize=10)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.15)
        cbar.set_label('Attribution Score', fontsize=10)
        
        # Add values
        for j in range(len(tokens)):
            color = "white" if abs(attr_norm[j]) > 0.5 else "black"
            ax.text(j, 0, f'{attributions[j]:.2f}',
                   ha="center", va="center", color=color, fontsize=8, fontweight='bold')
    
    def print_comparison_summary(self, text: str):
        """Print detailed text summary."""
        preds_comp = self.compare_predictions(text)
        
        print("\n" + "="*90)
        print(f"COMPARISON: {self.model1_name} vs {self.model2_name}")
        print("="*90)
        
        print(f"\nInput text: \"{text}\"")
        
        print(f"\n{'Class':<20} {self.model1_name:<15} {self.model2_name:<15} "
              f"{'Difference':<15} {'Agreement':<12}")
        print("-"*85)
        
        for class_name in preds_comp['model1'].keys():
            score1 = preds_comp['model1'][class_name]
            score2 = preds_comp['model2'][class_name]
            diff = preds_comp['differences'][class_name]
            agreement = preds_comp['agreement'][class_name]
            
            symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
            
            print(f"{class_name:<20} {score1:>6.4f}         {score2:>6.4f}         "
                  f"{symbol} {abs(diff):>6.4f}        {agreement:<12}")
        
        # Key findings
        print(f"\nKey Findings:")
        disagreements = [c for c, a in preds_comp['agreement'].items() if a == 'DISAGREE']
        if disagreements:
            print(f"  ⚠ Models disagree on: {', '.join(disagreements)}")
        else:
            print(f"  ✓ Models agree on all classes")
        
        avg_diff = np.abs(list(preds_comp['differences'].values())).mean()
        print(f"  • Average score difference: {avg_diff:.4f}")
        
        print("\n" + "="*90 + "\n")


def main():
    """Test comparison."""
    from model import BaselineBERT, MultiTaskBERT
    
    # Load models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    baseline = BaselineBERT()
    checkpoint = torch.load("models/baseline_bert.pt", map_location=device)
    baseline.load_state_dict(checkpoint['model_state_dict'])
    
    multitask = MultiTaskBERT()
    checkpoint = torch.load("models/multitask_bert.pt", map_location=device)
    multitask.load_state_dict(checkpoint['model_state_dict'])
    
    # Create comparator
    comparator = ModelComparator(baseline, multitask, device=device,
                                model1_name="Baseline", model2_name="Multi-Task")
    
    # Test texts
    test_texts = [
        "I'm so angry at this policy",
        "You are an idiot",
        "I hope you get hit by a car",
        "This is wonderful"
    ]
    
    for text in test_texts:
        comparator.print_comparison_summary(text)
        comparator.visualize_comparison(text, output_path=f"comparison_{hash(text) % 10000}.png")


if __name__ == "__main__":
    main()
