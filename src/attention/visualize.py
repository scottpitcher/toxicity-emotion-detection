# visualize.py
"""
Visualize token-level attribution and predictions for a single model.
Creates heatmap with token importance and per-class contributions.
"""

import torch
import numpy as np
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, List, Dict

from attention import AttentionExtractor
from attribution import AttributionCalculator
from predictions import PredictionGetter


class SingleModelVisualizer:
    """Visualize predictions and attributions for single model."""
    
    def __init__(self, model, device="cpu", tokenizer_name="bert-base-uncased"):
        """
        Args:
            model: Trained BERT model
            device: Device to run on
            tokenizer_name: HuggingFace tokenizer name
        """
        self.model = model
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        
        self.attention_extractor = AttentionExtractor(model, device, tokenizer_name)
        self.attribution_calculator = AttributionCalculator(model, device, tokenizer_name)
        self.prediction_getter = PredictionGetter(model, device, tokenizer_name)
    
    def visualize_full(self, text: str, output_path: str = None, figsize: Tuple[int, int] = (16, 10)):
        """
        Create comprehensive visualization with:
        - Token importance (attention)
        - Per-class attributions
        - Predictions
        
        Args:
            text: Input text
            output_path: Path to save figure
            figsize: Figure size
        """
        # Get data
        tokens, saliency = self.attribution_calculator.get_gradient_saliency(text)
        tokens_attr, attributions = self.attribution_calculator.get_all_class_attributions(text)
        predictions = self.prediction_getter.predict(text)
        
        # Normalize saliency
        if saliency.max() > saliency.min():
            saliency_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min())
        else:
            saliency_norm = np.zeros_like(saliency)
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 1, height_ratios=[2, 3, 1], hspace=0.4)
        
        # ===== Panel 1: Token Heatmap with Importance =====
        ax1 = fig.add_subplot(gs[0])
        self._draw_token_heatmap(ax1, tokens, saliency_norm, text)
        
        # ===== Panel 2: Token Contributions by Class =====
        ax2 = fig.add_subplot(gs[1])
        self._draw_attribution_matrix(ax2, tokens_attr, attributions)
        
        # Panel 3: Predictions 
        ax3 = fig.add_subplot(gs[2])
        self._draw_predictions(ax3, predictions)
        
        plt.suptitle(f'Model Predictions and Attribution Analysis\nText: "{text}"',
                    fontsize=16, fontweight='bold', y=0.98)
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {output_path}")
        else:
            plt.show()
        
        return fig
    
    def _draw_token_heatmap(self, ax, tokens: List[str], saliency: np.ndarray, text: str):
        """Draw colored tokens with importance."""
        # Colors
        colors = plt.cm.Reds(saliency)
        
        x_pos = 0
        max_x = len(tokens)
        
        for i, (token, sal) in enumerate(zip(tokens, saliency)):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            
            # Rectangle
            rect = patches.Rectangle((x_pos, 0), 1, 1,
                                    linewidth=2,
                                    edgecolor='black',
                                    facecolor=colors[i][:3])
            ax.add_patch(rect)
            
            # Token text
            text_color = 'white' if sal > 0.5 else 'black'
            ax.text(x_pos + 0.5, 0.5, token,
                   ha='center', va='center',
                   fontsize=11, fontweight='bold',
                   color=text_color)
            
            x_pos += 1
        
        ax.set_xlim(0, x_pos)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Token Importance (Red = High Importance)', 
                    fontsize=12, fontweight='bold', loc='left')
    
    def _draw_attribution_matrix(self, ax, tokens: List[str], attributions: Dict):
        """Draw attribution matrix (tokens × classes)."""
        class_names = list(attributions.keys())
        
        # Create matrix
        matrix = np.zeros((len(class_names), len(tokens)))
        for i, class_name in enumerate(class_names):
            matrix[i] = attributions[class_name]
        
        # Normalize for visualization
        max_abs = np.abs(matrix).max()
        if max_abs > 0:
            matrix_norm = matrix / max_abs
        else:
            matrix_norm = matrix
        
        # Plot
        im = ax.imshow(matrix_norm, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # Labels
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(class_names, fontsize=10)
        
        ax.set_xlabel('Tokens', fontsize=11, fontweight='bold')
        ax.set_ylabel('Toxicity Classes', fontsize=11, fontweight='bold')
        ax.set_title('Token Contributions by Class (Red = Increases Score, Blue = Decreases)',
                    fontsize=12, fontweight='bold', loc='left')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('Attribution Score', fontsize=10)
        
        # Add values in cells
        for i in range(len(class_names)):
            for j in range(len(tokens)):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                             ha="center", va="center",
                             color="white" if abs(matrix_norm[i, j]) > 0.5 else "black",
                             fontsize=8)
    
    def _draw_predictions(self, ax, predictions: Dict):
        """Draw prediction scores."""
        class_names = list(predictions.keys())
        scores = list(predictions.values())
        colors_bar = ['red' if s > 0.5 else 'orange' if s > 0.3 else 'green' 
                     for s in scores]
        
        bars = ax.barh(class_names, scores, color=colors_bar, edgecolor='black', linewidth=2)
        
        # Add score labels
        for bar, score in zip(bars, scores):
            ax.text(score + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{score:.3f}',
                   va='center', fontsize=11, fontweight='bold')
        
        ax.set_xlim(0, 1.0)
        ax.set_xlabel('Prediction Score', fontsize=11, fontweight='bold')
        ax.set_title('Toxicity Predictions (Green < 0.3, Orange 0.3-0.5, Red > 0.5)',
                    fontsize=12, fontweight='bold', loc='left')
        ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Threshold (0.5)')
        ax.grid(axis='x', alpha=0.3)
        ax.legend(loc='lower right')
    
    def print_summary(self, text: str):
        """Print text summary."""
        tokens, saliency = self.attribution_calculator.get_gradient_saliency(text)
        tokens_attr, attributions = self.attribution_calculator.get_all_class_attributions(text)
        predictions = self.prediction_getter.predict(text)
        
        print("\n" + "="*80)
        print("SINGLE MODEL ANALYSIS")
        print("="*80)
        
        print(f"\nInput: \"{text}\"")
        print(f"\nTokens: {tokens}")
        
        print(f"\n{'Top 5 Important Tokens (by Saliency):'}")
        top_indices = np.argsort(saliency)[-5:][::-1]
        for rank, idx in enumerate(top_indices, 1):
            print(f"  {rank}. '{tokens[idx]}': {saliency[idx]:.4f}")
        
        print(f"\n{'Predictions:':<20} {'Score':<10} {'Classification'}")
        print("-"*40)
        for class_name, score in predictions.items():
            classification = "TOXIC" if score > 0.5 else "NON-TOXIC"
            print(f"{class_name:<20} {score:.4f}     {classification}")
        
        print(f"\n{'Token Contributions by Class:'}")
        for class_name, attr_scores in attributions.items():
            print(f"\n  {class_name.upper()}:")
            top_indices = np.argsort(np.abs(attr_scores))[-3:][::-1]
            for rank, idx in enumerate(top_indices, 1):
                direction = "↑" if attr_scores[idx] > 0 else "↓"
                print(f"    {rank}. '{tokens_attr[idx]}' {direction} {abs(attr_scores[idx]):.4f}")
        
        print("\n" + "="*80 + "\n")


def main():
    """Test visualization."""
    from model import BaselineBERT
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BaselineBERT()
    checkpoint = torch.load("models/baseline_bert.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create visualizer
    visualizer = SingleModelVisualizer(model, device=device)
    
    # Test texts
    test_texts = [
        "I'm so angry at this policy",
        "You are an idiot",
        "I hope you get hit by a car"
    ]
    
    for text in test_texts:
        visualizer.print_summary(text)
        visualizer.visualize_full(text, output_path=f"viz_{hash(text) % 10000}.png")


if __name__ == "__main__":
    main()
