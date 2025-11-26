"""
Script to generate attention-based heatmap explanations for toxicity predictions.
Takes a model, input text, and generates visualization showing which tokens
contribute most to the toxicity prediction.
"""

import torch
import numpy as np
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, List, Dict
import seaborn as sns


class ToxicityExplainer:
    """Generate attention-based explanations for toxicity predictions."""
    
    def __init__(self, model, device="cpu", tokenizer_name="bert-base-uncased"):
        """
        Args:
            model: Trained toxicity model (BaselineBERT or MultiTaskBERT)
            device: Device to run on (cpu or cuda)
            tokenizer_name: HuggingFace tokenizer name
        """
        self.model = model
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.model.eval()
        self.model.to(device)
    
    def get_gradient_saliency(self, text: str) -> Tuple[List[str], np.ndarray, torch.Tensor]:
        """
        Compute gradient-based saliency for each token.
        Shows which tokens have highest gradient w.r.t. toxicity prediction.
        
        Args:
            text: Input text to explain
            
        Returns:
            tokens: List of tokens
            saliency: Numpy array of saliency scores per token
            predictions: Model predictions (logits)
        """
        # Tokenize
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_tensors='pt',
            padding='max_length',
            max_length=128,
            truncation=True
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Enable gradient computation
        input_ids.requires_grad = False
        
        # Get embeddings and enable grad
        with torch.enable_grad():
            # Get embedding layer
            embedding_layer = self.model.shared_encoder.embeddings.word_embeddings
            embeddings = embedding_layer(input_ids)
            embeddings.requires_grad = True
            
            # Forward pass through shared encoder
            outputs = self.model.shared_encoder(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                output_attentions=False
            )
            
            cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            cls_output = self.model.dropout(cls_output)
            
            # Get toxicity predictions
            logits = self.model.toxicity_head(cls_output)  # Shape: (1, 6)
            
            # Use max toxicity score (most toxic class)
            max_tox_score = logits[0].max()
            
            # Compute gradient
            max_tox_score.backward()
            
            # Get saliency from embedding gradients
            saliency = embeddings.grad.abs().mean(dim=-1).detach().cpu().numpy()[0]
        
        # Decode tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
        
        # Remove padding and special tokens for clean output
        actual_length = attention_mask.sum().item()
        tokens = tokens[:actual_length]
        saliency = saliency[:actual_length]
        
        return tokens, saliency, logits[0].detach().cpu()
    
    def get_predictions(self, text: str) -> Dict:
        """
        Get model predictions for text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_tensors='pt',
            padding='max_length',
            max_length=128,
            truncation=True
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            if isinstance(logits, tuple):  # MultiTaskBERT
                logits = logits[0]
            
            probs = torch.sigmoid(logits[0]).cpu().numpy()
        
        toxicity_types = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        
        predictions = {}
        for i, tox_type in enumerate(toxicity_types):
            predictions[tox_type] = float(probs[i])
        
        return predictions
    
    def visualize_heatmap(self, text: str, output_path: str = None, figsize: Tuple[int, int] = (14, 6)):
        """
        Create and display heatmap visualization.
        
        Args:
            text: Input text to visualize
            output_path: Path to save figure (if None, displays)
            figsize: Figure size
        """
        tokens, saliency, logits = self.get_gradient_saliency(text)
        predictions = self.get_predictions(text)
        
        # Normalize saliency to 0-1
        saliency_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
        
        # ===== Top: Token heatmap =====
        # Create color array (red for high saliency)
        colors = plt.cm.Reds(saliency_norm)
        
        # Draw colored boxes with tokens
        x_pos = 0
        for i, (token, sal) in enumerate(zip(tokens, saliency_norm)):
            # Skip [CLS] and [SEP] for cleaner viz
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            
            # Draw rectangle
            rect = patches.Rectangle((x_pos, 0), 1, 1, 
                                     linewidth=1, 
                                     edgecolor='black',
                                     facecolor=colors[i][:3])
            ax1.add_patch(rect)
            
            # Add token text
            ax1.text(x_pos + 0.5, 0.5, token, 
                    ha='center', va='center', 
                    fontsize=10, fontweight='bold',
                    color='white' if sal > 0.5 else 'black')
            
            x_pos += 1
        
        ax1.set_xlim(0, x_pos)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        ax1.set_title(f'Token Importance Heatmap for Toxicity Detection\n"...{text[-60:]}"', 
                     fontsize=12, fontweight='bold')
        
        # ===== Bottom: Toxicity type predictions =====
        tox_types = list(predictions.keys())
        scores = list(predictions.values())
        colors_bar = ['red' if s > 0.5 else 'orange' if s > 0.3 else 'green' for s in scores]
        
        bars = ax2.barh(tox_types, scores, color=colors_bar, edgecolor='black', linewidth=1.5)
        
        # Add score labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax2.text(score + 0.02, i, f'{score:.2f}', 
                    va='center', fontsize=10, fontweight='bold')
        
        ax2.set_xlim(0, 1)
        ax2.set_xlabel('Prediction Score', fontsize=11, fontweight='bold')
        ax2.set_title('Toxicity Type Scores', fontsize=11, fontweight='bold')
        ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')
        ax2.legend(loc='lower right')
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {output_path}")
        else:
            plt.show()
        
        return fig
    
    def print_explanation(self, text: str):
        """Print text explanation of predictions."""
        tokens, saliency, logits = self.get_gradient_saliency(text)
        predictions = self.get_predictions(text)
        
        print("\n" + "="*60)
        print("TOXICITY PREDICTION EXPLANATION")
        print("="*60)
        
        print(f"\nInput text: \"{text}\"")
        print(f"\n{'Toxicity Type':<20} {'Score':<10} {'Classification':<15}")
        print("-"*45)
        
        for tox_type, score in predictions.items():
            classification = "TOXIC" if score > 0.5 else "NON-TOXIC"
            print(f"{tox_type:<20} {score:.4f}     {classification:<15}")
        
        # Top contributing tokens
        top_indices = np.argsort(saliency)[-5:][::-1]
        print(f"\nTop 5 most important tokens:")
        for rank, idx in enumerate(top_indices, 1):
            if tokens[idx] not in ['[CLS]', '[SEP]', '[PAD]']:
                print(f"  {rank}. '{tokens[idx]}' (importance: {saliency[idx]:.4f})")
        
        print("\n" + "="*60 + "\n")


def main():
    """Example usage."""
    import sys
    
    # Load model
    from model import BaselineBERT, MultiTaskBERT
    
    model_path = "../models/baseline_final.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize model
    model = BaselineBERT()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create explainer
    explainer = ToxicityExplainer(model, device=device)
    
    # Example texts
    test_texts = [
        "I really love this movie",
        "You are an idiot",
        "I'm so angry at this policy",
        "I hope you get hit by a car",
        "This is absolutely wonderful!"
    ]
    
    for text in test_texts:
        explainer.print_explanation(text)
        explainer.visualize_heatmap(text, output_path=f"explanation_{hash(text) % 10000}.png")


if __name__ == "__main__":
    main()