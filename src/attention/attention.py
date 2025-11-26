"""
Extract and analyze BERT attention weights.
Shows which tokens attend to which tokens across all layers.
"""

import torch
import numpy as np
from transformers import BertTokenizer
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt


class AttentionExtractor:
    """Extract and analyze BERT attention weights."""
    
    def __init__(self, model, device="cpu", tokenizer_name="bert-base-uncased"):
        """
        Args:
            model: Trained BERT model with attention outputs
            device: Device to run on (cpu or cuda)
            tokenizer_name: HuggingFace tokenizer name
        """
        self.model = model
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.model.eval()
        self.model.to(device)
    
    def get_attention_weights(self, text: str, layer: int = -1, head: int = None) -> Tuple[List[str], np.ndarray]:
        """
        Extract attention weights from specified layer.
        
        Args:
            text: Input text
            layer: Which layer to extract (-1 for last layer, 0-11 for specific)
            head: Which attention head (None = average all heads)
            
        Returns:
            tokens: List of tokens
            attention_weights: Array of shape (seq_len, seq_len) or (num_heads, seq_len, seq_len)
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
        
        # Forward pass with attention outputs
        with torch.no_grad():
            outputs = self.model.shared_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
        
        # Extract attention from specified layer
        # outputs.attentions is tuple of 12 layers
        # each layer: (batch_size, num_heads, seq_len, seq_len)
        layer_attention = outputs.attentions[layer]  # (1, 12, 128, 128)
        layer_attention = layer_attention[0].cpu().numpy()  # (12, 128, 128)
        
        # Average across heads if not specified
        if head is None:
            layer_attention = layer_attention.mean(axis=0)  # (128, 128)
        else:
            layer_attention = layer_attention[head]  # (128, 128)
        
        # Get actual tokens (remove padding)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
        actual_length = attention_mask.sum().item()
        tokens = tokens[:actual_length]
        layer_attention = layer_attention[:actual_length, :actual_length]
        
        return tokens, layer_attention
    
    def get_token_attention_scores(self, text: str, token_idx: int) -> Dict:
        """
        Get attention scores FOR a specific token (what does it attend to?).
        
        Args:
            text: Input text
            token_idx: Index of token to analyze
            
        Returns:
            Dictionary with attention distribution across all tokens
        """
        tokens, attention = self.get_attention_weights(text)
        
        if token_idx >= len(tokens):
            raise ValueError(f"Token index {token_idx} out of range ({len(tokens)} tokens)")
        
        # Get attention row for this token (what this token attends to)
        attention_row = attention[token_idx]
        
        result = {
            'query_token': tokens[token_idx],
            'attention_to_tokens': {}
        }
        
        for i, token in enumerate(tokens):
            result['attention_to_tokens'][token] = float(attention_row[i])
        
        return result
    
    def get_all_layer_attention(self, text: str) -> Dict:
        """
        Get average attention from all 12 layers.
        Aggregates attention across all layers and heads.
        
        Args:
            text: Input text
            
        Returns:
            tokens and aggregated attention matrix
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
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model.shared_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
        
        # Average across all layers and heads
        all_attention = []
        for layer_attn in outputs.attentions:
            # layer_attn: (1, 12, 128, 128)
            all_attention.append(layer_attn[0].cpu().numpy())  # (12, 128, 128)
        
        all_attention = np.array(all_attention)  # (12, 12, 128, 128)
        averaged_attention = all_attention.mean(axis=(0, 1))  # (128, 128)
        
        # Get actual tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
        actual_length = attention_mask.sum().item()
        tokens = tokens[:actual_length]
        averaged_attention = averaged_attention[:actual_length, :actual_length]
        
        return tokens, averaged_attention
    
    def visualize_attention_heatmap(self, text: str, layer: int = -1, output_path: str = None):
        """
        Create heatmap visualization of attention weights.
        
        Args:
            text: Input text
            layer: Which layer to visualize
            output_path: Path to save figure
        """
        tokens, attention = self.get_attention_weights(text, layer=layer)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        im = ax.imshow(attention, cmap='viridis', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticklabels(tokens)
        
        ax.set_xlabel('Attending To (Key)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Attending From (Query)', fontsize=12, fontweight='bold')
        ax.set_title(f'BERT Attention Heatmap (Layer {layer})\nText: "{text}"', 
                    fontsize=14, fontweight='bold')
        
        plt.colorbar(im, ax=ax, label='Attention Weight')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved attention heatmap to {output_path}")
        else:
            plt.show()


def main():
    """Test attention extraction."""
    from model import BaselineBERT
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BaselineBERT()
    checkpoint = torch.load("models/baseline_bert.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create extractor
    extractor = AttentionExtractor(model, device=device)
    
    # Test texts
    test_texts = [
        "I'm so angry at this policy",
        "You are an idiot",
        "This is wonderful"
    ]
    
    for text in test_texts:
        print(f"\n{'='*60}")
        print(f"Text: {text}")
        print('='*60)
        
        # Get attention for token 2 (e.g., "so")
        tokens, attention = extractor.get_attention_weights(text)
        print(f"\nTokens: {tokens}")
        
        # Show what token 2 attends to
        if len(tokens) > 2:
            token_attention = extractor.get_token_attention_scores(text, token_idx=2)
            print(f"\nToken '{token_attention['query_token']}' attends to:")
            for tok, score in sorted(
                token_attention['attention_to_tokens'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]:
                print(f"  {tok}: {score:.4f}")
        
        # Visualize
        extractor.visualize_attention_heatmap(text, output_path=f"attention_{hash(text) % 10000}.png")


if __name__ == "__main__":
    main()