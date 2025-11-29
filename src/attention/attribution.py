"""
Compute token-level attribution using Integrated Gradients.
Shows how much each token contributes to each toxicity class prediction.
"""

import torch
import numpy as np
from transformers import BertTokenizer
from typing import Tuple, List, Dict


class AttributionCalculator:
    """Compute token-level attribution for model predictions."""
    
    def __init__(self, model, device="cpu", tokenizer_name="bert-base-uncased", num_steps: int = 50):
        """
        Args:
            model: Trained BERT model
            device: Device to run on (cpu or cuda)
            tokenizer_name: HuggingFace tokenizer name
            num_steps: Number of interpolation steps for Integrated Gradients
        """
        self.model = model
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.num_steps = num_steps
        self.model.eval()
        self.model.to(device)
        
        # Toxicity class names
        self.class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    def get_integrated_gradients(self, text: str, class_idx: int = 0) -> Tuple[List[str], np.ndarray]:
        """
        Compute Integrated Gradients attribution.
        
        Args:
            text: Input text
            class_idx: Which toxicity class to attribute (0-5)
            
        Returns:
            tokens: List of tokens
            attributions: Attribution scores for each token
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
        actual_length = attention_mask.sum().item()
        
        # Get embedding layer
        embedding_layer = self.model.shared_encoder.embeddings.word_embeddings
        
        # Get embeddings for input
        with torch.no_grad():
            input_embeddings = embedding_layer(input_ids)  # (1, 128, 768)
        
        # Create baseline (all padding)
        baseline_embeddings = embedding_layer(torch.full_like(input_ids, self.tokenizer.pad_token_id))
        
        # Initialize gradients
        accumulated_grads = torch.zeros_like(input_embeddings)
        
        # Integrated Gradients: interpolate from baseline to input
        for step in range(self.num_steps):
            # Interpolation coefficient
            alpha = step / self.num_steps
            
            # Interpolated embeddings
            interpolated_embeddings = baseline_embeddings + alpha * (input_embeddings - baseline_embeddings)
            interpolated_embeddings = interpolated_embeddings.detach().requires_grad_(True)
            
            # Forward pass
            outputs = self.model.shared_encoder(
                inputs_embeds=interpolated_embeddings,
                attention_mask=attention_mask
            )
            
            cls_output = outputs.last_hidden_state[:, 0, :]
            cls_output = self.model.dropout(cls_output)
            logits = self.model.toxicity_head(cls_output)
            
            # Get logit for specified class
            class_logit = logits[0, class_idx]
            
            # Backward
            class_logit.backward(retain_graph=True)
            
            # Accumulate gradients
            if interpolated_embeddings.grad is not None:
                accumulated_grads += interpolated_embeddings.grad.detach()
            
            # Clear gradients
            if interpolated_embeddings.grad is not None:
                interpolated_embeddings.grad.zero_()
        
        # Average gradients across steps
        avg_grads = accumulated_grads / self.num_steps
        
        # Compute attribution = gradient * (input - baseline)
        input_embeddings_detached = input_embeddings.detach()
        attribution = (input_embeddings_detached - baseline_embeddings) * avg_grads
        
        # Sum across embedding dimension
        attribution_scores = attribution.sum(dim=-1)[0].detach().cpu().numpy()  
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
        tokens = tokens[:actual_length]
        attribution_scores = attribution_scores[:actual_length]
        
        return tokens, attribution_scores
    
    def get_all_class_attributions(self, text: str) -> Tuple[List[str], Dict[str, np.ndarray]]:
        """
        Compute attribution for all toxicity classes.
        
        Args:
            text: Input text
            
        Returns:
            tokens: List of tokens
            attributions: Dict mapping class name → attribution scores
        """
        attributions = {}
        
        for class_idx, class_name in enumerate(self.class_names):
            tokens, attr_scores = self.get_integrated_gradients(text, class_idx=class_idx)
            attributions[class_name] = attr_scores
        
        return tokens, attributions
    def get_top_attributed_tokens(self, text: str, class_idx: int = 0, top_k: int = 5) -> List[Dict]:
        """
        Get top-k tokens by absolute attribution for a class.
        
        Args:
            text: Input text
            class_idx: Toxicity class
            top_k: Number of top tokens to return
            
        Returns:
            List of dicts with token, attribution, sign
        """
        tokens, attr = self.get_integrated_gradients(text, class_idx)
        
        # Get top by absolute value
        top_indices = np.argsort(np.abs(attr))[-top_k:][::-1]
        
        result = []
        for idx in top_indices:
            result.append({
                'token': tokens[idx],
                'attribution': float(attr[idx]),
                'sign': 'increases' if attr[idx] > 0 else 'decreases',
                'index': idx
            })
    
    def get_gradient_saliency(self, text: str) -> Tuple[List[str], np.ndarray]:
        """
        Simpler approach: gradient-based saliency (faster than Integrated Gradients).
        Shows which tokens have highest gradient magnitude.
        
        Args:
            text: Input text
            
        Returns:
            tokens: List of tokens
            saliency: Saliency scores for each token
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
        actual_length = attention_mask.sum().item()
        
        # Get embeddings with gradients
        embedding_layer = self.model.shared_encoder.embeddings.word_embeddings
        embeddings = embedding_layer(input_ids)
        embeddings.requires_grad = True
        
        # Forward pass
        outputs = self.model.shared_encoder(
            inputs_embeds=embeddings,
            attention_mask=attention_mask
        )
        
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.model.dropout(cls_output)
        logits = self.model.toxicity_head(cls_output)
        
        # Use max toxicity score
        max_score = logits[0].max()
        
        # Backward
        max_score.backward()
        
        # Get saliency
        saliency = embeddings.grad.abs().mean(dim=-1)[0].detach().cpu().numpy()
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
        tokens = tokens[:actual_length]
        saliency = saliency[:actual_length]
        
        return tokens, saliency


def main():
    """Test attribution calculation."""
    from model import BaselineBERT
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BaselineBERT()
    checkpoint = torch.load("models/baseline_bert.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create calculator
    calculator = AttributionCalculator(model, device=device, num_steps=50)
    
    # Test texts
    test_texts = [
        "I'm so angry at this policy",
        "You are an idiot",
        "I hope you get hit by a car"
    ]
    
    for text in test_texts:
        print(f"\n{'='*70}")
        print(f"Text: {text}")
        print('='*70)
        
        # Get saliency (faster)
        print("\n--- Gradient Saliency (Fast) ---")
        tokens, saliency = calculator.get_gradient_saliency(text)
        print(f"Tokens: {tokens}")
        print(f"Top 5 tokens by saliency:")
        top_indices = np.argsort(saliency)[-5:][::-1]
        for rank, idx in enumerate(top_indices, 1):
            print(f"  {rank}. '{tokens[idx]}': {saliency[idx]:.4f}")
        
        # Get per-class attributions
        print("\n--- Per-Class Attributions (Integrated Gradients) ---")
        tokens, attributions = calculator.get_all_class_attributions(text)
        
        for class_name, attr_scores in attributions.items():
            print(f"\n{class_name.upper()}:")
            top_indices = np.argsort(attr_scores)[-3:][::-1]
            for rank, idx in enumerate(top_indices, 1):
                print(f"  {rank}. '{tokens[idx]}': +{attr_scores[idx]:.4f}")


if __name__ == "__main__":
    main()