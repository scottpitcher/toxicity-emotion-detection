# predictions.py
"""
Get model predictions for input text.
Returns class probabilities for all toxicity types.
"""

import torch
import numpy as np
from transformers import BertTokenizer
from typing import Dict, Tuple
import json


class PredictionGetter:
    """Get model predictions."""
    
    def __init__(self, model, device="cpu", tokenizer_name="bert-base-uncased"):
        """
        Args:
            model: Trained BERT model
            device: Device to run on (cpu or cuda)
            tokenizer_name: HuggingFace tokenizer name
        """
        self.model = model
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.model.eval()
        self.model.to(device)
        
        self.class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    def predict(self, text: str) -> Dict:
        """
        Get predictions for input text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with predictions for each class
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
            logits = self.model(input_ids, attention_mask)
            
            # Handle MultiTaskBERT (returns tuple)
            if isinstance(logits, tuple):
                logits = logits[0]
            
            # Apply sigmoid for multi-label classification
            probs = torch.sigmoid(logits[0]).cpu().numpy()
        
        # Create predictions dict
        predictions = {}
        for class_name, prob in zip(self.class_names, probs):
            predictions[class_name] = float(prob)
        
        return predictions
    
    def predict_batch(self, texts: list) -> list:
        """
        Get predictions for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results
    
    def predict_with_confidence(self, text: str, threshold: float = 0.5) -> Dict:
        """
        Get predictions with confidence classification.
        
        Args:
            text: Input text
            threshold: Classification threshold (default 0.5)
            
        Returns:
            Dictionary with predictions, classifications, and confidence levels
        """
        predictions = self.predict(text)
        
        result = {
            'input': text,
            'predictions': predictions,
            'classifications': {},
            'high_confidence': [],
            'medium_confidence': [],
            'low_confidence': []
        }
        
        for class_name, prob in predictions.items():
            # Classify
            if prob > threshold:
                classification = 'POSITIVE'
            else:
                classification = 'NEGATIVE'
            
            result['classifications'][class_name] = classification
            
            # Confidence level
            confidence = abs(prob - 0.5) * 2  # 0-1 scale, 0.5 = no confidence
            
            if confidence > 0.7:
                result['high_confidence'].append({
                    'class': class_name,
                    'score': prob,
                    'confidence': confidence
                })
            elif confidence > 0.3:
                result['medium_confidence'].append({
                    'class': class_name,
                    'score': prob,
                    'confidence': confidence
                })
            else:
                result['low_confidence'].append({
                    'class': class_name,
                    'score': prob,
                    'confidence': confidence
                })
        
        return result
    
    def get_primary_toxicity(self, text: str) -> Dict:
        """
        Get the primary toxicity class (highest probability).
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with primary class info
        """
        predictions = self.predict(text)
        primary_class = max(predictions.items(), key=lambda x: x[1])
        
        return {
            'input': text,
            'primary_class': primary_class[0],
            'score': primary_class[1],
            'all_predictions': predictions
        }
    
    def compare_predictions(self, text: str, model1, model2) -> Dict:
        """
        Compare predictions between two models.
        
        Args:
            text: Input text
            model1: First model (e.g., baseline)
            model2: Second model (e.g., multi-task)
            
        Returns:
            Comparison dictionary
        """
        # Get predictions from both
        pred_getter1 = PredictionGetter(model1, device=self.device)
        pred_getter2 = PredictionGetter(model2, device=self.device)
        
        preds1 = pred_getter1.predict(text)
        preds2 = pred_getter2.predict(text)
        
        # Compare
        comparison = {
            'input': text,
            'model1_predictions': preds1,
            'model2_predictions': preds2,
            'differences': {},
            'agreement': {}
        }
        
        for class_name in self.class_names:
            diff = preds2[class_name] - preds1[class_name]
            comparison['differences'][class_name] = diff
            
            # Check agreement
            model1_pos = preds1[class_name] > 0.5
            model2_pos = preds2[class_name] > 0.5
            
            if model1_pos == model2_pos:
                comparison['agreement'][class_name] = 'AGREE'
            else:
                comparison['agreement'][class_name] = 'DISAGREE'
        
        return comparison


def main():
    """Test prediction getting."""
    from model import BaselineBERT, MultiTaskBERT
    
    # Load baseline model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    baseline_model = BaselineBERT()
    checkpoint = torch.load("models/baseline_bert.pt", map_location=device)
    baseline_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load multi-task model
    multitask_model = MultiTaskBERT()
    checkpoint = torch.load("models/multitask_bert.pt", map_location=device)
    multitask_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create getter
    getter = PredictionGetter(baseline_model, device=device)
    
    # Test texts
    test_texts = [
        "I'm so angry at this policy",
        "You are an idiot",
        "This is wonderful",
        "I hope you get hit by a car"
    ]
    
    print("="*70)
    print("BASELINE MODEL PREDICTIONS")
    print("="*70)
    
    for text in test_texts:
        print(f"\nText: \"{text}\"")
        
        # Get basic predictions
        preds = getter.predict(text)
        print(f"\nPredictions:")
        for class_name, score in sorted(preds.items(), key=lambda x: x[1], reverse=True):
            status = "✓ TOXIC" if score > 0.5 else "✗ NOT TOXIC"
            print(f"  {class_name:<15} {score:.4f}  {status}")
        
        # Get primary
        primary = getter.get_primary_toxicity(text)
        print(f"\nPrimary class: {primary['primary_class']} ({primary['score']:.4f})")
    
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    for text in test_texts:
        print(f"\nText: \"{text}\"")
        
        comparison = getter.compare_predictions(text, baseline_model, multitask_model)
        
        print(f"\n{'Class':<15} {'Baseline':<12} {'Multi-Task':<12} {'Difference':<12} {'Agreement'}")
        print("-"*65)
        
        for class_name in getter.class_names:
            baseline_score = comparison['model1_predictions'][class_name]
            multitask_score = comparison['model2_predictions'][class_name]
            diff = comparison['differences'][class_name]
            agreement = comparison['agreement'][class_name]
            
            symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
            print(f"{class_name:<15} {baseline_score:>6.4f}       {multitask_score:>6.4f}       "
                  f"{symbol}{abs(diff):>6.4f}       {agreement}")


if __name__ == "__main__":
    main()
