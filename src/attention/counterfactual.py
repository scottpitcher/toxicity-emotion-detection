# counterfactual.py
"""
Simple counterfactual rewriting: find most toxic token → find synonyms → replace until non-toxic.
"""

import torch
import numpy as np
from transformers import BertTokenizer, pipeline
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.attention.attribution import AttributionCalculator
from src.attention.predictions import PredictionGetter


class ContextualSynonymGenerator:
    """Generate contextual synonyms using MLM and/or seq2seq models."""
    
    def __init__(self, device="cpu", use_paraphrase_model: bool = True):
        """
        Args:
            device: Device to run on
            use_paraphrase_model: Load pre-trained paraphrase model
        """
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
        # MLM pipeline for contextual synonyms
        self.mlm_pipeline = None
        self._init_mlm_pipeline()
        
        # Seq2seq paraphrase model
        self.paraphrase_model = None
        self.paraphrase_tokenizer = None
        if use_paraphrase_model:
            self._init_paraphrase_model()
    
    def _init_mlm_pipeline(self):
        """Initialize BERT MLM for fast synonym generation."""
        try:
            self.mlm_pipeline = pipeline(
                "fill-mask",
                model="bert-base-uncased",
                device=0 if self.device == "cuda" else -1
            )
        except Exception as e:
            print(f"Warning: MLM pipeline failed: {e}")
    
    def _init_paraphrase_model(self):
        """Initialize BART paraphrase model."""
        try:
            from transformers import BartForConditionalGeneration, BartTokenizer
            model_name = "eugenesiow/bart-paraphrase"
            self.paraphrase_model = BartForConditionalGeneration.from_pretrained(model_name)
            self.paraphrase_tokenizer = BartTokenizer.from_pretrained(model_name)
            self.paraphrase_model.to(self.device)
            self.paraphrase_model.eval()
            print(f"Loaded paraphrase model: {model_name}")
        except Exception as e:
            print(f"Warning: Paraphrase model failed: {e}")
            self.paraphrase_model = None
    
    def get_synonyms_mlm(self, text: str, token_position: int, top_k: int = 10) -> List[str]:
        """
        Generate synonyms by masking token and predicting alternatives.
        
        Args:
            text: Input text
            token_position: Position of token to replace (word index, not subword)
            top_k: Number of candidates to return
            
        Returns:
            List of candidate words
        """
        if not self.mlm_pipeline:
            return []
        
        words = text.split()
        if token_position >= len(words):
            return []
        
        # Replace target word with [MASK]
        masked_text = ' '.join(
            '[MASK]' if i == token_position else word 
            for i, word in enumerate(words)
        )
        
        try:
            predictions = self.mlm_pipeline(masked_text, top_k=top_k)
            candidates = [p['token_str'].strip() for p in predictions]
            return candidates
        except Exception as e:
            print(f"MLM error: {e}")
            return []
    
    def get_synonyms_paraphrase(self, text: str, token_position: int, top_k: int = 10) -> List[str]:
        """
        Generate paraphrases of full text and extract alternative words.
        
        Args:
            text: Input text
            token_position: Position of token to replace
            top_k: Number of candidates to return
            
        Returns:
            List of candidate words
        """
        if not self.paraphrase_model:
            return []
        
        try:
            inputs = self.paraphrase_tokenizer(
                text,
                return_tensors="pt",
                max_length=128,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.paraphrase_model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=5,
                    num_return_sequences=min(5, top_k),
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9
                )
            
            paraphrases = self.paraphrase_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            
            # Extract words that appear in paraphrases but not original
            original_words = set(text.lower().split())
            candidates = []
            
            for paraphrase in paraphrases:
                para_words = set(paraphrase.lower().split())
                new_words = para_words - original_words
                candidates.extend(list(new_words)[:3])
            
            return list(dict.fromkeys(candidates))[:top_k]
        except Exception as e:
            print(f"Paraphrase error: {e}")
            return []
    
    def get_synonyms(self, text: str, token_position: int, 
                    strategy: str = "hybrid", top_k: int = 10) -> List[str]:
        """
        Get synonyms using selected strategy.
        
        Args:
            text: Input text
            token_position: Position of token to replace
            strategy: "mlm", "paraphrase", or "hybrid"
            top_k: Number of candidates
            
        Returns:
            List of candidate synonyms
        """
        candidates = []
        
        if strategy in ["mlm", "hybrid"]:
            mlm_cands = self.get_synonyms_mlm(text, token_position, top_k)
            candidates.extend(mlm_cands)
        
        if strategy in ["paraphrase", "hybrid"] and self.paraphrase_model:
            para_cands = self.get_synonyms_paraphrase(text, token_position, top_k)
            candidates.extend(para_cands)
        
        # Remove duplicates, keep order
        seen = set()
        result = []
        for c in candidates:
            if c.lower() not in seen:
                result.append(c)
                seen.add(c.lower())
        
        return result[:top_k] if result else []


class CounterfactualRewriter:
    """
    Rewrite toxic text by replacing most-toxic token with synonyms.
    """
    
    def __init__(self, model, device="cpu", tokenizer_name="bert-base-uncased",
                 synonym_strategy: str = "hybrid", use_paraphrase_model: bool = True):
        """
        Args:
            model: Trained BERT toxicity model
            device: Device to run on
            tokenizer_name: Tokenizer name
            synonym_strategy: "mlm", "paraphrase", or "hybrid"
            use_paraphrase_model: Load paraphrase model
        """
        self.model = model
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.synonym_strategy = synonym_strategy
        
        self.attribution_calculator = AttributionCalculator(model, device, tokenizer_name)
        self.prediction_getter = PredictionGetter(model, device, tokenizer_name)
        self.synonym_generator = ContextualSynonymGenerator(device, use_paraphrase_model)
    
    def get_most_toxic_token(self, text: str, 
                            toxicity_class: str = 'toxic') -> Optional[Dict]:
        """
        Find token contributing most to toxicity.
        
        Args:
            text: Input text
            toxicity_class: Which toxicity class to analyze
            
        Returns:
            Dict with token, position, and attribution score. None if no toxic tokens.
        """
        tokens, attributions = self.attribution_calculator.get_all_class_attributions(text)
        target_attr = attributions[toxicity_class]
        
        # Ignore special tokens
        special_tokens = {'[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]'}
        
        # Find max attribution token (excluding special tokens and punctuation)
        max_attr = -float('inf')
        max_idx = -1
        
        for i, (token, attr) in enumerate(zip(tokens, target_attr)):
            if token not in special_tokens and attr > max_attr:
                max_attr = attr
                max_idx = i
        
        if max_idx == -1 or max_attr <= 0:
            return None
        
        return {
            'token': tokens[max_idx],
            'position': max_idx,
            'attribution': float(max_attr),
            'normalized': float(max_attr / (target_attr.max() + 1e-8))
        }
    
    def replace_token_in_text(self, text: str, token_position: int, replacement: str) -> str:
        """
        Replace a token at given word position.
        
        Args:
            text: Input text
            token_position: Word position (not subword index)
            replacement: Replacement word
            
        Returns:
            Modified text
        """
        words = text.split()
        
        if token_position < 0 or token_position >= len(words):
            return text
        
        if replacement == '[REMOVE]':
            words.pop(token_position)
        else:
            words[token_position] = replacement
        
        return ' '.join(words)
    
    def evaluate_candidate(self, original: str, modified: str, 
                          toxicity_class: str = 'toxic') -> Dict:
        """
        Score a candidate rewrite.
        
        Args:
            original: Original text
            modified: Modified text
            toxicity_class: Toxicity class to minimize
            
        Returns:
            Evaluation dict with scores and metrics
        """
        orig_pred = self.prediction_getter.predict(original)
        mod_pred = self.prediction_getter.predict(modified)
        
        orig_tox = orig_pred[toxicity_class]
        mod_tox = mod_pred[toxicity_class]
        
        # Semantic similarity via Jaccard overlap
        orig_words = set(original.lower().split())
        mod_words = set(modified.lower().split())
        overlap = len(orig_words & mod_words) / len(orig_words | mod_words) if orig_words or mod_words else 0
        
        return {
            'original_toxicity': orig_tox,
            'modified_toxicity': mod_tox,
            'toxicity_reduced': orig_tox - mod_tox,
            'is_non_toxic': mod_tox < 0.5,
            'semantic_similarity': overlap,
            'all_pred_original': orig_pred,
            'all_pred_modified': mod_pred
        }
    
    def rewrite(self, text: str, toxicity_class: str = 'toxic', 
               max_attempts: int = 20, verbose: bool = False) -> Dict:
        """
        Rewrite text until non-toxic (<0.5) by replacing most-toxic token.
        
        Core algorithm:
        1. Get current toxicity score
        2. If already non-toxic, return
        3. Find most toxic token by attribution
        4. Generate synonyms for that token
        5. Try each synonym, pick one that reduces toxicity most
        6. If found, apply it and repeat; otherwise stop
        
        Args:
            text: Input text
            toxicity_class: Toxicity class to minimize
            max_attempts: Max iterations (each iteration = replace 1 token)
            verbose: Print progress
            
        Returns:
            Result dict with original, rewritten, edits, etc.
        """
        current_text = text
        edits = []
        
        # Get initial toxicity
        initial_pred = self.prediction_getter.predict(text)
        initial_tox = initial_pred[toxicity_class]
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Starting toxicity: {initial_tox:.4f}")
            print(f"Target: < 0.5 (non-toxic)")
            print('='*70)
        
        for attempt in range(max_attempts):
            # Get current toxicity
            current_pred = self.prediction_getter.predict(current_text)
            current_tox = current_pred[toxicity_class]
            
            if verbose:
                print(f"\nAttempt {attempt + 1}: Current toxicity = {current_tox:.4f}")
            
            # Stop if non-toxic
            if current_tox < 0.5:
                if verbose:
                    print(f"✓ Non-toxic threshold reached! ({current_tox:.4f} < 0.5)")
                break
            
            # Find most toxic token
            toxic_token_info = self.get_most_toxic_token(current_text, toxicity_class)
            
            if not toxic_token_info:
                if verbose:
                    print("No toxic tokens identified.")
                break
            
            token = toxic_token_info['token']
            position = toxic_token_info['position']
            attribution = toxic_token_info['attribution']
            
            if verbose:
                print(f"  Most toxic token: '{token}' (attribution: {attribution:.4f})")
            
            # Generate synonyms
            synonyms = self.synonym_generator.get_synonyms(
                current_text, position, 
                strategy=self.synonym_strategy, 
                top_k=15
            )
            
            if verbose:
                print(f"  Candidates: {synonyms[:5]}")
            
            # Try each synonym, find best
            best_candidate = None
            best_reduction = 0
            best_eval = None
            
            for synonym in synonyms:
                # Skip if same as original
                if synonym.lower() == token.lower():
                    continue
                
                modified_text = self.replace_token_in_text(current_text, position, synonym)
                
                # Skip if empty
                if not modified_text.strip():
                    continue
                
                eval_result = self.evaluate_candidate(
                    current_text, modified_text, toxicity_class
                )
                
                reduction = eval_result['toxicity_reduced']
                
                # Pick best
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_candidate = synonym
                    best_eval = eval_result
            
            # Apply best candidate if found and reduces toxicity
            if best_candidate and best_reduction > 0.01:
                current_text = self.replace_token_in_text(current_text, position, best_candidate)
                
                edits.append({
                    'original_token': token,
                    'replacement': best_candidate,
                    'attribution_before': attribution,
                    'toxicity_before': best_eval['original_toxicity'],
                    'toxicity_after': best_eval['modified_toxicity'],
                    'reduction': best_reduction,
                    'text_after': current_text
                })
                
                if verbose:
                    print(f"  ✓ Applied: '{token}' → '{best_candidate}'")
                    print(f"    Toxicity: {best_eval['original_toxicity']:.4f} → {best_eval['modified_toxicity']:.4f}")
            else:
                if verbose:
                    print(f"  ✗ No beneficial synonym found.")
                break
        
        # Final evaluation
        final_pred = self.prediction_getter.predict(current_text)
        final_tox = final_pred[toxicity_class]
        
        return {
            'input': text,
            'output': current_text,
            'initial_toxicity': initial_tox,
            'final_toxicity': final_tox,
            'toxicity_reduced': initial_tox - final_tox,
            'is_non_toxic': final_tox < 0.5,
            'num_edits': len(edits),
            'edits': edits,
            'target_class': toxicity_class,
            'all_predictions_input': initial_pred,
            'all_predictions_output': final_pred
        }
    
    def visualize(self, result: Dict, output_path: str = None):
        """Visualize rewriting result."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Panel 1: Before/After
        ax = axes[0, 0]
        ax.axis('off')
        ax.text(0.5, 0.75, "Original:", fontsize=12, fontweight='bold', ha='center')
        ax.text(0.5, 0.60, result['input'], fontsize=11, ha='center', wrap=True,
                style='italic', bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.7))
        ax.text(0.5, 0.45, "Rewritten:", fontsize=12, fontweight='bold', ha='center')
        ax.text(0.5, 0.30, result['output'], fontsize=11, ha='center', wrap=True,
                style='italic', bbox=dict(boxstyle='round', facecolor='#ccffcc', alpha=0.7))
        
        # Panel 2: Toxicity before/after
        ax = axes[0, 1]
        classes = list(result['all_predictions_input'].keys())
        input_scores = list(result['all_predictions_input'].values())
        output_scores = list(result['all_predictions_output'].values())
        
        x = np.arange(len(classes))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, input_scores, width, label='Input', color='#e74c3c', alpha=0.7)
        bars2 = ax.bar(x + width/2, output_scores, width, label='Output', color='#27ae60', alpha=0.7)
        
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Threshold (0.5)')
        ax.set_ylabel('Toxicity Score', fontsize=11, fontweight='bold')
        ax.set_title('All Toxicity Classes', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Panel 3: Edits
        ax = axes[1, 0]
        ax.axis('off')
        
        if result['edits']:
            edits_text = "Token Replacements:\n\n"
            for i, edit in enumerate(result['edits'][:6], 1):
                edits_text += (
                    f"{i}. '{edit['original_token']}' → '{edit['replacement']}'\n"
                    f"   Toxicity: {edit['toxicity_before']:.3f} → {edit['toxicity_after']:.3f}\n\n"
                )
        else:
            edits_text = "No edits applied."
        
        ax.text(0.05, 0.95, edits_text, fontsize=10, verticalalignment='top',
               family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # Panel 4: Summary
        ax = axes[1, 1]
        ax.axis('off')
        
        reduction_pct = (1 - result['final_toxicity'] / max(result['initial_toxicity'], 0.001)) * 100
        status = "✓ SUCCESS" if result['is_non_toxic'] else "◐ IMPROVED" if result['toxicity_reduced'] > 0.1 else "✗ MINIMAL CHANGE"
        
        summary = (
            f"Summary\n"
            f"{'─'*50}\n"
            f"Initial Toxicity:  {result['initial_toxicity']:.4f}\n"
            f"Final Toxicity:    {result['final_toxicity']:.4f}\n"
            f"Reduction:         {result['toxicity_reduced']:.4f} ({reduction_pct:.1f}%)\n"
            f"\n"
            f"Token Edits:       {result['num_edits']}\n"
            f"Target Class:      {result['target_class']}\n"
            f"Non-Toxic:         {'Yes ✓' if result['is_non_toxic'] else 'No ✗'}\n"
            f"{'─'*50}\n"
            f"Status: {status}"
        )
        
        ax.text(0.05, 0.95, summary, fontsize=11, verticalalignment='top',
               family='monospace', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        plt.suptitle('Counterfactual Rewriting Result', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\nVisualization saved to {output_path}")
        else:
            plt.show()
    
    def print_summary(self, result: Dict):
        """Print text summary."""
        print("\n" + "="*80)
        print("COUNTERFACTUAL REWRITING")
        print("="*80)
        print(f"\nInput:  \"{result['input']}\"")
        print(f"Output: \"{result['output']}\"")
        print(f"\nToxicity ({result['target_class']}):")
        print(f"  Initial: {result['initial_toxicity']:.4f}")
        print(f"  Final:   {result['final_toxicity']:.4f}")
        print(f"  Reduced: {result['toxicity_reduced']:.4f}")
        print(f"  Status:  {'✓ Non-toxic' if result['is_non_toxic'] else '✗ Still toxic'}")
        
        if result['edits']:
            print(f"\nEdits ({len(result['edits'])} total):")
            for i, edit in enumerate(result['edits'], 1):
                print(f"  {i}. '{edit['original_token']}' → '{edit['replacement']}'")
                print(f"     {edit['toxicity_before']:.4f} → {edit['toxicity_after']:.4f}")
        
        print("="*80 + "\n")


def main():
    """Test rewriting."""
    from src.model import BaselineBERT
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BaselineBERT()
    checkpoint = torch.load("models/bert_base_unw.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    rewriter = CounterfactualRewriter(
        model, device=device,
        synonym_strategy="hybrid",
        use_paraphrase_model=True
    )
    
    texts = [
        "You are an idiot",
        "I'm so angry at this policy",
        "I hate this so much"
    ]
    
    for text in texts:
        result = rewriter.rewrite(text, verbose=True)
        rewriter.print_summary(result)
        rewriter.visualize(result)


if __name__ == "__main__":
    main()