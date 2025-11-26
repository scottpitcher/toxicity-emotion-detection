# eval_baseline.py
"""
Evaluation script for BaselineBERT models.
Evaluates both weighted and unweighted checkpoints on toxicity test set.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import os
import logging
import argparse
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, accuracy_score, classification_report
)
import numpy as np

from model import BaselineBERT
from train_utils import load_toxicity_test_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_baseline(model, test_loader, device):
    """
    Evaluate baseline model on toxicity test set.

    Returns:
        dict: Comprehensive metrics including F1, precision, recall, ROC-AUC, accuracy
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0

    logger.info("Running evaluation...")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            logits = model(input_ids, attention_mask)

            # Compute loss
            loss = model.compute_loss(logits, labels)
            total_loss += loss.item()

            # Get predictions (sigmoid for multi-label)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics
    metrics = {}

    # Macro F1 (average across all labels)
    metrics['f1_macro'] = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # Micro F1 (aggregate across all labels)
    metrics['f1_micro'] = f1_score(all_labels, all_preds, average='micro', zero_division=0)

    # Weighted F1
    metrics['f1_weighted'] = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    # Precision and Recall
    metrics['precision_macro'] = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    metrics['precision_micro'] = precision_score(all_labels, all_preds, average='micro', zero_division=0)
    metrics['recall_macro'] = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    metrics['recall_micro'] = recall_score(all_labels, all_preds, average='micro', zero_division=0)

    # Subset accuracy (all labels must match)
    metrics['subset_accuracy'] = accuracy_score(all_labels, all_preds)

    # ROC-AUC (using probabilities)
    try:
        metrics['roc_auc_macro'] = roc_auc_score(all_labels, all_probs, average='macro')
        metrics['roc_auc_micro'] = roc_auc_score(all_labels, all_probs, average='micro')
    except ValueError:
        metrics['roc_auc_macro'] = 0.0
        metrics['roc_auc_micro'] = 0.0

    # Per-class metrics
    label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

    for i, label in enumerate(label_names):
        metrics[f'f1_{label}'] = per_class_f1[i]

    # Average loss
    metrics['test_loss'] = total_loss / len(test_loader)

    return metrics


def main():
    """Main evaluation function."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file")
    parser.add_argument("--output", type=str, default="../results/baseline_eval_results.csv",
                        help="Path to save results CSV")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {DEVICE}")

    try:
        # Load checkpoint
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)

        # Extract metadata
        class_weights = checkpoint.get('class_weights', None)
        is_weighted = class_weights is not None

        logger.info(f"Checkpoint type: {'Weighted' if is_weighted else 'Unweighted'}")
        if 'epochs' in checkpoint:
            logger.info(f"Trained for {checkpoint['epochs']} epochs")

        # Initialize model
        logger.info("Initializing BaselineBERT model...")
        model = BaselineBERT(num_toxicity_labels=6, class_weights=class_weights)

        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Model loaded successfully")

        # Load test data
        logger.info("Loading toxicity test data...")
        test_loader = load_toxicity_test_data(batch_size=args.batch_size)
        logger.info(f"Test set: {len(test_loader)} batches")

        # Evaluate
        logger.info("Starting evaluation...")
        metrics = evaluate_baseline(model, test_loader, DEVICE)

        # Print results
        logger.info("\n" + "="*60)
        logger.info("EVALUATION RESULTS")
        logger.info("="*60)
        logger.info(f"F1 Score (Macro):     {metrics['f1_macro']:.4f}")
        logger.info(f"F1 Score (Micro):     {metrics['f1_micro']:.4f}")
        logger.info(f"F1 Score (Weighted):  {metrics['f1_weighted']:.4f}")
        logger.info(f"Precision (Macro):    {metrics['precision_macro']:.4f}")
        logger.info(f"Recall (Macro):       {metrics['recall_macro']:.4f}")
        logger.info(f"ROC-AUC (Macro):      {metrics['roc_auc_macro']:.4f}")
        logger.info(f"Subset Accuracy:      {metrics['subset_accuracy']:.4f}")
        logger.info(f"Test Loss:            {metrics['test_loss']:.4f}")

        logger.info("\nPer-class F1 Scores:")
        label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        for label in label_names:
            logger.info(f"  {label:15s}: {metrics[f'f1_{label}']:.4f}")
        logger.info("="*60)

        # Save results
        results_df = pd.DataFrame([{
            'checkpoint': args.checkpoint,
            'model_type': 'baseline',
            'weighted': is_weighted,
            **metrics
        }])

        # Create output directory if needed
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

        # Append to existing results or create new file
        if os.path.exists(args.output):
            existing_df = pd.read_csv(args.output)
            results_df = pd.concat([existing_df, results_df], ignore_index=True)

        results_df.to_csv(args.output, index=False)
        logger.info(f"\nResults saved to {args.output}")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise


if __name__ == "__main__":
    main()
