# eval_multitask.py
"""
Evaluation script for MultiTaskBERT models.
Evaluates both weighted and unweighted checkpoints on both toxicity and emotion test sets.
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
    roc_auc_score, accuracy_score
)
import numpy as np

from model import MultiTaskBERT
from train_utils import load_toxicity_test_data, load_emotion_test_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_toxicity(model, test_loader, device):
    """
    Evaluate multi-task model on toxicity test set.

    Returns:
        dict: Toxicity metrics
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0

    logger.info("Evaluating toxicity task...")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Toxicity"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass - get toxicity logits
            tox_logits, _ = model(input_ids, attention_mask)

            # Compute loss
            loss = model.compute_tox_loss(tox_logits, labels)
            total_loss += loss.item()

            # Get predictions
            probs = torch.sigmoid(tox_logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics
    metrics = {
        'tox_f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'tox_f1_micro': f1_score(all_labels, all_preds, average='micro', zero_division=0),
        'tox_f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'tox_precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'tox_precision_micro': precision_score(all_labels, all_preds, average='micro', zero_division=0),
        'tox_recall_macro': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'tox_recall_micro': recall_score(all_labels, all_preds, average='micro', zero_division=0),
        'tox_subset_accuracy': accuracy_score(all_labels, all_preds),
        'tox_test_loss': total_loss / len(test_loader)
    }

    # ROC-AUC
    try:
        metrics['tox_roc_auc_macro'] = roc_auc_score(all_labels, all_probs, average='macro')
        metrics['tox_roc_auc_micro'] = roc_auc_score(all_labels, all_probs, average='micro')
    except ValueError:
        metrics['tox_roc_auc_macro'] = 0.0
        metrics['tox_roc_auc_micro'] = 0.0

    # Per-class F1
    label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    for i, label in enumerate(label_names):
        metrics[f'tox_f1_{label}'] = per_class_f1[i]

    return metrics


def evaluate_emotion(model, test_loader, device):
    """
    Evaluate multi-task model on emotion test set.

    Returns:
        dict: Emotion metrics
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0

    logger.info("Evaluating emotion task...")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Emotion"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass - get emotion logits
            _, emo_logits = model(input_ids, attention_mask)

            # Compute loss
            loss = model.compute_emo_loss(emo_logits, labels)
            total_loss += loss.item()

            # Get predictions
            probs = torch.sigmoid(emo_logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics
    metrics = {
        'emo_f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'emo_f1_micro': f1_score(all_labels, all_preds, average='micro', zero_division=0),
        'emo_f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'emo_precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'emo_precision_micro': precision_score(all_labels, all_preds, average='micro', zero_division=0),
        'emo_recall_macro': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'emo_recall_micro': recall_score(all_labels, all_preds, average='micro', zero_division=0),
        'emo_subset_accuracy': accuracy_score(all_labels, all_preds),
        'emo_test_loss': total_loss / len(test_loader)
    }

    # ROC-AUC
    try:
        metrics['emo_roc_auc_macro'] = roc_auc_score(all_labels, all_probs, average='macro')
        metrics['emo_roc_auc_micro'] = roc_auc_score(all_labels, all_probs, average='micro')
    except ValueError:
        metrics['emo_roc_auc_macro'] = 0.0
        metrics['emo_roc_auc_micro'] = 0.0

    return metrics


def main():
    """Main evaluation function."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file")
    parser.add_argument("--output", type=str, default="../results/multitask_eval_results.csv",
                        help="Path to save results CSV")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--eval_emotion", action="store_true",
                        help="Also evaluate on emotion test set")
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {DEVICE}")

    try:
        # Load checkpoint
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)

        # Extract metadata
        tox_weights = checkpoint.get('tox_class_weights', None)
        emo_weights = checkpoint.get('emo_class_weights', None)
        lambda_tox = checkpoint.get('lambda_tox', 1.0)
        lambda_emo = checkpoint.get('lambda_emo', 1.0)
        is_weighted = tox_weights is not None or emo_weights is not None

        logger.info(f"Checkpoint type: {'Weighted' if is_weighted else 'Unweighted'}")
        logger.info(f"Lambda toxicity: {lambda_tox}, Lambda emotion: {lambda_emo}")

        # Initialize model
        logger.info("Initializing MultiTaskBERT model...")
        model = MultiTaskBERT(
            num_toxicity_labels=6,
            num_emotion_labels=28,
            lambda_tox=lambda_tox,
            lambda_emo=lambda_emo,
            tox_class_weights=tox_weights,
            emo_class_weights=emo_weights
        )

        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Model loaded successfully")

        # Load test data
        logger.info("Loading toxicity test data...")
        tox_test_loader = load_toxicity_test_data(batch_size=args.batch_size)
        logger.info(f"Toxicity test set: {len(tox_test_loader)} batches")

        # Evaluate toxicity
        logger.info("Starting toxicity evaluation...")
        metrics = evaluate_toxicity(model, tox_test_loader, DEVICE)

        # Print toxicity results
        logger.info("\n" + "="*60)
        logger.info("TOXICITY EVALUATION RESULTS")
        logger.info("="*60)
        logger.info(f"F1 Score (Macro):     {metrics['tox_f1_macro']:.4f}")
        logger.info(f"F1 Score (Micro):     {metrics['tox_f1_micro']:.4f}")
        logger.info(f"F1 Score (Weighted):  {metrics['tox_f1_weighted']:.4f}")
        logger.info(f"Precision (Macro):    {metrics['tox_precision_macro']:.4f}")
        logger.info(f"Recall (Macro):       {metrics['tox_recall_macro']:.4f}")
        logger.info(f"ROC-AUC (Macro):      {metrics['tox_roc_auc_macro']:.4f}")
        logger.info(f"Subset Accuracy:      {metrics['tox_subset_accuracy']:.4f}")
        logger.info(f"Test Loss:            {metrics['tox_test_loss']:.4f}")
        logger.info("="*60)

        # Evaluate emotion if requested
        if args.eval_emotion:
            logger.info("\nLoading emotion test data...")
            emo_test_loader = load_emotion_test_data(batch_size=args.batch_size)
            logger.info(f"Emotion test set: {len(emo_test_loader)} batches")

            logger.info("Starting emotion evaluation...")
            emo_metrics = evaluate_emotion(model, emo_test_loader, DEVICE)
            metrics.update(emo_metrics)

            # Print emotion results
            logger.info("\n" + "="*60)
            logger.info("EMOTION EVALUATION RESULTS")
            logger.info("="*60)
            logger.info(f"F1 Score (Macro):     {emo_metrics['emo_f1_macro']:.4f}")
            logger.info(f"F1 Score (Micro):     {emo_metrics['emo_f1_micro']:.4f}")
            logger.info(f"F1 Score (Weighted):  {emo_metrics['emo_f1_weighted']:.4f}")
            logger.info(f"Precision (Macro):    {emo_metrics['emo_precision_macro']:.4f}")
            logger.info(f"Recall (Macro):       {emo_metrics['emo_recall_macro']:.4f}")
            logger.info(f"ROC-AUC (Macro):      {emo_metrics['emo_roc_auc_macro']:.4f}")
            logger.info(f"Subset Accuracy:      {emo_metrics['emo_subset_accuracy']:.4f}")
            logger.info(f"Test Loss:            {emo_metrics['emo_test_loss']:.4f}")
            logger.info("="*60)

        # Save results
        results_df = pd.DataFrame([{
            'checkpoint': args.checkpoint,
            'model_type': 'multitask',
            'weighted': is_weighted,
            'lambda_tox': lambda_tox,
            'lambda_emo': lambda_emo,
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
