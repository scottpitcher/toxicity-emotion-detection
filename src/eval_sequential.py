# eval_sequential.py
"""
Evaluation script for Sequential training models.
Evaluates sequential models on both toxicity and emotion test sets.
Includes emotion-toxicity correspondence analysis.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import os
import sys
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

# Import correspondence analysis from FlexibleModel
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'FlexibleModel'))
from evaluation import analyze_emotion_toxicity_correlation, get_emotion_labels, get_toxicity_labels

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_emotion_labels_fixed():
    """Load emotion labels from emotion_map.txt - fixed version."""
    try:
        emotion_map_path = "data/emotion_map.txt"
        emotion_labels = []
        with open(emotion_map_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Try tab-separated first
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        emotion_labels.append(parts[1])
                    else:
                        # Just use the line itself
                        emotion_labels.append(line)

        # If we got labels, return them
        if emotion_labels:
            return emotion_labels
        else:
            raise ValueError("No labels loaded")

    except Exception as e:
        logger.warning(f"Could not load emotion labels from file: {e}")
        # Return default 28 emotion labels
        return [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
            'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ]


def evaluate_toxicity(model, test_loader, device, return_predictions=False):
    """
    Evaluate sequential model on toxicity test set.

    Args:
        model: MultiTaskBERT model
        test_loader: Toxicity test dataloader
        device: Device to run on
        return_predictions: If True, return predictions and labels

    Returns:
        dict: Toxicity metrics
        (optional) tuple: (predictions, labels) if return_predictions=True
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

    if return_predictions:
        return metrics, all_preds, all_labels
    return metrics


def evaluate_emotion(model, test_loader, device, return_predictions=False):
    """
    Evaluate sequential model on emotion test set.

    Args:
        model: MultiTaskBERT model
        test_loader: Emotion test dataloader
        device: Device to run on
        return_predictions: If True, return predictions and labels

    Returns:
        dict: Emotion metrics
        (optional) tuple: (predictions, labels) if return_predictions=True
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

    if return_predictions:
        return metrics, all_preds, all_labels
    return metrics


def evaluate_both_tasks_on_toxicity(model, test_loader, device):
    """
    Evaluate both toxicity AND emotion predictions on the toxicity test set.
    This allows for correspondence analysis between the two tasks.

    Args:
        model: MultiTaskBERT model
        test_loader: Toxicity test dataloader
        device: Device to run on

    Returns:
        tuple: (tox_preds, emo_preds) - predictions for both tasks on same samples
    """
    model.eval()
    model.to(device)

    all_tox_preds = []
    all_emo_preds = []

    logger.info("Getting both toxicity and emotion predictions for correspondence analysis...")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Both Tasks"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass - get BOTH toxicity and emotion logits
            tox_logits, emo_logits = model(input_ids, attention_mask)

            # Get predictions for both tasks
            tox_probs = torch.sigmoid(tox_logits).cpu().numpy()
            tox_preds = (tox_probs > 0.5).astype(int)

            emo_probs = torch.sigmoid(emo_logits).cpu().numpy()
            emo_preds = (emo_probs > 0.5).astype(int)

            all_tox_preds.extend(tox_preds)
            all_emo_preds.extend(emo_preds)

    # Convert to numpy
    all_tox_preds = np.array(all_tox_preds)
    all_emo_preds = np.array(all_emo_preds)

    return all_tox_preds, all_emo_preds


def main():
    """Main evaluation function."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file")
    parser.add_argument("--output", type=str, default="../results/sequential_eval_results.csv",
                        help="Path to save results CSV")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--eval_emotion", action="store_true",
                        help="Also evaluate on emotion test set and run correspondence analysis")
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
        is_weighted = checkpoint.get('weighted', tox_weights is not None or emo_weights is not None)
        is_balanced = checkpoint.get('balanced', False)
        training_mode = checkpoint.get('training_mode', 'sequential')

        logger.info(f"Model type: Sequential")
        logger.info(f"Training mode: {training_mode}")
        logger.info(f"Balanced: {is_balanced}")
        logger.info(f"Weighted: {is_weighted}")
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

        # Evaluate toxicity (with predictions if we need correspondence analysis)
        logger.info("Starting toxicity evaluation...")
        if args.eval_emotion:
            metrics, tox_preds, tox_labels = evaluate_toxicity(model, tox_test_loader, DEVICE, return_predictions=True)
        else:
            metrics = evaluate_toxicity(model, tox_test_loader, DEVICE, return_predictions=False)

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

            logger.info("Starting emotion evaluation on emotion test set...")
            emo_metrics = evaluate_emotion(model, emo_test_loader, DEVICE, return_predictions=False)
            metrics.update(emo_metrics)

            # Print emotion results
            logger.info("\n" + "="*60)
            logger.info("EMOTION EVALUATION RESULTS (on emotion test set)")
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

            # Run emotion-toxicity correspondence analysis on toxicity test set
            # Get both predictions from the SAME samples (toxicity test set)
            logger.info("\nRunning Emotion-Toxicity Correspondence Analysis...")
            logger.info("(Using toxicity test set for both predictions)")
            tox_preds_corr, emo_preds_corr = evaluate_both_tasks_on_toxicity(model, tox_test_loader, DEVICE)

            emotion_labels = get_emotion_labels_fixed()
            toxicity_labels = get_toxicity_labels()

            correlations = analyze_emotion_toxicity_correlation(
                emo_preds_corr,
                tox_preds_corr,
                emotion_labels=emotion_labels,
                toxicity_labels=toxicity_labels
            )

        # Save results
        results_df = pd.DataFrame([{
            'checkpoint': args.checkpoint,
            'model_type': 'sequential',
            'training_mode': training_mode,
            'balanced': is_balanced,
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
