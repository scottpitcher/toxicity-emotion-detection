# evaluation.py
"""
Evaluation utilities for toxicity and emotion classification models.
Includes metrics calculation and emotion-toxicity correlation analysis.
"""

import torch
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)
import logging

logger = logging.getLogger(__name__)


def compute_metrics(y_true, y_pred, class_names=None, task_name="Classification"):
    """
    Compute classification metrics for multi-label tasks.

    Args:
        y_true: Ground truth labels (numpy array or tensor) [batch, num_classes]
        y_pred: Predicted labels (numpy array or tensor) [batch, num_classes]
        class_names: List of class names for per-class metrics
        task_name: Name of the task for logging

    Returns:
        Dictionary of metrics
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    # Ensure binary labels
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    # Overall metrics
    metrics = {
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
    }

    # Per-class F1 scores
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    metrics['per_class_f1'] = per_class_f1

    # Print metrics
    logger.info(f"\n{'='*60}")
    logger.info(f"{task_name} Metrics")
    logger.info(f"{'='*60}")
    logger.info(f"Macro F1:        {metrics['f1_macro']:.4f}")
    logger.info(f"Macro Precision: {metrics['precision_macro']:.4f}")
    logger.info(f"Macro Recall:    {metrics['recall_macro']:.4f}")
    logger.info(f"Weighted F1:     {metrics['f1_weighted']:.4f}")
    logger.info(f"Accuracy:        {metrics['accuracy']:.4f}")

    logger.info(f"\nPer-Class F1 Scores:")
    logger.info(f"{'-'*60}")
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(len(per_class_f1))]

    for name, score in zip(class_names, per_class_f1):
        logger.info(f"  {name:20s}: {score:.4f}")
    logger.info(f"{'='*60}\n")

    return metrics


def evaluate_model(model, data_loader, device, task="toxicity"):
    """
    Evaluate model on a dataset and return predictions and ground truth.

    Args:
        model: The model to evaluate
        data_loader: DataLoader for the dataset
        device: Device to run evaluation on
        task: "toxicity", "emotion", or "both"

    Returns:
        Dictionary with predictions and labels
    """
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Get predictions based on task
            tox_logits, emo_logits = model(input_ids, attention_mask)

            if task == "toxicity":
                logits = tox_logits
            elif task == "emotion":
                logits = emo_logits
            else:
                raise ValueError(f"Task must be 'toxicity' or 'emotion', got {task}")

            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    predictions = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return predictions, labels


def evaluate_both_tasks(model, tox_loader, emo_loader, device):
    """
    Evaluate model on both toxicity and emotion tasks.

    Args:
        model: The model to evaluate
        tox_loader: DataLoader for toxicity data
        emo_loader: DataLoader for emotion data
        device: Device to run evaluation on

    Returns:
        Dictionary with predictions and labels for both tasks
    """
    model.eval()

    # Evaluate toxicity
    tox_preds, tox_labels = evaluate_model(model, tox_loader, device, task="toxicity")

    # Evaluate emotion
    emo_preds, emo_labels = evaluate_model(model, emo_loader, device, task="emotion")

    return {
        'toxicity': {'predictions': tox_preds, 'labels': tox_labels},
        'emotion': {'predictions': emo_preds, 'labels': emo_labels}
    }


def analyze_emotion_toxicity_correlation(emo_preds, tox_preds, emotion_labels=None, toxicity_labels=None):
    """
    Analyze which emotions contribute most to toxic commentary.

    Args:
        emo_preds: Emotion predictions [batch, num_emotions]
        tox_preds: Toxicity predictions [batch, num_toxicity_types]
        emotion_labels: List of emotion label names
        toxicity_labels: List of toxicity label names

    Returns:
        Correlation statistics
    """
    # Convert to numpy if needed
    if isinstance(emo_preds, torch.Tensor):
        emo_preds = emo_preds.cpu().numpy()
    if isinstance(tox_preds, torch.Tensor):
        tox_preds = tox_preds.cpu().numpy()

    # Compute which samples are toxic (any toxicity label = 1)
    is_toxic = (tox_preds.sum(axis=1) > 0).astype(float)
    num_toxic = is_toxic.sum()
    num_total = len(is_toxic)

    num_emotions = emo_preds.shape[1]

    if emotion_labels is None:
        emotion_labels = [f"Emotion_{i}" for i in range(num_emotions)]

    if toxicity_labels is None:
        toxicity_labels = [f"Toxicity_{i}" for i in range(tox_preds.shape[1])]

    logger.info(f"\n{'='*80}")
    logger.info(f"Emotion-Toxicity Correspondence Analysis")
    logger.info(f"{'='*80}")
    logger.info(f"Total samples: {num_total}, Toxic samples: {int(num_toxic)} ({num_toxic/num_total*100:.1f}%)")
    logger.info(f"\n{'Emotion':<25} {'Correlation':>12} {'% in Toxic':>12} {'Count in Toxic':>15}")
    logger.info(f"{'-'*80}")

    correlations = []

    for i in range(num_emotions):
        emotion_presence = emo_preds[:, i]

        # Calculate correlation with overall toxicity
        if num_toxic > 0:
            corr = np.corrcoef(emotion_presence, is_toxic)[0, 1]

            # Calculate percentage of toxic samples with this emotion
            toxic_mask = is_toxic == 1
            count_in_toxic = emotion_presence[toxic_mask].sum()
            pct_in_toxic = (count_in_toxic / num_toxic) * 100
        else:
            corr = 0
            pct_in_toxic = 0
            count_in_toxic = 0

        correlations.append({
            'emotion': emotion_labels[i],
            'correlation': corr,
            'pct_in_toxic': pct_in_toxic,
            'count_in_toxic': int(count_in_toxic)
        })

        logger.info(f"{emotion_labels[i]:<25} {corr:>12.4f} {pct_in_toxic:>11.1f}% {int(count_in_toxic):>15}")

    # Sort by correlation and show top emotions
    correlations_sorted = sorted(correlations, key=lambda x: x['correlation'], reverse=True)

    logger.info(f"\n{'='*80}")
    logger.info(f"Top 10 Emotions Most Correlated with Toxicity:")
    logger.info(f"{'='*80}")
    for i, item in enumerate(correlations_sorted[:10], 1):
        logger.info(f"{i:2d}. {item['emotion']:<25} (r={item['correlation']:.4f}, {item['pct_in_toxic']:.1f}% of toxic)")

    logger.info(f"{'='*80}\n")

    return correlations_sorted


def get_emotion_labels():
    """
    Load emotion labels from emotion_map.txt.

    Returns:
        List of emotion label names
    """
    try:
        emotion_map_path = "data/emotion_map.txt"
        emotion_labels = []
        with open(emotion_map_path, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        emotion_labels.append(parts[1])
        return emotion_labels
    except Exception as e:
        logger.warning(f"Could not load emotion labels: {e}")
        return None


def get_toxicity_labels():
    """
    Return toxicity label names.

    Returns:
        List of toxicity label names
    """
    return ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
