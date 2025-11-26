# eval_multitask.py
"""
Evaluation script for Experiment 2: Multitask Learning
Loads trained multitask model and evaluates on both toxicity and emotion test sets.
Includes emotion-toxicity correspondence analysis.
"""

import torch
import logging
import argparse

from flexible_model import FlexibleBERT
from train_utils import load_toxicity_test_data, load_emotion_test_data
from evaluation import (
    compute_metrics,
    evaluate_model,
    analyze_emotion_toxicity_correlation,
    get_toxicity_labels,
    get_emotion_labels
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Multitask Experiment')
    parser.add_argument("--model_path", type=str, default="models/multitask_unbalanced.pt",
                        help="Path to trained model")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for evaluation")
    args = parser.parse_args()

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {DEVICE}")

    try:
        # Load test data
        logger.info("Loading test data...")
        tox_test_loader = load_toxicity_test_data(batch_size=args.batch_size)
        emo_test_loader = load_emotion_test_data(
            data_root="data/processed_sampled",
            batch_size=args.batch_size
        )

        # Initialize model
        logger.info(f"Loading model from {args.model_path}...")
        model = FlexibleBERT(mode='multitask')

        # Load trained weights
        checkpoint = torch.load(args.model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)

        logger.info("Model loaded successfully!")

        # Get class names
        toxicity_labels = get_toxicity_labels()
        emotion_labels = get_emotion_labels()

        # Evaluate on toxicity test set
        logger.info("\n" + "="*80)
        logger.info("Evaluating on Toxicity Test Set...")
        logger.info("="*80)
        tox_predictions, tox_labels = evaluate_model(model, tox_test_loader, DEVICE, task="toxicity")

        tox_metrics = compute_metrics(
            tox_labels,
            tox_predictions,
            class_names=toxicity_labels,
            task_name="Multitask - Toxicity Classification"
        )

        # Evaluate on emotion test set
        logger.info("\n" + "="*80)
        logger.info("Evaluating on Emotion Test Set...")
        logger.info("="*80)
        emo_predictions, emo_labels = evaluate_model(model, emo_test_loader, DEVICE, task="emotion")

        emo_metrics = compute_metrics(
            emo_labels,
            emo_predictions,
            class_names=emotion_labels,
            task_name="Multitask - Emotion Classification"
        )

        # Emotion-Toxicity Correspondence Analysis
        logger.info("\n" + "="*80)
        logger.info("Running Emotion-Toxicity Correspondence Analysis...")
        logger.info("="*80)
        analyze_emotion_toxicity_correlation(
            emo_predictions,
            tox_predictions,
            emotion_labels=emotion_labels,
            toxicity_labels=toxicity_labels
        )

        logger.info("\nEvaluation complete!")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise


if __name__ == "__main__":
    main()
