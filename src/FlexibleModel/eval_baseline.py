# eval_baseline.py
"""
Evaluation script for Experiment 1: Baseline (Toxicity-only classification)
Loads trained baseline model and evaluates on test set.
"""

import torch
import logging
import argparse

from flexible_model import FlexibleBERT
from train_utils import load_toxicity_test_data
from evaluation import compute_metrics, evaluate_model, get_toxicity_labels

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Baseline Experiment')
    parser.add_argument("--model_path", type=str, default="models/baseline_unbalanced.pt",
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
        test_loader = load_toxicity_test_data(batch_size=args.batch_size)

        # Initialize model
        logger.info(f"Loading model from {args.model_path}...")
        model = FlexibleBERT(mode='baseline')

        # Load trained weights
        checkpoint = torch.load(args.model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)

        logger.info("Model loaded successfully!")

        # Evaluate
        logger.info("\nEvaluating on test set...")
        predictions, labels = evaluate_model(model, test_loader, DEVICE, task="toxicity")

        # Get toxicity class names
        toxicity_labels = get_toxicity_labels()

        # Compute and display metrics
        metrics = compute_metrics(
            labels,
            predictions,
            class_names=toxicity_labels,
            task_name="Baseline - Toxicity Classification"
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
