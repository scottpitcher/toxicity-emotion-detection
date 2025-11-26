# baseline_train.py
"""
Baseline training script: Fine-tunes BERT on toxicity classification only.
Single-task learning on toxicity labels.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import os
import logging
from train_utils import load_toxicity_data, train_baseline
from model import BaselineBERT
import argparse
import pandas as pd
from pathlib import Path
from compute_weights import load_class_weights


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training function for baseline."""
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--weighted", type=str, default="False",
                        help="True/False — whether to use class-weighted loss.")
    args = parser.parse_args()

    # normalize string to bool
    weighted_flag = args.weighted.lower() == "true"


    ## CLASS WEIGHT LOGIC ##
    if weighted_flag:
        logger.info("Loading class weights for toxicity labels...")

        # baseline should always use the NON-SAMPLED weights
        class_weights = load_class_weights(task="toxicity", sampled=False)

        logger.info(f"Class weights: {class_weights}")

    else:
        class_weights = None
        logger.info("Not using class-weighted loss.")


    ## TRAIN MODEL ##
    # Hyperparameters
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5

    # Set save path based on weighted flag
    if weighted_flag:
        MODEL_SAVE_PATH = "../models/baseline/baseline_bert_weighted.pt"
    else:
        MODEL_SAVE_PATH = "../models/baseline/baseline_bert.pt"

    GRADIENT_CLIP = 1.0
    
    DEVICE = "cuda" if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {DEVICE}")

    try:
        # Load toxicity data only
        logger.info("Loading toxicity data...")
        train_loader, val_loader = load_toxicity_data(batch_size=BATCH_SIZE)
        
        # Initialize baseline model
        logger.info("Initializing BaselineBERT model...")
        model = BaselineBERT(class_weights=class_weights)
        
        # Train model
        logger.info("Starting baseline training...")
        model = train_baseline(
            model,
            train_loader,
            val_loader,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            device=DEVICE,
            save_path=MODEL_SAVE_PATH,
            gradient_clip=GRADIENT_CLIP
        )
        
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "class_weights": class_weights,
            "epochs": EPOCHS,
            "lr": LEARNING_RATE
        }
        # First attempt: save normally
        try:
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(checkpoint, MODEL_SAVE_PATH)
            logger.info(f"Saved checkpoint to {MODEL_SAVE_PATH}")

        # If that fails: try without "../"
        except Exception:
            alt_path = MODEL_SAVE_PATH.replace("../", "")
            os.makedirs(os.path.dirname(alt_path), exist_ok=True)
            torch.save(checkpoint, alt_path)
            logger.info(f"Saved checkpoint to fallback path {alt_path}")

        logger.info(f"Baseline training complete!")
                
    except FileNotFoundError as e:
        logger.error(f"Data loading error: {e}")
        raise
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise


if __name__ == "__main__":
    main()