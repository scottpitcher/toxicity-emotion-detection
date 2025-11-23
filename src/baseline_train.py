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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training function for baseline."""
    
    # Hyperparameters
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    MODEL_SAVE_PATH = "models/baseline_bert.pt"
    GRADIENT_CLIP = 1.0
    
    DEVICE = "cuda" if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {DEVICE}")
    
    try:
        # Load toxicity data only
        logger.info("Loading toxicity data...")
        train_loader, val_loader = load_toxicity_data(batch_size=BATCH_SIZE)
        
        # Initialize baseline model
        logger.info("Initializing BaselineBERT model...")
        model = BaselineBERT()
        
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
        
        logger.info(f"\nBaseline training complete! Model saved to {MODEL_SAVE_PATH}")
        
    except FileNotFoundError as e:
        logger.error(f"Data loading error: {e}")
        raise
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise


if __name__ == "__main__":
    main()