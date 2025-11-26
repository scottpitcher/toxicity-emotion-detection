# multi_train.py
"""
Multi-task training script: Trains BERT on toxicity and emotion jointly.
Uses separate dataloaders and alternates between tasks.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import os
import logging
from train_utils import load_toxicity_data, load_emotion_data, train_multitask
from model import MultiTaskBERT
import argparse
from compute_weights import load_class_weights

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training function for multi-task."""
    
    # Hyperparameters
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    MODEL_SAVE_PATH = "../models/multi/multitask_bert.pt"
    GRADIENT_CLIP = 1.0
    LAMBDA_TOX = 1.0
    LAMBDA_EMO = 1.0

    # Whether to use sampled or non-sampled class weights and data
    tox_sampled_bool = False
    emo_sampled_bool = True
    
    DEVICE = "cuda" if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {DEVICE}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--weighted", type=str, default="False",
                        help="True/False — whether to use class-weighted loss.")
    args = parser.parse_args()

    # normalize string to bool
    weighted_flag = args.weighted.lower() == "true"


    ## CLASS WEIGHT LOGIC ##
    if weighted_flag:
        logger.info("Loading class weights for toxicity labels...")

        # toxic should use non-sampled 
        tox_weights = load_class_weights(task="toxicity", sampled=tox_sampled_bool)
        emo_weights = load_class_weights(task="emotion", sampled=emo_sampled_bool)

        logger.info(f"Class weights for tox: {tox_weights}")
        logger.info(f"Class weights for emo: {emo_weights}")

    else:
        tox_weights = None
        emo_weights = None
        logger.info("Not using class-weighted loss.")

    
    try:
        # Load both datasets
        logger.info("Loading toxicity and emotion data...")
        tox_train, tox_val = load_toxicity_data(tox_sampled_bool, batch_size=BATCH_SIZE)
        emo_train, emo_val = load_emotion_data(emo_sampled_bool, batch_size=BATCH_SIZE)
        logger.info("Data loading complete.")
        
        # Initialize multi-task model
        logger.info("Initializing MultiTaskBERT model...")
        model = MultiTaskBERT(
            tox_class_weights=tox_weights,
            emo_class_weights=emo_weights,
            lambda_tox=LAMBDA_TOX,
            lambda_emo=LAMBDA_EMO
        )
        
        # Train model
        logger.info("Starting multi-task training...")
        model = train_multitask(
            model,
            tox_train,
            tox_val,
            emo_train,
            emo_val,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            device=DEVICE,
            save_path=MODEL_SAVE_PATH,
            gradient_clip=GRADIENT_CLIP
        )

        # Prepare checkpoint
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "tox_class_weights": tox_weights,
            "emo_class_weights": emo_weights,
            "lambda_tox": LAMBDA_TOX,
            "lambda_emo": LAMBDA_EMO,
            "epoch": EPOCHS
        }

        # Try normal save
        try:
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(checkpoint, MODEL_SAVE_PATH)
            logger.info(f"Saved checkpoint to {MODEL_SAVE_PATH}")

        # Fallback: save without "../" prefix (for Colab odd paths)
        except Exception:
            alt_path = MODEL_SAVE_PATH.replace("../", "")
            os.makedirs(os.path.dirname(alt_path), exist_ok=True)
            torch.save(checkpoint, alt_path)
            logger.info(f"Saved checkpoint to fallback path {alt_path}")




        
        logger.info(f"\nMulti-task training complete! Model saved to {MODEL_SAVE_PATH}")
        
    except FileNotFoundError as e:
        logger.error(f"Data loading error: {e}")
        raise
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise


if __name__ == "__main__":
    main()