# experiment_baseline.py
"""
Experiment 1: Baseline
Fine-tunes BERT on toxicity classification only (single-task learning).
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import os
import logging
import argparse
from pathlib import Path

from flexible_model import FlexibleBERT
from train_utils import load_toxicity_data
from compute_weights import load_class_weights
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_baseline_experiment(
    model,
    train_loader,
    val_loader,
    epochs=3,
    lr=2e-5,
    device="cuda",
    save_path="models/experiment_baseline.pt",
    gradient_clip=1.0
):
    """Train baseline FlexibleBERT model on toxicity only."""

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    model.to(device)
    best_val_loss = float('inf')

    logger.info(f"Trainable parameters: {model.get_trainable_params()}")

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0

        logger.info(f"\nEpoch {epoch+1}/{epochs}")

        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass - baseline returns (tox_logits, None)
            tox_logits, _ = model(input_ids, attention_mask)
            loss = model.compute_loss(tox_logits, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{total_train_loss / (progress_bar.n + 1):.4f}'})

        avg_train_loss = total_train_loss / len(train_loader)
        logger.info(f"Train Loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                tox_logits, _ = model(input_ids, attention_mask)
                loss = model.compute_loss(tox_logits, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        logger.info(f"Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'mode': 'baseline'
            }, save_path)
            logger.info(f"Saved best model (val_loss: {avg_val_loss:.4f})")

    return model


def main():
    """Main function for baseline experiment."""

    parser = argparse.ArgumentParser(description='Baseline Experiment: Toxicity-only classification')
    parser.add_argument("--weighted", type=str, default="False",
                        help="True/False - whether to use class-weighted loss")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--save_path", type=str, default="models/experiment_baseline.pt",
                        help="Path to save model")
    args = parser.parse_args()

    # Parse weighted flag
    weighted_flag = args.weighted.lower() == "true"

    # Class weights
    if weighted_flag:
        logger.info("Loading class weights for toxicity labels...")
        tox_class_weights = load_class_weights(task="toxicity", sampled=False)
        logger.info(f"Class weights: {tox_class_weights}")
    else:
        tox_class_weights = None
        logger.info("Not using class-weighted loss.")

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {DEVICE}")

    try:
        # Load toxicity data only
        logger.info("Loading toxicity data...")
        train_loader, val_loader = load_toxicity_data(batch_size=args.batch_size)

        # Initialize baseline FlexibleBERT model
        logger.info("Initializing FlexibleBERT in baseline mode...")
        model = FlexibleBERT(
            mode='baseline',
            tox_class_weights=tox_class_weights
        )

        # Train model
        logger.info("Starting baseline experiment training...")
        model = train_baseline_experiment(
            model,
            train_loader,
            val_loader,
            epochs=args.epochs,
            lr=args.lr,
            device=DEVICE,
            save_path=args.save_path,
            gradient_clip=1.0
        )

        logger.info(f"\nBaseline experiment complete! Model saved to {args.save_path}")

    except FileNotFoundError as e:
        logger.error(f"Data loading error: {e}")
        raise
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise


if __name__ == "__main__":
    main()
