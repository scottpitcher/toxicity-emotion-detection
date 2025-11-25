# experiment_multitask.py
"""
Experiment 2: Multitask Learning
Jointly trains BERT on both toxicity and emotion classification with combined loss.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import os
import logging
import argparse
from pathlib import Path

from flexible_model import FlexibleBERT
from train_utils import load_toxicity_data, load_emotion_data
from compute_weights import load_class_weights
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_multitask_experiment(
    model,
    tox_train_loader,
    tox_val_loader,
    emo_train_loader,
    emo_val_loader,
    epochs=3,
    lr=2e-5,
    device="cuda",
    save_path="models/experiment_multitask.pt",
    gradient_clip=1.0
):
    """Train multitask FlexibleBERT model on both toxicity and emotion."""

    optimizer = AdamW(model.parameters(), lr=lr)
    steps_per_epoch = max(len(tox_train_loader), len(emo_train_loader))
    total_steps = steps_per_epoch * epochs

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
        total_tox_loss = 0
        total_emo_loss = 0
        total_joint_loss = 0

        logger.info(f"\nEpoch {epoch+1}/{epochs}")

        # Create iterators
        tox_iter = iter(tox_train_loader)
        emo_iter = iter(emo_train_loader)

        progress_bar = tqdm(range(steps_per_epoch), desc="Training")
        for step in progress_bar:

            # Train on toxicity batch
            try:
                tox_batch = next(tox_iter)

                optimizer.zero_grad()

                input_ids = tox_batch["input_ids"].to(device)
                attention_mask = tox_batch["attention_mask"].to(device)
                tox_labels = tox_batch["labels"].to(device)

                # Get both outputs but only compute toxicity loss
                tox_logits, _ = model(input_ids, attention_mask)
                loss = model.lambda_tox * model.compute_loss(tox_logits, tox_labels)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
                scheduler.step()

                total_tox_loss += loss.item()

            except StopIteration:
                tox_iter = iter(tox_train_loader)

            # Train on emotion batch
            try:
                emo_batch = next(emo_iter)

                optimizer.zero_grad()

                input_ids = emo_batch["input_ids"].to(device)
                attention_mask = emo_batch["attention_mask"].to(device)
                emo_labels = emo_batch["labels"].to(device)

                # Get both outputs but only compute emotion loss
                _, emo_logits = model(input_ids, attention_mask)
                loss = model.lambda_emo * model.compute_loss(None, None, emo_logits, emo_labels)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
                scheduler.step()

                total_emo_loss += loss.item()

            except StopIteration:
                emo_iter = iter(emo_train_loader)

            progress_bar.set_postfix({
                'tox_loss': f'{total_tox_loss/(step+1):.4f}',
                'emo_loss': f'{total_emo_loss/(step+1):.4f}'
            })

        avg_train_tox_loss = total_tox_loss / steps_per_epoch
        avg_train_emo_loss = total_emo_loss / steps_per_epoch
        logger.info(f"Train Loss - Tox: {avg_train_tox_loss:.4f}, Emo: {avg_train_emo_loss:.4f}")

        # Validation phase
        model.eval()
        val_tox_loss = 0
        val_emo_loss = 0

        with torch.no_grad():
            # Validate toxicity
            for batch in tqdm(tox_val_loader, desc="Val Toxicity"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                tox_logits, _ = model(input_ids, attention_mask)
                loss = model.compute_loss(tox_logits, labels)
                val_tox_loss += loss.item()

            # Validate emotion
            for batch in tqdm(emo_val_loader, desc="Val Emotion"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                _, emo_logits = model(input_ids, attention_mask)
                loss = model.compute_loss(None, None, emo_logits, labels)
                val_emo_loss += loss.item()

        avg_val_tox_loss = val_tox_loss / len(tox_val_loader)
        avg_val_emo_loss = val_emo_loss / len(emo_val_loader)
        avg_val_loss = (val_tox_loss + val_emo_loss) / (len(tox_val_loader) + len(emo_val_loader))

        logger.info(f"Val Loss - Total: {avg_val_loss:.4f}, Tox: {avg_val_tox_loss:.4f}, Emo: {avg_val_emo_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_tox_loss': avg_train_tox_loss,
                'train_emo_loss': avg_train_emo_loss,
                'val_tox_loss': avg_val_tox_loss,
                'val_emo_loss': avg_val_emo_loss,
                'val_loss': avg_val_loss,
                'mode': 'multitask',
                'lambda_tox': model.lambda_tox,
                'lambda_emo': model.lambda_emo
            }, save_path)
            logger.info(f"Saved best model (val_loss: {avg_val_loss:.4f})")

    return model


def main():
    """Main function for multitask experiment."""

    parser = argparse.ArgumentParser(description='Multitask Experiment: Joint toxicity and emotion classification')
    parser.add_argument("--weighted", type=str, default="False",
                        help="True/False - whether to use class-weighted loss")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--lambda_tox", type=float, default=1.0,
                        help="Weight for toxicity loss")
    parser.add_argument("--lambda_emo", type=float, default=1.0,
                        help="Weight for emotion loss")
    parser.add_argument("--save_path", type=str, default="models/experiment_multitask.pt",
                        help="Path to save model")
    args = parser.parse_args()

    # Parse weighted flag
    weighted_flag = args.weighted.lower() == "true"

    # Class weights
    if weighted_flag:
        logger.info("Loading class weights...")
        tox_class_weights = load_class_weights(task="toxicity", sampled=False)
        emo_class_weights = load_class_weights(task="emotion", sampled=True)
        logger.info(f"Toxicity class weights: {tox_class_weights}")
        logger.info(f"Emotion class weights: {emo_class_weights}")
    else:
        tox_class_weights = None
        emo_class_weights = None
        logger.info("Not using class-weighted loss.")

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {DEVICE}")

    try:
        # Load both toxicity and emotion data
        logger.info("Loading toxicity and emotion data...")
        tox_train_loader, tox_val_loader = load_toxicity_data(
            data_root="data/processed/tokenized",
            batch_size=args.batch_size
        )
        emo_train_loader, emo_val_loader = load_emotion_data(
            data_root="data/processed_sampled",
            batch_size=args.batch_size
        )

        # Initialize multitask FlexibleBERT model
        logger.info("Initializing FlexibleBERT in multitask mode...")
        model = FlexibleBERT(
            mode='multitask',
            lambda_tox=args.lambda_tox,
            lambda_emo=args.lambda_emo,
            tox_class_weights=tox_class_weights,
            emo_class_weights=emo_class_weights
        )

        # Train model
        logger.info("Starting multitask experiment training...")
        model = train_multitask_experiment(
            model,
            tox_train_loader,
            tox_val_loader,
            emo_train_loader,
            emo_val_loader,
            epochs=args.epochs,
            lr=args.lr,
            device=DEVICE,
            save_path=args.save_path,
            gradient_clip=1.0
        )

        logger.info(f"\nMultitask experiment complete! Model saved to {args.save_path}")

    except FileNotFoundError as e:
        logger.error(f"Data loading error: {e}")
        raise
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise


if __name__ == "__main__":
    main()
