# experiment_sequential.py
"""
Experiment 3: Sequential Learning
Phase 1: Pretrain BERT on emotion classification
Phase 2: Finetune on toxicity classification (optionally with frozen encoder)
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import os
import logging
import argparse
from pathlib import Path

from flexible_model import FlexibleBERT
from train_utils import load_emotion_data, load_toxicity_data
from compute_weights import load_class_weights
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_emotion_pretraining(
    model,
    train_loader,
    val_loader,
    epochs=3,
    lr=2e-5,
    device="cuda",
    save_path="models/sequential_emotion_pretrained.pt",
    gradient_clip=1.0
):
    """Phase 1: Pretrain on emotion classification."""

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    model.to(device)
    best_val_loss = float('inf')

    logger.info("=" * 60)
    logger.info("PHASE 1: EMOTION PRETRAINING")
    logger.info("=" * 60)
    logger.info(f"Trainable parameters: {model.get_trainable_params()}")

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0

        logger.info(f"\nEpoch {epoch+1}/{epochs}")

        progress_bar = tqdm(train_loader, desc="Training (Emotion)")
        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass - get emotion logits
            _, emo_logits = model(input_ids, attention_mask)

            # Compute emotion loss only
            loss = model.compute_loss(None, None, emo_logits, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{total_train_loss / (progress_bar.n + 1):.4f}'})

        avg_train_loss = total_train_loss / len(train_loader)
        logger.info(f"Train Loss (Emotion): {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating (Emotion)"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                _, emo_logits = model(input_ids, attention_mask)
                loss = model.compute_loss(None, None, emo_logits, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        logger.info(f"Val Loss (Emotion): {avg_val_loss:.4f}")

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
                'phase': 'emotion_pretraining',
                'mode': 'sequential'
            }, save_path)
            logger.info(f"Saved best emotion-pretrained model (val_loss: {avg_val_loss:.4f})")

    return model


def train_toxicity_finetuning(
    model,
    train_loader,
    val_loader,
    epochs=3,
    lr=2e-5,
    device="cuda",
    save_path="models/experiment_sequential.pt",
    gradient_clip=1.0,
    freeze_encoder=False
):
    """Phase 2: Finetune on toxicity classification."""

    # Optionally freeze encoder
    if freeze_encoder:
        logger.info("Freezing BERT encoder for toxicity finetuning...")
        model.freeze_encoder()
    else:
        logger.info("Encoder remains trainable during toxicity finetuning...")

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    model.to(device)
    best_val_loss = float('inf')

    logger.info("=" * 60)
    logger.info("PHASE 2: TOXICITY FINETUNING")
    logger.info("=" * 60)
    logger.info(f"Trainable parameters: {model.get_trainable_params()}")

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0

        logger.info(f"\nEpoch {epoch+1}/{epochs}")

        progress_bar = tqdm(train_loader, desc="Training (Toxicity)")
        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass - get toxicity logits
            tox_logits, _ = model(input_ids, attention_mask, return_emotion=False)

            # Compute toxicity loss only
            loss = model.compute_loss(tox_logits, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{total_train_loss / (progress_bar.n + 1):.4f}'})

        avg_train_loss = total_train_loss / len(train_loader)
        logger.info(f"Train Loss (Toxicity): {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating (Toxicity)"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                tox_logits, _ = model(input_ids, attention_mask, return_emotion=False)
                loss = model.compute_loss(tox_logits, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        logger.info(f"Val Loss (Toxicity): {avg_val_loss:.4f}")

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
                'phase': 'toxicity_finetuning',
                'mode': 'sequential',
                'freeze_encoder': freeze_encoder
            }, save_path)
            logger.info(f"Saved best finetuned model (val_loss: {avg_val_loss:.4f})")

    return model


def main():
    """Main function for sequential experiment."""

    parser = argparse.ArgumentParser(description='Sequential Experiment: Emotion pretraining -> Toxicity finetuning')
    parser.add_argument("--weighted", type=str, default="False",
                        help="True/False - whether to use class-weighted loss")
    parser.add_argument("--pretrain_epochs", type=int, default=3,
                        help="Number of emotion pretraining epochs")
    parser.add_argument("--finetune_epochs", type=int, default=3,
                        help="Number of toxicity finetuning epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--pretrain_lr", type=float, default=2e-5,
                        help="Learning rate for emotion pretraining")
    parser.add_argument("--finetune_lr", type=float, default=2e-5,
                        help="Learning rate for toxicity finetuning")
    parser.add_argument("--freeze_encoder", type=str, default="False",
                        help="True/False - whether to freeze encoder during toxicity finetuning")
    parser.add_argument("--pretrain_save_path", type=str, default="models/sequential_emotion_pretrained.pt",
                        help="Path to save emotion-pretrained model")
    parser.add_argument("--final_save_path", type=str, default="models/experiment_sequential.pt",
                        help="Path to save final finetuned model")
    args = parser.parse_args()

    # Parse flags
    weighted_flag = args.weighted.lower() == "true"
    freeze_encoder_flag = args.freeze_encoder.lower() == "true"

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
        # Initialize sequential FlexibleBERT model
        logger.info("Initializing FlexibleBERT in sequential mode...")
        model = FlexibleBERT(
            mode='sequential',
            tox_class_weights=tox_class_weights,
            emo_class_weights=emo_class_weights
        )

        # PHASE 1: Emotion Pretraining
        logger.info("\n" + "=" * 60)
        logger.info("Loading emotion data for pretraining...")
        emo_train_loader, emo_val_loader = load_emotion_data(
            data_root="data/processed_sampled",
            batch_size=args.batch_size
        )

        model = train_emotion_pretraining(
            model,
            emo_train_loader,
            emo_val_loader,
            epochs=args.pretrain_epochs,
            lr=args.pretrain_lr,
            device=DEVICE,
            save_path=args.pretrain_save_path,
            gradient_clip=1.0
        )

        logger.info(f"\nPhase 1 complete! Emotion-pretrained model saved to {args.pretrain_save_path}")

        # PHASE 2: Toxicity Finetuning
        logger.info("\n" + "=" * 60)
        logger.info("Loading toxicity data for finetuning...")
        tox_train_loader, tox_val_loader = load_toxicity_data(
            data_root="data/processed/tokenized",
            batch_size=args.batch_size
        )

        model = train_toxicity_finetuning(
            model,
            tox_train_loader,
            tox_val_loader,
            epochs=args.finetune_epochs,
            lr=args.finetune_lr,
            device=DEVICE,
            save_path=args.final_save_path,
            gradient_clip=1.0,
            freeze_encoder=freeze_encoder_flag
        )

        logger.info(f"\nPhase 2 complete! Final model saved to {args.final_save_path}")
        logger.info("\n" + "=" * 60)
        logger.info("SEQUENTIAL EXPERIMENT COMPLETE!")
        logger.info("=" * 60)

    except FileNotFoundError as e:
        logger.error(f"Data loading error: {e}")
        raise
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise


if __name__ == "__main__":
    main()
