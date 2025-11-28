# sequential_train.py
"""
Sequential training script: Trains MultiTaskBERT in two phases.
Phase 1: Pretrain on emotion classification
Phase 2: Fine-tune on toxicity classification
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import os
import logging
from train_utils import load_toxicity_data, load_emotion_data
from model import MultiTaskBERT
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import argparse
from compute_weights import load_class_weights

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_emotion_phase(
    model,
    train_loader,
    val_loader,
    epochs,
    lr,
    device,
    save_path,
    gradient_clip=1.0
):
    """
    Phase 1: Train on emotion classification only.

    Args:
        model: MultiTaskBERT model
        train_loader: Emotion training dataloader
        val_loader: Emotion validation dataloader
        epochs: Number of epochs
        lr: Learning rate
        device: Device to train on
        save_path: Path to save emotion-pretrained checkpoint
        gradient_clip: Gradient clipping value

    Returns:
        Trained model
    """
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

    for epoch in range(epochs):
        # Training
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
            loss = model.compute_emo_loss(emo_logits, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{total_train_loss / (progress_bar.n + 1):.4f}'})

        avg_train_loss = total_train_loss / len(train_loader)
        logger.info(f"Train Loss (Emotion): {avg_train_loss:.4f}")

        # Validation
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating (Emotion)"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                _, emo_logits = model(input_ids, attention_mask)
                loss = model.compute_emo_loss(emo_logits, labels)
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
                'phase': 'emotion_pretraining'
            }, save_path)
            logger.info(f"Saved emotion-pretrained checkpoint (val_loss: {avg_val_loss:.4f})")

    logger.info(f"\nPhase 1 complete! Best val loss: {best_val_loss:.4f}")
    return model


def train_toxicity_phase(
    model,
    train_loader,
    val_loader,
    epochs,
    lr,
    device,
    save_path,
    gradient_clip=1.0
):
    """
    Phase 2: Fine-tune on toxicity classification.

    Args:
        model: MultiTaskBERT model (emotion-pretrained)
        train_loader: Toxicity training dataloader
        val_loader: Toxicity validation dataloader
        epochs: Number of epochs
        lr: Learning rate
        device: Device to train on
        save_path: Path to save final checkpoint
        gradient_clip: Gradient clipping value

    Returns:
        Trained model
    """
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

    for epoch in range(epochs):
        # Training
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
            tox_logits, _ = model(input_ids, attention_mask)

            # Compute toxicity loss only
            loss = model.compute_tox_loss(tox_logits, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{total_train_loss / (progress_bar.n + 1):.4f}'})

        avg_train_loss = total_train_loss / len(train_loader)
        logger.info(f"Train Loss (Toxicity): {avg_train_loss:.4f}")

        # Validation
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating (Toxicity)"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                tox_logits, _ = model(input_ids, attention_mask)
                loss = model.compute_tox_loss(tox_logits, labels)
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
                'phase': 'toxicity_finetuning'
            }, save_path)
            logger.info(f"Saved finetuned checkpoint (val_loss: {avg_val_loss:.4f})")

    logger.info(f"\nPhase 2 complete! Best val loss: {best_val_loss:.4f}")
    return model


def main():
    """Main training function for sequential learning."""

    parser = argparse.ArgumentParser(description='Sequential Training: Emotion -> Toxicity')
    parser.add_argument("--balanced", type=str, default="False",
                        help="True/False - use balanced datasets for both tasks")
    parser.add_argument("--weighted", type=str, default="False",
                        help="True/False - use class-weighted loss")
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
    args = parser.parse_args()

    # Parse flags
    balanced_flag = args.balanced.lower() == "true"
    weighted_flag = args.weighted.lower() == "true"

    # Set sampled flags
    # If balanced=True, use sampled data for both
    # If balanced=False, use non-sampled for both
    tox_sampled_bool = balanced_flag
    emo_sampled_bool = balanced_flag

    # Hyperparameters
    GRADIENT_CLIP = 1.0
    LAMBDA_TOX = 1.0
    LAMBDA_EMO = 1.0

    DEVICE = "cuda" if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {DEVICE}")

    # Set save paths
    balanced_suffix = "_balanced" if balanced_flag else "_unbalanced"
    weighted_suffix = "_weighted" if weighted_flag else ""

    EMOTION_SAVE_PATH = f"../models/sequential/sequential_emotion_pretrained{balanced_suffix}{weighted_suffix}.pt"
    FINAL_SAVE_PATH = f"../models/sequential/sequential_bert{balanced_suffix}{weighted_suffix}.pt"

    # Load class weights if needed
    if weighted_flag:
        logger.info("Loading class weights...")
        tox_weights = load_class_weights(task="toxicity", sampled=tox_sampled_bool)
        emo_weights = load_class_weights(task="emotion", sampled=emo_sampled_bool)
        logger.info(f"Toxicity class weights: {tox_weights}")
        logger.info(f"Emotion class weights: {emo_weights}")
    else:
        tox_weights = None
        emo_weights = None
        logger.info("Not using class-weighted loss.")

    logger.info(f"Data mode: {'Balanced' if balanced_flag else 'Unbalanced'}")

    try:
        # Initialize MultiTaskBERT model
        logger.info("Initializing MultiTaskBERT model...")
        model = MultiTaskBERT(
            tox_class_weights=tox_weights,
            emo_class_weights=emo_weights,
            lambda_tox=LAMBDA_TOX,
            lambda_emo=LAMBDA_EMO
        )

        # PHASE 1: Emotion Pretraining
        logger.info("\n" + "=" * 60)
        logger.info("Loading emotion data for pretraining...")
        emo_train_loader, emo_val_loader = load_emotion_data(
            emo_sampled_bool=emo_sampled_bool,
            batch_size=args.batch_size
        )

        model = train_emotion_phase(
            model,
            emo_train_loader,
            emo_val_loader,
            epochs=args.pretrain_epochs,
            lr=args.pretrain_lr,
            device=DEVICE,
            save_path=EMOTION_SAVE_PATH,
            gradient_clip=GRADIENT_CLIP
        )

        logger.info(f"\nEmotion-pretrained checkpoint saved to {EMOTION_SAVE_PATH}")

        # PHASE 2: Toxicity Finetuning
        logger.info("\n" + "=" * 60)
        logger.info("Loading toxicity data for finetuning...")
        tox_train_loader, tox_val_loader = load_toxicity_data(
            tox_sampled_bool=tox_sampled_bool,
            batch_size=args.batch_size
        )

        model = train_toxicity_phase(
            model,
            tox_train_loader,
            tox_val_loader,
            epochs=args.finetune_epochs,
            lr=args.finetune_lr,
            device=DEVICE,
            save_path=FINAL_SAVE_PATH,
            gradient_clip=GRADIENT_CLIP
        )

        # Save final checkpoint with metadata
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "tox_class_weights": tox_weights,
            "emo_class_weights": emo_weights,
            "lambda_tox": LAMBDA_TOX,
            "lambda_emo": LAMBDA_EMO,
            "pretrain_epochs": args.pretrain_epochs,
            "finetune_epochs": args.finetune_epochs,
            "balanced": balanced_flag,
            "weighted": weighted_flag,
            "training_mode": "sequential"
        }

        # Try normal save
        try:
            os.makedirs(os.path.dirname(FINAL_SAVE_PATH), exist_ok=True)
            torch.save(checkpoint, FINAL_SAVE_PATH)
            logger.info(f"\nFinal checkpoint saved to {FINAL_SAVE_PATH}")
        except Exception:
            # Fallback: save without "../" prefix
            alt_path = FINAL_SAVE_PATH.replace("../", "")
            os.makedirs(os.path.dirname(alt_path), exist_ok=True)
            torch.save(checkpoint, alt_path)
            logger.info(f"\nFinal checkpoint saved to {alt_path}")

        logger.info("\n" + "=" * 60)
        logger.info("SEQUENTIAL TRAINING COMPLETE!")
        logger.info("=" * 60)

    except FileNotFoundError as e:
        logger.error(f"Data loading error: {e}")
        raise
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise


if __name__ == "__main__":
    main()
