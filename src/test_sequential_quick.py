# test_sequential_quick.py
"""
Quick test script for sequential training.
Trains for minimal epochs with small batch size to verify everything works.
Safe to run in Colab or locally.
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def quick_train_phase(model, train_loader, val_loader, epochs, lr, device, phase_name="Training"):
    """Quick training phase for testing."""
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    model.to(device)

    logger.info(f"\n{'='*60}")
    logger.info(f"{phase_name}")
    logger.info(f"{'='*60}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        logger.info(f"Epoch {epoch+1}/{epochs}")

        for batch in tqdm(train_loader, desc=f"{phase_name}"):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Determine which head to use based on phase
            if "Emotion" in phase_name:
                _, logits = model(input_ids, attention_mask)
                loss = model.compute_emo_loss(logits, labels)
            else:  # Toxicity
                logits, _ = model(input_ids, attention_mask)
                loss = model.compute_tox_loss(logits, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            # Early stop for quick test (only 5 batches per epoch)
            if num_batches >= 5:
                break

        avg_loss = total_loss / num_batches
        logger.info(f"  Train Loss: {avg_loss:.4f}")

        # Quick validation (only 3 batches)
        model.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                if "Emotion" in phase_name:
                    _, logits = model(input_ids, attention_mask)
                    loss = model.compute_emo_loss(logits, labels)
                else:
                    logits, _ = model(input_ids, attention_mask)
                    loss = model.compute_tox_loss(logits, labels)

                val_loss += loss.item()
                val_batches += 1

                if val_batches >= 3:
                    break

        avg_val_loss = val_loss / val_batches
        logger.info(f"  Val Loss: {avg_val_loss:.4f}")

    return model


def main():
    """Quick test of sequential training."""

    logger.info("="*60)
    logger.info("SEQUENTIAL TRAINING QUICK TEST")
    logger.info("="*60)
    logger.info("This is a quick test with minimal data to verify the pipeline works.")
    logger.info("")

    # Test parameters
    BATCH_SIZE = 8
    PRETRAIN_EPOCHS = 1
    FINETUNE_EPOCHS = 1
    LR = 2e-5

    # Use balanced datasets for test
    TOX_SAMPLED = False
    EMO_SAMPLED = False

    DEVICE = "cuda" if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Pretrain epochs: {PRETRAIN_EPOCHS}")
    logger.info(f"Finetune epochs: {FINETUNE_EPOCHS}")
    logger.info(f"Learning rate: {LR}")
    logger.info("")

    try:
        # Initialize model
        logger.info("Initializing MultiTaskBERT model...")
        model = MultiTaskBERT(
            num_toxicity_labels=6,
            num_emotion_labels=28
        )
        logger.info("✓ Model initialized")

        # PHASE 1: Emotion Pretraining
        logger.info("\nLoading emotion data...")
        try:
            emo_train_loader, emo_val_loader = load_emotion_data(
                emo_sampled_bool=EMO_SAMPLED,
                batch_size=BATCH_SIZE
            )
            logger.info(f"✓ Emotion data loaded")
        except Exception as e:
            logger.error(f"✗ Failed to load emotion data: {e}")
            logger.info("\nTIP: Make sure you're running this script from the project root directory")
            logger.info("Or adjust the paths in train_utils.py to match your directory structure")
            raise

        model = quick_train_phase(
            model,
            emo_train_loader,
            emo_val_loader,
            epochs=PRETRAIN_EPOCHS,
            lr=LR,
            device=DEVICE,
            phase_name="PHASE 1: Emotion Pretraining"
        )
        logger.info("✓ Emotion pretraining complete!")

        # PHASE 2: Toxicity Finetuning
        logger.info("\nLoading toxicity data...")
        try:
            tox_train_loader, tox_val_loader = load_toxicity_data(
                tox_sampled_bool=TOX_SAMPLED,
                batch_size=BATCH_SIZE
            )
            logger.info(f"✓ Toxicity data loaded")
        except Exception as e:
            logger.error(f"✗ Failed to load toxicity data: {e}")
            logger.info("\nTIP: Make sure you're running this script from the project root directory")
            logger.info("Or adjust the paths in train_utils.py to match your directory structure")
            raise

        model = quick_train_phase(
            model,
            tox_train_loader,
            tox_val_loader,
            epochs=FINETUNE_EPOCHS,
            lr=LR,
            device=DEVICE,
            phase_name="PHASE 2: Toxicity Finetuning"
        )
        logger.info("✓ Toxicity finetuning complete!")

        logger.info("\n" + "="*60)
        logger.info("✓ SEQUENTIAL TRAINING TEST PASSED!")
        logger.info("="*60)
        logger.info("\nThe sequential training pipeline works correctly.")
        logger.info("You can now run the full training with:")
        logger.info("  python src/sequential_train.py --balanced False --weighted False")
        logger.info("or")
        logger.info("  python src/sequential_train.py --balanced True --weighted True")

    except Exception as e:
        logger.error(f"\n✗ TEST FAILED: {e}")
        logger.error("\nPlease check:")
        logger.error("1. Data files exist in data/processed/ and data/processed_sampled/")
        logger.error("2. You're running from the correct directory")
        logger.error("3. Paths in train_utils.py are correct for your setup")
        raise


if __name__ == "__main__":
    main()
