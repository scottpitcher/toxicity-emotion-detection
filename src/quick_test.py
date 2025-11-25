# quick_test.py
"""Quick test to verify all three experiment models work correctly."""

import torch
import logging
from flexible_model import FlexibleBERT
from train_utils import load_toxicity_data, load_emotion_data, load_both_datasets

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_baseline():
    """Test baseline model."""
    logger.info("\n" + "="*60)
    logger.info("TESTING BASELINE MODEL")
    logger.info("="*60)

    try:
        # Load data (single batch)
        train_loader, val_loader = load_toxicity_data(batch_size=2)

        # Initialize model
        model = FlexibleBERT(mode='baseline')
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model.to(device)

        logger.info(f"Device: {device}")
        logger.info(f"Trainable params: {model.get_trainable_params()}")

        # Test forward pass
        model.train()
        batch = next(iter(train_loader))
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward
        tox_logits, emo_logits = model(input_ids, attention_mask)
        assert emo_logits is None, "Emotion logits should be None in baseline mode"
        assert tox_logits.shape == (2, 6), f"Expected shape (2, 6), got {tox_logits.shape}"

        # Loss
        loss = model.compute_loss(tox_logits, labels)
        assert loss.item() > 0, "Loss should be positive"

        # Backward
        loss.backward()

        logger.info(f"✓ Forward pass successful, output shape: {tox_logits.shape}")
        logger.info(f"✓ Loss computation successful: {loss.item():.4f}")
        logger.info(f"✓ Backward pass successful")
        logger.info("✓ BASELINE MODEL TEST PASSED")
        return True

    except Exception as e:
        logger.error(f"✗ BASELINE MODEL TEST FAILED: {e}")
        return False


def test_multitask():
    """Test multitask model."""
    logger.info("\n" + "="*60)
    logger.info("TESTING MULTITASK MODEL")
    logger.info("="*60)

    try:
        # Load data (single batch)
        tox_train, tox_val = load_toxicity_data(data_root="data/processed/tokenized", batch_size=2)
        emo_train, emo_val = load_emotion_data(data_root="data/processed_sampled", batch_size=2)

        # Initialize model
        model = FlexibleBERT(mode='multitask', lambda_tox=1.0, lambda_emo=1.0)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model.to(device)

        logger.info(f"Device: {device}")
        logger.info(f"Trainable params: {model.get_trainable_params()}")

        # Test forward pass with toxicity batch
        model.train()
        tox_batch = next(iter(tox_train))
        input_ids = tox_batch["input_ids"].to(device)
        attention_mask = tox_batch["attention_mask"].to(device)
        tox_labels = tox_batch["labels"].to(device)

        tox_logits, emo_logits = model(input_ids, attention_mask)
        assert tox_logits.shape == (2, 6), f"Expected tox shape (2, 6), got {tox_logits.shape}"
        assert emo_logits.shape == (2, 28), f"Expected emo shape (2, 28), got {emo_logits.shape}"

        # Test toxicity loss
        loss_tox = model.compute_loss(tox_logits, tox_labels)
        assert loss_tox.item() > 0, "Toxicity loss should be positive"
        loss_tox.backward()

        logger.info(f"✓ Toxicity forward pass successful, shapes: tox={tox_logits.shape}, emo={emo_logits.shape}")
        logger.info(f"✓ Toxicity loss: {loss_tox.item():.4f}")

        # Test forward pass with emotion batch
        model.zero_grad()
        emo_batch = next(iter(emo_train))
        input_ids = emo_batch["input_ids"].to(device)
        attention_mask = emo_batch["attention_mask"].to(device)
        emo_labels = emo_batch["labels"].to(device)

        tox_logits, emo_logits = model(input_ids, attention_mask)

        # Test emotion loss
        loss_emo = model.compute_loss(None, None, emo_logits, emo_labels)
        assert loss_emo.item() > 0, "Emotion loss should be positive"
        loss_emo.backward()

        logger.info(f"✓ Emotion loss: {loss_emo.item():.4f}")
        logger.info("✓ MULTITASK MODEL TEST PASSED")
        return True

    except Exception as e:
        logger.error(f"✗ MULTITASK MODEL TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sequential():
    """Test sequential model."""
    logger.info("\n" + "="*60)
    logger.info("TESTING SEQUENTIAL MODEL")
    logger.info("="*60)

    try:
        # Load data
        tox_train, tox_val = load_toxicity_data(data_root="data/processed/tokenized", batch_size=2)
        emo_train, emo_val = load_emotion_data(data_root="data/processed_sampled", batch_size=2)

        # Initialize model
        model = FlexibleBERT(mode='sequential')
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model.to(device)

        logger.info(f"Device: {device}")
        logger.info(f"Trainable params: {model.get_trainable_params()}")

        # PHASE 1: Test emotion pretraining
        logger.info("\nPhase 1: Emotion Pretraining")
        model.train()
        emo_batch = next(iter(emo_train))
        input_ids = emo_batch["input_ids"].to(device)
        attention_mask = emo_batch["attention_mask"].to(device)
        emo_labels = emo_batch["labels"].to(device)

        _, emo_logits = model(input_ids, attention_mask)
        assert emo_logits.shape == (2, 28), f"Expected emo shape (2, 28), got {emo_logits.shape}"

        loss_emo = model.compute_loss(None, None, emo_logits, emo_labels)
        assert loss_emo.item() > 0, "Emotion loss should be positive"
        loss_emo.backward()

        logger.info(f"✓ Emotion pretraining forward pass successful")
        logger.info(f"✓ Emotion loss: {loss_emo.item():.4f}")

        # PHASE 2: Test toxicity finetuning
        logger.info("\nPhase 2: Toxicity Finetuning")
        model.zero_grad()
        tox_batch = next(iter(tox_train))
        input_ids = tox_batch["input_ids"].to(device)
        attention_mask = tox_batch["attention_mask"].to(device)
        tox_labels = tox_batch["labels"].to(device)

        tox_logits, _ = model(input_ids, attention_mask, return_emotion=False)
        assert tox_logits.shape == (2, 6), f"Expected tox shape (2, 6), got {tox_logits.shape}"

        loss_tox = model.compute_loss(tox_logits, tox_labels)
        assert loss_tox.item() > 0, "Toxicity loss should be positive"
        loss_tox.backward()

        logger.info(f"✓ Toxicity finetuning forward pass successful")
        logger.info(f"✓ Toxicity loss: {loss_tox.item():.4f}")

        # Test freezing encoder
        model.freeze_encoder()
        params_frozen = model.get_trainable_params()
        assert params_frozen['encoder'] == 0, "Encoder should have 0 trainable params when frozen"
        logger.info(f"✓ Encoder freezing works correctly")

        logger.info("✓ SEQUENTIAL MODEL TEST PASSED")
        return True

    except Exception as e:
        logger.error(f"✗ SEQUENTIAL MODEL TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    logger.info("Starting quick tests for all three experiments...")

    results = {
        'baseline': test_baseline(),
        'multitask': test_multitask(),
        'sequential': test_sequential()
    }

    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{name.upper()}: {status}")

    if all(results.values()):
        logger.info("\n✓ ALL TESTS PASSED! Ready for full training.")
    else:
        logger.info("\n✗ SOME TESTS FAILED. Please fix before running full training.")
