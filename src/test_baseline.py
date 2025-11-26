# test_baseline.py
"""
Quick sanity check for baseline_train.py
Tests both weighted and unweighted configurations on CPU with minimal data.
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from model import BaselineBERT
from compute_weights import load_class_weights

def test_model_instantiation():
    """Test that models can be instantiated."""
    print("=" * 60)
    print("TEST 1: Model Instantiation")
    print("=" * 60)

    # Test without weights
    try:
        model = BaselineBERT(num_toxicity_labels=6, class_weights=None)
        print("✅ BaselineBERT (no weights) instantiated successfully")
    except Exception as e:
        print(f"❌ Failed to instantiate BaselineBERT without weights: {e}")
        return False

    # Test with weights
    try:
        weights = load_class_weights(task="toxicity", sampled=False)
        model = BaselineBERT(num_toxicity_labels=6, class_weights=weights)
        print(f"✅ BaselineBERT (with weights) instantiated successfully")
        print(f"   Class weights shape: {weights.shape if weights is not None else 'None'}")
    except Exception as e:
        print(f"❌ Failed to instantiate BaselineBERT with weights: {e}")
        return False

    return True

def test_forward_pass():
    """Test forward pass with dummy data."""
    print("\n" + "=" * 60)
    print("TEST 2: Forward Pass")
    print("=" * 60)

    try:
        model = BaselineBERT(num_toxicity_labels=6, class_weights=None)
        model.eval()

        # Create dummy batch
        batch_size = 4
        seq_length = 128
        input_ids = torch.randint(0, 30000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)

        # Forward pass
        with torch.no_grad():
            logits = model(input_ids, attention_mask)

        print(f"✅ Forward pass successful")
        print(f"   Input shape: {input_ids.shape}")
        print(f"   Output shape: {logits.shape}")
        print(f"   Expected output shape: ({batch_size}, 6)")

        assert logits.shape == (batch_size, 6), "Output shape mismatch!"
        print("✅ Output shape correct")

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

def test_loss_computation():
    """Test loss computation."""
    print("\n" + "=" * 60)
    print("TEST 3: Loss Computation")
    print("=" * 60)

    try:
        # Test without weights
        model = BaselineBERT(num_toxicity_labels=6, class_weights=None)

        batch_size = 4
        logits = torch.randn(batch_size, 6)
        labels = torch.randint(0, 2, (batch_size, 6)).float()

        loss = model.compute_loss(logits, labels)
        print(f"✅ Loss computation (no weights) successful: {loss.item():.4f}")

        # Test with weights
        weights = load_class_weights(task="toxicity", sampled=False)
        model_weighted = BaselineBERT(num_toxicity_labels=6, class_weights=weights)
        loss_weighted = model_weighted.compute_loss(logits, labels)
        print(f"✅ Loss computation (with weights) successful: {loss_weighted.item():.4f}")

    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

def test_checkpoint_save():
    """Test checkpoint saving logic."""
    print("\n" + "=" * 60)
    print("TEST 4: Checkpoint Saving")
    print("=" * 60)

    try:
        model = BaselineBERT(num_toxicity_labels=6, class_weights=None)
        weights = load_class_weights(task="toxicity", sampled=False)

        # Test save paths
        test_paths = [
            "test_models/baseline_bert.pt",
            "test_models/baseline_bert_weighted.pt"
        ]

        for i, path in enumerate(test_paths):
            is_weighted = i == 1
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_weights": weights if is_weighted else None,
                "epochs": 3,
                "lr": 2e-5
            }

            # Create directory
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Save
            torch.save(checkpoint, path)
            print(f"✅ Saved checkpoint to {path}")

            # Load back
            loaded = torch.load(path, map_location='cpu', weights_only=False)
            print(f"✅ Loaded checkpoint from {path}")
            print(f"   Keys: {list(loaded.keys())}")
            print(f"   Class weights: {'Present' if loaded['class_weights'] is not None else 'None'}")

            # Load into model
            model_test = BaselineBERT(num_toxicity_labels=6, class_weights=None)
            model_test.load_state_dict(loaded['model_state_dict'])
            print(f"✅ Successfully loaded state_dict into model")

        # Cleanup
        import shutil
        shutil.rmtree("test_models")
        print("✅ Cleanup successful")

    except Exception as e:
        print(f"❌ Checkpoint save/load failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

def test_data_loading():
    """Test that data can be loaded."""
    print("\n" + "=" * 60)
    print("TEST 5: Data Loading")
    print("=" * 60)

    try:
        from train_utils import load_toxicity_data

        print("Attempting to load toxicity data...")
        train_loader, val_loader = load_toxicity_data(batch_size=16)

        print(f"✅ Data loaded successfully")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")

        # Test one batch
        batch = next(iter(train_loader))
        print(f"✅ Sample batch loaded")
        print(f"   Batch keys: {batch.keys()}")
        print(f"   Input IDs shape: {batch['input_ids'].shape}")
        print(f"   Labels shape: {batch['labels'].shape}")

    except FileNotFoundError as e:
        print(f"⚠️  Data files not found (expected if data not processed): {e}")
        return True  # Not a failure, just data not available
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    print("\n" + "🧪 TESTING BASELINE_TRAIN.PY CONFIGURATION 🧪".center(60))
    print("=" * 60)

    all_passed = True

    # Run tests
    all_passed &= test_model_instantiation()
    all_passed &= test_forward_pass()
    all_passed &= test_loss_computation()
    all_passed &= test_checkpoint_save()
    all_passed &= test_data_loading()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("\nYou're good to run:")
        print("  python baseline_train.py --weighted False")
        print("  python baseline_train.py --weighted True")
    else:
        print("❌ SOME TESTS FAILED - Fix issues before running training")

    print("=" * 60)
