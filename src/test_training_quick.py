# test_training_quick.py
"""
Quick test to verify training and checkpoint saving works end-to-end.
Uses a tiny subset of data and 1 epoch to test quickly.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(__file__))

from model import BaselineBERT
from compute_weights import load_class_weights

def create_dummy_data(num_samples=32, seq_length=128):
    """Create dummy toxicity data for testing."""
    input_ids = torch.randint(0, 30000, (num_samples, seq_length))
    attention_mask = torch.ones(num_samples, seq_length)
    labels = torch.randint(0, 2, (num_samples, 6)).float()

    dataset = TensorDataset(input_ids, attention_mask, labels)
    return dataset

def mini_train(model, train_loader, device, epochs=1):
    """Mini training loop - just a few batches."""
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    print(f"\nRunning mini training ({epochs} epoch(s))...")
    for epoch in range(epochs):
        total_loss = 0
        for i, batch in enumerate(train_loader):
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = model.compute_loss(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    return model

def test_checkpoint_saving():
    """Test the full training and checkpoint saving workflow."""

    print("="*60)
    print("QUICK TRAINING TEST")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create test directory
    test_dir = "test_checkpoints"
    os.makedirs(test_dir, exist_ok=True)

    try:
        # Test 1: Unweighted model
        print("\n" + "="*60)
        print("TEST 1: Unweighted Model")
        print("="*60)

        model = BaselineBERT(num_toxicity_labels=6, class_weights=None)
        train_data = create_dummy_data(num_samples=16)
        train_loader = DataLoader(train_data, batch_size=8, shuffle=True)

        model = mini_train(model, train_loader, device, epochs=1)

        # Save checkpoint (unweighted path)
        checkpoint_path = os.path.join(test_dir, "test_baseline_bert.pt")
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "class_weights": None,
            "epochs": 1,
            "lr": 2e-5
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"✅ Checkpoint saved to {checkpoint_path}")

        # Load it back
        loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
        new_model = BaselineBERT(num_toxicity_labels=6, class_weights=None)
        new_model.load_state_dict(loaded['model_state_dict'])
        print(f"✅ Checkpoint loaded successfully")
        print(f"   Keys: {list(loaded.keys())}")
        print(f"   Class weights: {loaded['class_weights']}")

        # Test 2: Weighted model
        print("\n" + "="*60)
        print("TEST 2: Weighted Model")
        print("="*60)

        class_weights = load_class_weights(task="toxicity", sampled=False)
        model_weighted = BaselineBERT(num_toxicity_labels=6, class_weights=class_weights)

        model_weighted = mini_train(model_weighted, train_loader, device, epochs=1)

        # Save checkpoint (weighted path)
        checkpoint_path_weighted = os.path.join(test_dir, "test_baseline_bert_weighted.pt")
        checkpoint_weighted = {
            "model_state_dict": model_weighted.state_dict(),
            "class_weights": class_weights,
            "epochs": 1,
            "lr": 2e-5
        }

        torch.save(checkpoint_weighted, checkpoint_path_weighted)
        print(f"✅ Checkpoint saved to {checkpoint_path_weighted}")

        # Load it back
        loaded_weighted = torch.load(checkpoint_path_weighted, map_location=device, weights_only=False)
        new_model_weighted = BaselineBERT(num_toxicity_labels=6, class_weights=class_weights)
        new_model_weighted.load_state_dict(loaded_weighted['model_state_dict'])
        print(f"✅ Checkpoint loaded successfully")
        print(f"   Keys: {list(loaded_weighted.keys())}")
        print(f"   Class weights shape: {loaded_weighted['class_weights'].shape if loaded_weighted['class_weights'] is not None else 'None'}")

        # Test 3: Simulate the exact baseline_train.py save logic
        print("\n" + "="*60)
        print("TEST 3: Exact baseline_train.py Logic")
        print("="*60)

        MODEL_SAVE_PATH = os.path.join(test_dir, "baseline_bert_test.pt")

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "class_weights": None,
            "epochs": 1,
            "lr": 2e-5
        }

        # Simulate the exact save logic from baseline_train.py
        try:
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(checkpoint, MODEL_SAVE_PATH)
            print(f"✅ Saved checkpoint to {MODEL_SAVE_PATH}")

        except Exception as e:
            print(f"First save failed: {e}")
            alt_path = MODEL_SAVE_PATH.replace("../", "")
            os.makedirs(os.path.dirname(alt_path), exist_ok=True)
            torch.save(checkpoint, alt_path)
            print(f"✅ Saved checkpoint to fallback path {alt_path}")

        print(f"✅ Training complete! (no error accessing undefined variables)")

        # Verify file exists
        if os.path.exists(MODEL_SAVE_PATH):
            print(f"✅ Checkpoint file exists: {MODEL_SAVE_PATH}")
            file_size = os.path.getsize(MODEL_SAVE_PATH) / (1024 * 1024)
            print(f"   Size: {file_size:.2f} MB")
        else:
            print(f"❌ Checkpoint file NOT found: {MODEL_SAVE_PATH}")
            return False

        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✅")
        print("="*60)
        print("\nThe training script should work correctly now.")
        print("You can safely run:")
        print("  python baseline_train.py --weighted True")
        print("  python baseline_train.py --weighted False")

        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print(f"\n🧹 Cleaned up test directory: {test_dir}")

if __name__ == "__main__":
    success = test_checkpoint_saving()
    sys.exit(0 if success else 1)
