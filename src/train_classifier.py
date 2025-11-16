"""
This script trains a multi-task BERT model for toxicity and emotion classification.
It includes:
1. Loading prepared datasets
2. Initializing the MultiTaskBERT model
3. Setting up the training loop with joint loss computation
4. Saving the trained model with checkpointing
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import logging
from pathlib import Path
from multitask_bert import MultiTaskBERT, compute_joint_loss

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiTaskDataset(Dataset):
    """Dataset for multi-task learning with toxicity and emotion labels."""
    
    def __init__(self, toxicity_data, emotion_data):
        self.tox = toxicity_data
        self.emo = emotion_data
        self.size = len(self.tox["labels"])
        
        # Validate data sizes match
        if len(self.emo["labels"]) != self.size:
            raise ValueError(
                f"Toxicity and emotion data size mismatch: "
                f"{self.size} vs {len(self.emo['labels'])}"
            )

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "input_ids": self.tox["tokens"]["input_ids"][idx],
            "attention_mask": self.tox["tokens"]["attention_mask"][idx],
            "toxicity_labels": self.tox["labels"][idx],
            "emotion_labels": self.emo["labels"][idx]
        }


def collate_fn(batch):
    """Collate function to properly batch tensors."""
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "toxicity_labels": torch.stack([item["toxicity_labels"] for item in batch]),
        "emotion_labels": torch.stack([item["emotion_labels"] for item in batch])
    }


def load_data(data_root="data/processed/tokenized"):
    """Load prepared datasets for toxicity and emotion classification.
    
    Args:
        data_root: Root directory containing tokenized data
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    data_root = Path(data_root)
    
    # Check if data directories exist
    tox_dir = data_root / "toxicity"
    emo_dir = data_root / "emotion"
    
    if not tox_dir.exists() or not emo_dir.exists():
        raise FileNotFoundError(
            f"Data directories not found. Expected:\n"
            f"  - {tox_dir}\n"
            f"  - {emo_dir}"
        )
    
    logger.info("Loading toxicity data...")
    toxicity_train = torch.load(tox_dir / "train.pt")
    toxicity_val = torch.load(tox_dir / "val.pt")

    logger.info("Loading emotion data...")
    emotion_train = torch.load(emo_dir / "train.pt")
    emotion_val = torch.load(emo_dir / "val.pt")

    train_ds = MultiTaskDataset(toxicity_train, emotion_train)
    val_ds = MultiTaskDataset(toxicity_val, emotion_val)
    
    logger.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    return train_ds, val_ds


def train_model(
    model, 
    train_loader, 
    val_loader, 
    epochs=3, 
    lr=2e-5, 
    device="cuda",
    save_path="models/multitask_bert.pt",
    gradient_clip=1.0
):
    """Train the multi-task model.
    
    Args:
        model: MultiTaskBERT model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        save_path: Path to save the best model
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

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0

        logger.info(f"\nEpoch {epoch+1}/{epochs}")

        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            optimizer.zero_grad()

            # Move tensors to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tox_labels = batch["toxicity_labels"].float().to(device)
            emo_labels = batch["emotion_labels"].float().to(device)

            # Forward pass
            tox_logits, emo_logits = model(input_ids, attention_mask)

            # Compute loss
            loss = compute_joint_loss(
                tox_logits, emo_logits,
                tox_labels, emo_labels,
                lambda_tox=model.lambda_tox,
                lambda_emotion=model.lambda_emo
            )

            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"Train Loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                tox_labels = batch["toxicity_labels"].float().to(device)
                emo_labels = batch["emotion_labels"].float().to(device)

                tox_logits, emo_logits = model(input_ids, attention_mask)

                loss = compute_joint_loss(
                    tox_logits, emo_logits,
                    tox_labels, emo_labels,
                    lambda_tox=model.lambda_tox,
                    lambda_emotion=model.lambda_emo
                )
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
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
            }, save_path)
            logger.info(f"✓ Saved best model (val_loss: {avg_val_loss:.4f})")

    return model





def main():
    """Main training function."""
    # Hyperparameters
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5    
    MODEL_SAVE_PATH = "models/multitask_bert.pt"
    GRADIENT_CLIP = 1.0
    
    DEVICE = "cuda" if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else "cpu"

    try:
        # Load data
        train_ds, val_ds = load_data()

        # Create data loaders
        train_loader = DataLoader(
            train_ds, 
            batch_size=BATCH_SIZE, 
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0 
        )

        val_loader = DataLoader(
            val_ds, 
            batch_size=BATCH_SIZE,
            collate_fn=collate_fn,
            num_workers=0
        )

        # Initialize model
        logger.info("Initializing MultiTaskBERT model...")
        model = MultiTaskBERT(lambda_tox=1.0, lambda_emo=1.0)

        # Train model
        model = train_model(
            model,
            train_loader,
            val_loader,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            device=DEVICE,
            save_path=MODEL_SAVE_PATH,
            gradient_clip=GRADIENT_CLIP
        )

        logger.info(f"\n✓ Training complete! Model saved to {MODEL_SAVE_PATH}")
        
    except FileNotFoundError as e:
        logger.error(f"Data loading error: {e}")
        raise
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise


if __name__ == "__main__":
    main()