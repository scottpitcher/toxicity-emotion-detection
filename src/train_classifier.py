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

# DATASETS #
class ToxicityDataset(Dataset):
    """Dataset for toxicity classification."""
    
    def __init__(self, data):
        self.tokens = data["tokens"]
        self.labels = data["labels"]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.tokens["input_ids"][idx],
            "attention_mask": self.tokens["attention_mask"][idx],
            "labels": self.labels[idx]
        }

class EmotionDataset(Dataset):
    """Dataset for emotion classification."""
    
    def __init__(self, data):
        self.tokens = data["tokens"]
        self.labels = data["labels"]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.tokens["input_ids"][idx],
            "attention_mask": self.tokens["attention_mask"][idx],
            "labels": self.labels[idx]
        }


# COLLATE
## to batch samples
def toxicity_collate_fn(batch):
    """Collate function for toxicity batches."""
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch])
    }


def emotion_collate_fn(batch):
    """Collate function for emotion batches."""
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch])
    }

# LOAD DATA #
def load_data(data_root="../data/processed/tokenized"):
    """Load separate datasets for toxicity and emotion classification.
    
    Args:
        data_root: Root directory containing tokenized data
        
    Returns:
        Tuple of (tox_train, tox_val, emo_train, emo_val)
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
    toxicity_train_data = torch.load(tox_dir / "train.pt")
    toxicity_val_data = torch.load(tox_dir / "val.pt")

    logger.info("Loading emotion data...")
    emotion_train_data = torch.load(emo_dir / "train.pt")
    emotion_val_data = torch.load(emo_dir / "val.pt")

    # Create separate datasets
    tox_train = ToxicityDataset(toxicity_train_data)
    tox_val = ToxicityDataset(toxicity_val_data)
    emo_train = EmotionDataset(emotion_train_data)
    emo_val = EmotionDataset(emotion_val_data)
    
    logger.info(f"Toxicity - Train: {len(tox_train)}, Val: {len(tox_val)}")
    logger.info(f"Emotion - Train: {len(emo_train)}, Val: {len(emo_val)}")

    return tox_train, tox_val, emo_train, emo_val


# TRAINING LOOP #
def train_model(
    model,
    tox_train_loader,
    tox_val_loader,
    emo_train_loader,
    emo_val_loader,
    epochs=3,
    lr=2e-5,
    device="cuda",
    save_path="models/multitask_bert.pt",
    gradient_clip=1.0):

    """Train the multi-task model with separate dataloaders.
    
    Args:
        model: MultiTaskBERT model
        tox_train_loader: Toxicity training data loader
        tox_val_loader: Toxicity validation data loader
        emo_train_loader: Emotion training data loader
        emo_val_loader: Emotion validation data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        save_path: Path to save the best model
        gradient_clip: Gradient clipping value
        
    Returns:
        Trained model
    """

    optimizer = AdamW(model.parameters(), lr=lr)

    # Calculate total steps (interleaved batches)
    steps_per_epoch = max(len(tox_train_loader), len(emo_train_loader))
    total_steps = steps_per_epoch * epochs
    
    # Learning rate scheduler
    ## adjust learning rate over time to help with convergence
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
        total_tox_loss = 0
        total_emo_loss = 0
        total_batches = 0

        logger.info(f"\nEpoch {epoch+1}/{epochs}")

        # Create iterators
        ## Because we have separate dataloaders, we create iterators for each
        tox_iter = iter(tox_train_loader)
        emo_iter = iter(emo_train_loader)

        progress_bar = tqdm(range(steps_per_epoch), desc="Training") # used later in training to update progress
        for step in progress_bar:
            # Alternate between toxicity and emotion

            # PART 1: Train on toxicity #
            try:
                tox_batch = next(tox_iter)
                
                optimizer.zero_grad()
                
                input_ids = tox_batch["input_ids"].to(device)
                attention_mask = tox_batch["attention_mask"].to(device)
                tox_labels = tox_batch["labels"].long().to(device)  # Long for CrossEntropy
                
                tox_logits, _ = model(input_ids, attention_mask)
                
                # Only toxicity loss
                criterion_tox = nn.CrossEntropyLoss()
                loss = model.lambda_tox * criterion_tox(tox_logits, tox_labels)
                
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip) # grad clip to avoid exploding gradients
                optimizer.step()
                scheduler.step()
                
                total_tox_loss += loss.item()
                total_batches += 1
                
            except StopIteration:
                tox_iter = iter(tox_train_loader)  # Restart if exhausted
            
            # PART 2: Train on emotion (same as above) #
            try:
                emo_batch = next(emo_iter)
                
                optimizer.zero_grad()
                
                input_ids = emo_batch["input_ids"].to(device)
                attention_mask = emo_batch["attention_mask"].to(device)
                emo_labels = emo_batch["labels"].float().to(device)  # float for BCE
                
                _, emo_logits = model(input_ids, attention_mask)
                
                # Only emotion loss
                criterion_emo = nn.BCEWithLogitsLoss() # BCE for multi-label
                loss = model.lambda_emo * criterion_emo(emo_logits, emo_labels)
                
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
                scheduler.step()
                
                total_emo_loss += loss.item()
                total_batches += 1
                
            except StopIteration:
                emo_iter = iter(emo_train_loader)  # Restart if exhausted
            
            progress_bar.set_postfix({
                'tox_loss': f'{total_tox_loss/(step+1):.4f}',
                'emo_loss': f'{total_emo_loss/(step+1):.4f}'
            })

        avg_train_loss = (total_tox_loss + total_emo_loss) / total_batches
        logger.info(f"Train Loss - Total: {avg_train_loss:.4f}, Tox: {total_tox_loss/steps_per_epoch:.4f}, Emo: {total_emo_loss/steps_per_epoch:.4f}")

        # PART 3: Validation phase #
        model.eval()
        val_tox_loss = 0
        val_emo_loss = 0
        
        with torch.no_grad():
            # Validate toxicity
            for batch in tqdm(tox_val_loader, desc="Val Toxicity"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                tox_labels = batch["labels"].long().to(device)
                
                tox_logits, _ = model(input_ids, attention_mask)
                
                criterion_tox = nn.CrossEntropyLoss()
                loss = criterion_tox(tox_logits, tox_labels)
                val_tox_loss += loss.item()
            
            # Validate emotion
            for batch in tqdm(emo_val_loader, desc="Val Emotion"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                emo_labels = batch["labels"].float().to(device)
                
                _, emo_logits = model(input_ids, attention_mask)
                
                criterion_emo = nn.BCEWithLogitsLoss()
                loss = criterion_emo(emo_logits, emo_labels)
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
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_tox_loss': avg_val_tox_loss,
                'val_emo_loss': avg_val_emo_loss,
            }, save_path)
            logger.info(f"Saved best model (val_loss: {avg_val_loss:.4f})")

    return model





def main():
    """Main training function."""
    # Hyperparameters
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    MODEL_SAVE_PATH = "models/multitask_bert.pt"
    GRADIENT_CLIP = 1.0
    LAMBDA_TOX = 1.0
    LAMBDA_EMO = 1.0
    
    DEVICE = "cuda" if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {DEVICE}")
    
    try:
        # Load separate datasets
        tox_train, tox_val, emo_train, emo_val = load_data()

        # Create separate data loaders
        tox_train_loader = DataLoader(
            tox_train,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=toxicity_collate_fn,
            num_workers=0
        )
        
        tox_val_loader = DataLoader(
            tox_val,
            batch_size=BATCH_SIZE,
            collate_fn=toxicity_collate_fn,
            num_workers=0
        )
        
        emo_train_loader = DataLoader(
            emo_train,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=emotion_collate_fn,
            num_workers=0
        )
        
        emo_val_loader = DataLoader(
            emo_val,
            batch_size=BATCH_SIZE,
            collate_fn=emotion_collate_fn,
            num_workers=0
        )

        # Initialize model
        logger.info("Initializing MultiTaskBERT model...")
        model = MultiTaskBERT(lambda_tox=LAMBDA_TOX, lambda_emo=LAMBDA_EMO)

        # Train model
        model = train_model(
            model,
            tox_train_loader,
            tox_val_loader,
            emo_train_loader,
            emo_val_loader,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            device=DEVICE,
            save_path=MODEL_SAVE_PATH,
            gradient_clip=GRADIENT_CLIP
        )

        logger.info(f"\nTraining complete! Model saved to {MODEL_SAVE_PATH}")
        
    except FileNotFoundError as e:
        logger.error(f"Data loading error: {e}")
        raise
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise


if __name__ == "__main__":
    main()
