# train_utils.py
"""
Shared utilities for training toxicity and emotion models.
Includes dataset classes, collate functions, and data loading.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import os
import torch.nn as nn


logger = logging.getLogger(__name__)

## DATASETS AND DATALOADERS ##
class ToxicityDataset(Dataset):
    """Dataset for multi-label toxicity classification."""
    
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
    """Dataset for multi-label emotion classification."""
    
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

## COLLATE FUNCTIONS FOR DATALOADERS ##
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


## DATA LOADING FUNCTIONS ##
def load_toxicity_data(data_root="data/processed/tokenized", batch_size=16):
    """Load toxicity datasets and create dataloaders.
    
    Args:
        data_root: Root directory containing tokenized data
        batch_size: Batch size for dataloaders
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_root = Path(data_root)
    tox_dir = data_root / "toxicity"
    
    if not tox_dir.exists():
        raise FileNotFoundError(f"Toxicity data directory not found: {tox_dir}")
    
    logger.info("Loading toxicity data...")
    toxicity_train_data = torch.load(tox_dir / "train.pt")
    toxicity_val_data = torch.load(tox_dir / "val.pt")

    tox_train = ToxicityDataset(toxicity_train_data)
    tox_val = ToxicityDataset(toxicity_val_data)
    
    logger.info(f"Toxicity - Train: {len(tox_train)}, Val: {len(tox_val)}")

    train_loader = DataLoader(
        tox_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=toxicity_collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        tox_val,
        batch_size=batch_size,
        collate_fn=toxicity_collate_fn,
        num_workers=0
    )

    return train_loader, val_loader


def load_emotion_data(data_root="../data/processed/tokenized", batch_size=16):
    """Load emotion datasets and create dataloaders.
    
    Args:
        data_root: Root directory containing tokenized data
        batch_size: Batch size for dataloaders
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_root = Path(data_root)
    emo_dir = data_root / "emotion"
    
    if not emo_dir.exists():
        raise FileNotFoundError(f"Emotion data directory not found: {emo_dir}")
    
    logger.info("Loading emotion data...")
    emotion_train_data = torch.load(emo_dir / "train.pt")
    emotion_val_data = torch.load(emo_dir / "val.pt")

    emo_train = EmotionDataset(emotion_train_data)
    emo_val = EmotionDataset(emotion_val_data)
    
    logger.info(f"Emotion - Train: {len(emo_train)}, Val: {len(emo_val)}")

    train_loader = DataLoader(
        emo_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=emotion_collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        emo_val,
        batch_size=batch_size,
        collate_fn=emotion_collate_fn,
        num_workers=0
    )

    return train_loader, val_loader

## COMBINED DATA LOADING FUNCTION ##
def load_both_datasets(data_root="../data/processed/tokenized", batch_size=16):
    """Load both toxicity and emotion datasets.
    
    Args:
        data_root: Root directory containing tokenized data
        batch_size: Batch size for dataloaders
        
    Returns:
        Tuple of (tox_train_loader, tox_val_loader, emo_train_loader, emo_val_loader)
    """
    tox_train, tox_val = load_toxicity_data(data_root, batch_size)
    emo_train, emo_val = load_emotion_data(data_root, batch_size)
    
    return tox_train, tox_val, emo_train, emo_val

## TRAINING UTILITIES ##
def train_baseline(
    model,
    train_loader,
    val_loader,
    epochs=3,
    lr=2e-5,
    device="cuda",
    save_path="models/baseline_bert.pt",
    gradient_clip=1.0
):
    """Train baseline single-task BERT model.
    
    Args:
        model: BaselineBERT model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        save_path: Path to save best model
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
        total_train_loss = 0
        
        logger.info(f"\nEpoch {epoch+1}/{epochs}")
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = model.compute_loss(logits, labels)
            
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
                
                logits = model(input_ids, attention_mask)
                loss = model.compute_loss(logits, labels)
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
            }, save_path)
            logger.info(f"Saved best model (val_loss: {avg_val_loss:.4f})")
    
    return model

def train_multitask(
    model,
    tox_train_loader,
    tox_val_loader,
    emo_train_loader,
    emo_val_loader,
    epochs=3,
    lr=2e-5,
    device="cuda",
    save_path="models/multitask_bert.pt",
    gradient_clip=1.0
):
    """Train multi-task BERT model with alternating batches.
    
    Args:
        model: MultiTaskBERT model
        tox_train_loader: Toxicity training data loader
        tox_val_loader: Toxicity validation data loader
        emo_train_loader: Emotion training data loader
        emo_val_loader: Emotion validation data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        save_path: Path to save best model
        gradient_clip: Gradient clipping value
        
    Returns:
        Trained model
    """
    
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
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_tox_loss = 0
        total_emo_loss = 0
        total_batches = 0
        
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
                
                tox_logits, _ = model(input_ids, attention_mask)
                loss = model.lambda_tox * model.compute_tox_loss(tox_logits, tox_labels)
                
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
                scheduler.step()
                
                total_tox_loss += loss.item()
                total_batches += 1
                
            except StopIteration:
                tox_iter = iter(tox_train_loader)
            
            # Train on emotion batch
            try:
                emo_batch = next(emo_iter)
                
                optimizer.zero_grad()
                
                input_ids = emo_batch["input_ids"].to(device)
                attention_mask = emo_batch["attention_mask"].to(device)
                emo_labels = emo_batch["labels"].to(device)
                
                _, emo_logits = model(input_ids, attention_mask)
                loss = model.lambda_emo * model.compute_emo_loss(emo_logits, emo_labels)
                
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
                scheduler.step()
                
                total_emo_loss += loss.item()
                total_batches += 1
                
            except StopIteration:
                emo_iter = iter(emo_train_loader)
            
            progress_bar.set_postfix({
                'tox_loss': f'{total_tox_loss/(step+1):.4f}',
                'emo_loss': f'{total_emo_loss/(step+1):.4f}'
            })
        
        avg_train_tox_loss = total_tox_loss / max(1, steps_per_epoch)
        avg_train_emo_loss = total_emo_loss / max(1, steps_per_epoch)
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
                loss = model.compute_tox_loss(tox_logits, labels)
                val_tox_loss += loss.item()
            
            # Validate emotion
            for batch in tqdm(emo_val_loader, desc="Val Emotion"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                _, emo_logits = model(input_ids, attention_mask)
                loss = model.compute_emo_loss(emo_logits, labels)
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
            }, save_path)
            logger.info(f"Saved best model (val_loss: {avg_val_loss:.4f})")
    
    return model
