# model.py
import torch
import torch.nn as nn
from transformers import BertModel

class BaselineBERT(nn.Module):
    """Single-task BERT for toxicity classification only."""
    
    def __init__(
        self,
        bert_model_name: str = 'bert-base-uncased',
        num_toxicity_labels: int = 6,
        dropout_prob: float = 0.1,
        class_weights: torch.Tensor = None
    ):
        super().__init__()
        
        self.shared_encoder = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(self.shared_encoder.config.hidden_dropout_prob)
        
        hidden_size = self.shared_encoder.config.hidden_size
        self.toxicity_head = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, num_toxicity_labels)
        )
        
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights) 
    
    def forward(self, input_ids, attention_mask):
        outputs = self.shared_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        toxicity_logits = self.toxicity_head(cls_output)
        
        return toxicity_logits
    
    def compute_loss(self, logits, labels):
        """Compute BCEWithLogitsLoss for multi-label toxicity."""
        return self.criterion(logits, labels)


class MultiTaskBERT(nn.Module):
    """Multi-task BERT for toxicity and emotion classification."""
    
    def __init__(
        self,
        bert_model_name: str = 'bert-base-uncased',
        num_toxicity_labels: int = 6,
        num_emotion_labels: int = 28,
        dropout_prob: float = 0.1,
        lambda_tox: float = 1.0,
        lambda_emo: float = 1.0,
        tox_class_weights: torch.Tensor = None,
        emo_class_weights: torch.Tensor = None
    ):
        super().__init__()
        
        self.lambda_tox = lambda_tox
        self.lambda_emo = lambda_emo
        
        # Shared encoder
        self.shared_encoder = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(self.shared_encoder.config.hidden_dropout_prob)
        
        # Task-specific heads with dropout
        hidden_size = self.shared_encoder.config.hidden_size
        self.toxicity_head = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, num_toxicity_labels)
        )
        self.emotion_head = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, num_emotion_labels)
        )
        
        # Loss functions
        self.criterion_tox = nn.BCEWithLogitsLoss(pos_weight=tox_class_weights)
        self.criterion_emo = nn.BCEWithLogitsLoss(pos_weight=emo_class_weights)

    
    def forward(self, input_ids, attention_mask):
        outputs = self.shared_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        
        toxicity_logits = self.toxicity_head(cls_output)
        emotion_logits = self.emotion_head(cls_output)
        
        return toxicity_logits, emotion_logits
    
    def compute_tox_loss(self, logits, labels):
        """Compute loss for toxicity task (multi-label)."""
        return self.criterion_tox(logits, labels.float())
    
    def compute_emo_loss(self, logits, labels):
        """Compute loss for emotion task (multi-label)."""
        return self.criterion_emo(logits, labels.float())
    
    def compute_joint_loss(self, tox_logits, emo_logits, tox_labels, emo_labels):
        """Compute weighted joint loss."""
        loss_tox = self.compute_tox_loss(tox_logits, tox_labels)
        loss_emo = self.compute_emo_loss(emo_logits, emo_labels)
        return self.lambda_tox * loss_tox + self.lambda_emo * loss_emo

class SequentialBERT(nn.Module):
    """Sequential multi-task BERT: first toxicity, then emotion."""
    
    def __init__(
        self,
        bert_model_name: str = 'bert-base-uncased',
        num_toxicity_labels: int = 6,
        num_emotion_labels: int = 28,
        dropout_prob: float = 0.1,
        lambda_tox: float = 1.0,
        lambda_emo: float = 1.0,
        tox_class_weights: torch.Tensor = None,
        emo_class_weights: torch.Tensor = None
    ):
        super().__init__()
        
        self.lambda_tox = lambda_tox
        self.lambda_emo = lambda_emo
        
        # Shared encoder
        self.shared_encoder = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(self.shared_encoder.config.hidden_dropout_prob)
        
        # Task-specific heads with dropout
        hidden_size = self.shared_encoder.config.hidden_size
        self.toxicity_head = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, num_toxicity_labels)
        )
        self.emotion_head = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size + num_toxicity_labels, num_emotion_labels)
        )
        
        # Loss functions
        self.criterion_tox = nn.BCEWithLogitsLoss(pos_weight=tox_class_weights)
        self.criterion_emo = nn.BCEWithLogitsLoss(pos_weight=emo_class_weights)

    
    def forward(self, input_ids, attention_mask):
        outputs = self.shared_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        
        toxicity_logits = self.toxicity_head(cls_output)
        
        # Concatenate toxicity logits to cls_output for emotion head
        emo_input = torch.cat((cls_output, toxicity_logits), dim=1)
        emotion_logits = self.emotion_head(emo_input)
        
        return toxicity_logits, emotion_logits
    
    def compute_tox_loss(self, logits, labels):
        """Compute loss for toxicity task (multi-label)."""
        return self.criterion_tox(logits, labels.float())
    
    def compute_emo_loss(self, logits, labels):
        """Compute loss for emotion task (multi-label)."""
        return self.criterion_emo(logits, labels.float())