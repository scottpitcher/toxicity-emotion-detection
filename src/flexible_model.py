# flexible_model.py
"""
Unified BERT model that supports three training modes:
1. baseline: Toxicity classification only
2. multitask: Joint toxicity and emotion classification
3. sequential: Pretrain on emotion, then finetune on toxicity
"""

import torch
import torch.nn as nn
from transformers import BertModel
from typing import Optional, Tuple, Literal


class FlexibleBERT(nn.Module):
    """
    Flexible BERT model for toxicity and emotion classification experiments.

    Supports three modes:
    - baseline: Single-task learning on toxicity only
    - multitask: Joint learning on both toxicity and emotion
    - sequential: Two-phase learning (emotion pretraining -> toxicity finetuning)
    """

    def __init__(
        self,
        mode: Literal['baseline', 'multitask', 'sequential'] = 'baseline',
        bert_model_name: str = 'bert-base-uncased',
        num_toxicity_labels: int = 6,
        num_emotion_labels: int = 28,
        dropout_prob: float = 0.1,
        lambda_tox: float = 1.0,
        lambda_emo: float = 1.0,
        tox_class_weights: Optional[torch.Tensor] = None,
        emo_class_weights: Optional[torch.Tensor] = None
    ):
        """
        Initialize FlexibleBERT model.

        Args:
            mode: Training mode ('baseline', 'multitask', or 'sequential')
            bert_model_name: Pretrained BERT model name
            num_toxicity_labels: Number of toxicity labels
            num_emotion_labels: Number of emotion labels
            dropout_prob: Dropout probability for task heads
            lambda_tox: Weight for toxicity loss in multitask mode
            lambda_emo: Weight for emotion loss in multitask mode
            tox_class_weights: Class weights for toxicity BCE loss
            emo_class_weights: Class weights for emotion BCE loss
        """
        super().__init__()

        self.mode = mode
        self.lambda_tox = lambda_tox
        self.lambda_emo = lambda_emo

        # Shared BERT encoder
        self.shared_encoder = BertModel.from_pretrained(bert_model_name)
        hidden_size = self.shared_encoder.config.hidden_size
        self.dropout = nn.Dropout(self.shared_encoder.config.hidden_dropout_prob)

        # Toxicity head (used in all modes eventually)
        self.toxicity_head = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, num_toxicity_labels)
        )

        # Emotion head (used in multitask and sequential modes)
        if mode in ['multitask', 'sequential']:
            self.emotion_head = nn.Sequential(
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_size, num_emotion_labels)
            )
        else:
            self.emotion_head = None

        # Loss functions
        self.criterion_tox = nn.BCEWithLogitsLoss(pos_weight=tox_class_weights)
        self.criterion_emo = nn.BCEWithLogitsLoss(pos_weight=emo_class_weights) if mode in ['multitask', 'sequential'] else None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_emotion: bool = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            return_emotion: Whether to return emotion logits (auto-determined from mode if None)

        Returns:
            toxicity_logits: Logits for toxicity classification
            emotion_logits: Logits for emotion classification (None in baseline mode)
        """
        # Determine whether to return emotion logits
        if return_emotion is None:
            return_emotion = self.mode in ['multitask', 'sequential']

        # Encode input
        outputs = self.shared_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Extract [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)

        # Compute toxicity logits
        toxicity_logits = self.toxicity_head(cls_output)

        # Compute emotion logits if needed
        emotion_logits = None
        if return_emotion and self.emotion_head is not None:
            emotion_logits = self.emotion_head(cls_output)

        return toxicity_logits, emotion_logits

    def compute_loss(
        self,
        toxicity_logits: torch.Tensor,
        toxicity_labels: torch.Tensor,
        emotion_logits: Optional[torch.Tensor] = None,
        emotion_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute loss based on current mode.

        Args:
            toxicity_logits: Predicted toxicity logits
            toxicity_labels: Ground truth toxicity labels
            emotion_logits: Predicted emotion logits (optional)
            emotion_labels: Ground truth emotion labels (optional)

        Returns:
            loss: Computed loss value
        """
        if self.mode == 'baseline':
            # Baseline: only toxicity loss
            return self.criterion_tox(toxicity_logits, toxicity_labels.float())

        elif self.mode == 'multitask':
            # Multitask: can compute either individual losses or combined loss
            # Check which data is provided and compute appropriate loss
            has_tox = toxicity_logits is not None and toxicity_labels is not None
            has_emo = emotion_logits is not None and emotion_labels is not None

            if has_tox and has_emo:
                # Both provided: compute combined weighted loss
                loss_tox = self.criterion_tox(toxicity_logits, toxicity_labels.float())
                loss_emo = self.criterion_emo(emotion_logits, emotion_labels.float())
                return self.lambda_tox * loss_tox + self.lambda_emo * loss_emo
            elif has_tox:
                # Only toxicity: compute toxicity loss only
                return self.criterion_tox(toxicity_logits, toxicity_labels.float())
            elif has_emo:
                # Only emotion: compute emotion loss only
                return self.criterion_emo(emotion_logits, emotion_labels.float())
            else:
                raise ValueError("Must provide either toxicity or emotion logits and labels in multitask mode")

        elif self.mode == 'sequential':
            # Sequential: compute loss based on which labels are provided
            # During emotion pretraining: only emotion loss
            # During toxicity finetuning: only toxicity loss
            if emotion_logits is not None and emotion_labels is not None:
                # Emotion pretraining phase
                return self.criterion_emo(emotion_logits, emotion_labels.float())
            else:
                # Toxicity finetuning phase
                return self.criterion_tox(toxicity_logits, toxicity_labels.float())

    def freeze_encoder(self):
        """Freeze the shared BERT encoder (useful for sequential finetuning)."""
        for param in self.shared_encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze the shared BERT encoder."""
        for param in self.shared_encoder.parameters():
            param.requires_grad = True

    def freeze_toxicity_head(self):
        """Freeze the toxicity classification head."""
        for param in self.toxicity_head.parameters():
            param.requires_grad = False

    def unfreeze_toxicity_head(self):
        """Unfreeze the toxicity classification head."""
        for param in self.toxicity_head.parameters():
            param.requires_grad = True

    def freeze_emotion_head(self):
        """Freeze the emotion classification head."""
        if self.emotion_head is not None:
            for param in self.emotion_head.parameters():
                param.requires_grad = False

    def unfreeze_emotion_head(self):
        """Unfreeze the emotion classification head."""
        if self.emotion_head is not None:
            for param in self.emotion_head.parameters():
                param.requires_grad = True

    def switch_mode(self, new_mode: Literal['baseline', 'multitask', 'sequential']):
        """
        Switch the model's operating mode.

        Useful for sequential training where you might want to change from
        pretraining mode to finetuning mode.

        Args:
            new_mode: New mode to switch to
        """
        if new_mode == 'baseline' and self.emotion_head is None:
            raise ValueError("Cannot switch to baseline mode when emotion head was not initialized")

        self.mode = new_mode

    def get_trainable_params(self) -> dict:
        """
        Get information about which parameters are trainable.

        Returns:
            dict: Dictionary with parameter counts for each component
        """
        encoder_params = sum(p.numel() for p in self.shared_encoder.parameters() if p.requires_grad)
        tox_head_params = sum(p.numel() for p in self.toxicity_head.parameters() if p.requires_grad)
        emo_head_params = sum(p.numel() for p in self.emotion_head.parameters() if p.requires_grad) if self.emotion_head else 0

        return {
            'encoder': encoder_params,
            'toxicity_head': tox_head_params,
            'emotion_head': emo_head_params,
            'total': encoder_params + tox_head_params + emo_head_params
        }
