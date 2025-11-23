# multitask_bert.py
"""
This script does the following:
1. Load's in pretrained BERT model
2. create our models arch (shared BERT Encoder -> toxicity or emotion head)
3. Defines the joint loss fn (L = lambda(tox)*L_tox + lambda(emotion)*L_emotion)
4. Implements adjustable loss weighting
5. build the train loop for model
"""
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F

class MultiTaskBERT(nn.Module):
    def __init__(
        self,
        bert_model_name: str = 'bert-base-uncased',
        num_toxicity_labels: int = 6,
        num_emotion_labels: int = 28, 
        dropout_prob: float = 0.1,
        lambda_tox: float = 1.0,
        lambda_emo: float = 1.0
    ):
        super().__init__()

        # Shared encoder
        self.shared_encoder = BertModel.from_pretrained(bert_model_name)

        self.dropout = nn.Dropout(self.shared_encoder.config.hidden_dropout_prob)

        # Task-specific heads
        hidden_size = self.shared_encoder.config.hidden_size
        self.toxicity_head = nn.Linear(hidden_size, num_toxicity_labels)
        self.emotion_head = nn.Linear(hidden_size, num_emotion_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.shared_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS embedding
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)

        # Task logits
        toxicity_logits = self.toxicity_head(cls_output)
        emotion_logits = self.emotion_head(cls_output)

        return toxicity_logits, emotion_logits


# Joint Loss Function
def compute_joint_loss(
    toxicity_logits, emotion_logits,
    toxicity_labels, emotion_labels,
    lambda_tox=1.0, lambda_emotion=1.0
):
    """
    Multi-task loss:
    L = λ_tox * BCEWithLogits(tox) + λ_emo * BCEWithLogits(emotion)
    """
    # Both are now multi-label
    criterion_tox = nn.BCEWithLogitsLoss()
    criterion_emo = nn.BCEWithLogitsLoss()

    loss_tox = criterion_tox(toxicity_logits, toxicity_labels.float())
    loss_emo = criterion_emo(emotion_logits, emotion_labels.float())

    return lambda_tox * loss_tox + lambda_emotion * loss_emo

if __name__ == "__main__":
    print("Testing MultiTaskBERT model...")
    
    model = MultiTaskBERT()

    # Dummy inputs
    batch_size = 8
    seq_length = 128
    
    input_ids = torch.randint(0, 30522, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))

    # Forward pass
    tox_logits, emo_logits = model(input_ids, attention_mask)

    print(f"Toxicity logits shape: {tox_logits.shape}")
    print(f"Emotion logits shape: {emo_logits.shape}")
    
    # Test loss computation
    dummy_tox_labels = torch.rand((batch_size, 6))
    dummy_tox_labels = (dummy_tox_labels > 0.5).float()  # Multi-hot encoding
    dummy_emo_labels = torch.rand((batch_size, 28))
    dummy_emo_labels = (dummy_emo_labels > 0.5).float()
    
    # Compute joint loss
    loss = compute_joint_loss(
        tox_logits, emo_logits,
        dummy_tox_labels, dummy_emo_labels,
        lambda_tox=1.0, lambda_emotion=1.0
    )
    
    print(f"Combined loss: {loss.item():.4f}")
    print("\nModel architecture test passed!")