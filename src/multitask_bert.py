# multitask_bert.py
"""
This script does the following:
1. Load's in pretrained BERT model
2. creat our models arch (shared BERT Encoder -> toxicity or emotion head)
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
        num_emotion_labels: int = 28,      # GoEmotions = 28 labels
        dropout_prob: float = 0.1,
        lambda_tox: float = 1.0,
        lambda_emo: float = 1.0
    ):
        super().__init__()

        self.lambda_tox = lambda_tox
        self.lambda_emo = lambda_emo

        # Load pretrained BERT
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)

        # Shared encoder
        self.shared_encoder = self.bert

        # Task-specific heads
        hidden_size = self.bert.config.hidden_size
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
    L = λ_tox * BCE(tox) + λ_emo * BCE(emotion)
    """
    criterion = nn.BCEWithLogitsLoss()

    loss_tox = criterion(toxicity_logits, toxicity_labels)
    loss_emo = criterion(emotion_logits, emotion_labels)

    return lambda_tox * loss_tox + lambda_emotion * loss_emo

if __name__ == "__main__":
    model = MultiTaskBERT()

    # dummy inputs
    input_ids = torch.randint(0, 30522, (8, 128))
    attention_mask = torch.ones((8, 128))

    tox_logits, emo_logits = model(input_ids, attention_mask)

    print("Toxicity logits:", tox_logits.shape)   # (8, 6)
    print("Emotion logits:", emo_logits.shape)     # (8, 28)