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
    def __init__(self, bert_model_name='bert-base-uncased', num_toxicity_labels=6, num_emotion_labels=11):
        super(MultiTaskBERT, self).__init__()
        
        # Load pretrained BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Shared BERT Encoder
        self.shared_encoder = self.bert
        
        # Toxicity classification head
        self.toxicity_head = nn.Linear(self.bert.config.hidden_size, num_toxicity_labels)
        
        # Emotion classification head
        self.emotion_head = nn.Linear(self.bert.config.hidden_size, num_emotion_labels)
        
    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.shared_encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use the [CLS] token representation for classification
        cls_output = outputs.last_hidden_state[:, 0, :]  # shape: (batch_size, hidden_size)
        
        # Get logits for both tasks
        toxicity_logits = self.toxicity_head(cls_output)  # shape: (batch_size, num_toxicity_labels)
        emotion_logits = self.emotion_head(cls_output)    # shape: (batch_size, num_emotion_labels)
        
        return toxicity_logits, emotion_logits

def compute_joint_loss(toxicity_logits, emotion_logits, toxicity_labels, emotion_labels, lambda_tox=1.0, lambda_emotion=1.0):
    # Define loss functions
    criterion_tox = nn.BCEWithLogitsLoss()
    criterion_emotion = nn.BCEWithLogitsLoss()
    
    # Compute individual losses
    loss_tox = criterion_tox(toxicity_logits, toxicity_labels)
    loss_emotion = criterion_emotion(emotion_logits, emotion_labels)
    
    # Compute joint loss with adjustable weights
    joint_loss = lambda_tox * loss_tox + lambda_emotion * loss_emotion
    
    return joint_loss

if __name__ == "__main__":
    # Example usage
    model = MultiTaskBERT()
    input_ids = torch.randint(0, 30522, (8, 128))  # Example input
    attention_mask = torch.ones((8, 128))           # Example attention mask
    
    toxicity_logits, emotion_logits = model(input_ids, attention_mask)
    
    print("Toxicity logits shape:", toxicity_logits.shape)
    print("Emotion logits shape:", emotion_logits.shape)
