# prepare_data.py
"""
This file takes the processed data files, and then prepares them for the modelling.
For the toxicity dataset: tokenize, split into train/val/test, save as torch datasets
For the emotion dataset: conver to multi-hot vector, tokenize, split into train/val/test, save as torch datasets
"""
import os
from pathlib import Path
import pandas as pd
import torch
from transformers import BertTokenizerFast
from sklearn.model_selection import train_test_split

# Load BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


# Convert a list of emotion indices into a multi-hot vector
def to_multi_hot(indices, num_classes=28):
    """
    Convert list of emotion indices [3,10] into multi-hot vector of length 28.
    """
    vec = torch.zeros(num_classes, dtype=torch.float32)
    for idx in indices:
        vec[idx] = 1.0
    return vec


# Tokenize a batch of text
def tokenize_texts(text_list, max_len=128):
    """
    Tokenize list of texts → return dict of tensors.
    """
    return tokenizer(
        text_list,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )


# Process one dataset (toxicity or emotion)
def process_dataset(csv_path, output_dir, is_emotion=False):
    """
    Load cleaned CSV and produce tokenized tensors with labels.
    """

    # load CSV
    df = pd.read_csv(csv_path)

    df = df[df["clean_text"].notnull()]    
    df = df[df["clean_text"].str.strip() != ""]  

    # enforce clean_text as string
    texts = df["clean_text"].astype(str).tolist()

    # tokenize clean_text
    tokenized = tokenize_texts(texts)

    # build label tensor
    if is_emotion:
        # emotion labels (multi-label)
        labels = torch.stack([
            to_multi_hot(eval(row)) for row in df["label_encoded"]
        ])
    else:
        # toxicity labels (single integer)
        labels = torch.tensor(df["label_encoded"].tolist(), dtype=torch.long)

    # split train/val/test (80/10/10)
    ## Train, test/val
    train_idx, temp_idx = train_test_split(
        list(range(len(df))), test_size=0.20, random_state=42
    )
    ## Val, test
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=42
    )

    # helper to subset tensors
    def subset(tensor_dict, indices):
        return {k: v[indices] for k, v in tensor_dict.items()}

    # subset tokenized tensors
    train_tokens = subset(tokenized, train_idx)
    val_tokens = subset(tokenized, val_idx)
    test_tokens = subset(tokenized, test_idx)

    # subset labels
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    # save all splits
    torch.save({"tokens": train_tokens, "labels": train_labels},
               output_dir / "train.pt")
    torch.save({"tokens": val_tokens, "labels": val_labels},
               output_dir / "val.pt")
    torch.save({"tokens": test_tokens, "labels": test_labels},
               output_dir / "test.pt")

    print(f"Saved tokenized splits to: {output_dir}")


# Main entrypoint
if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    processed_dir = root / "data" / "processed"
    tokenized_dir = processed_dir / "tokenized"
    tokenized_dir.mkdir(parents=True, exist_ok=True)

    # identify processed CSVs
    toxicity_file = processed_dir / "toxicity_raw_processed.csv"
    emotion_train_file = processed_dir / "emotion_raw_train_processed.csv"
    emotion_test_file = processed_dir / "emotion_raw_test_processed.csv"

    # combine emotion train+test for split later
    emotion_combined = processed_dir / "emotion_all.csv"

    # combine emotion files only if not created
    if not emotion_combined.exists():
        df1 = pd.read_csv(emotion_train_file)
        df2 = pd.read_csv(emotion_test_file)
        pd.concat([df1, df2], ignore_index=True).to_csv(emotion_combined, index=False)
        print("Created combined emotion dataset.")

    # process toxicity
    tox_out = tokenized_dir / "toxicity"
    tox_out.mkdir(exist_ok=True)
    process_dataset(toxicity_file, tox_out, is_emotion=False)

    # process emotion
    emo_out = tokenized_dir / "emotion"
    emo_out.mkdir(exist_ok=True)
    process_dataset(emotion_combined, emo_out, is_emotion=True)

    print("All tokenized datasets saved.")
