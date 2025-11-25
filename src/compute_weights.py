# compute_weights.py
"""
Compute class weights for toxicity and emotion datasets.

This script should be run once after processing the data.
It loads the processed CSVs, converts label_encoded from string→list,
computes frequency-based class weights, and saves them into weights.json
so training scripts never need to recompute them.
"""

import json
import ast
from pathlib import Path
import pandas as pd
import torch


def load_label_matrix(csv_path: Path, num_classes: int, is_emotion: bool) -> torch.Tensor:
    df = pd.read_csv(csv_path)

    # Toxicity: already multi-hot
    if not is_emotion:
        import ast
        df["label_encoded"] = df["label_encoded"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        return torch.tensor(df["label_encoded"].tolist(), dtype=torch.float32)

    # Emotion: convert list-of-indices to multi-hot
    vectors = []
    import ast
    for x in df["label_encoded"]:
        if isinstance(x, str):
            x = ast.literal_eval(x)

        vec = torch.zeros(num_classes)
        for idx in x:
            vec[idx] = 1.0

        vectors.append(vec)

    return torch.stack(vectors)


def compute_class_weights(label_matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute class weights using inverse frequency:
        weight[c] = N_total / (num_pos[c] + eps)
    """
    eps = 1e-6
    pos_counts = label_matrix.sum(dim=0)
    N = label_matrix.shape[0]


    pos_weight = (N - pos_counts) / (pos_counts + 1e-6)
    return pos_weight


def compute_all_weights(root: Path):
    """
    Loads toxicity and emotion processed CSV files,
    computes weights, and writes weights.json.
    """

    processed = root / "data" / "processed"
    sampled = root / "data" / "processed_sampled"

    output_path = root / "data" / "weights.json"
    result = {}

    # ---- Toxicity ----
    for subdir_name in ["processed", "processed_sampled"]:
        subdir = root / "data" / subdir_name
        tox_csv = subdir / "toxicity_raw_processed.csv"

        if tox_csv.exists():
            print(f"Computing TOXICITY weights from: {tox_csv}")
            label_matrix = load_label_matrix(tox_csv, num_classes=6, is_emotion=False)
            weights = compute_class_weights(label_matrix)
            result[f"toxicity_{subdir_name}"] = weights.tolist()

    # ---- Emotion ----
    for subdir_name in ["processed", "processed_sampled"]:
        subdir = root / "data" / subdir_name
        emo_csv = subdir / "emotion_raw_all_processed.csv"

        if emo_csv.exists():
            print(f"Computing EMOTION weights from: {emo_csv}")
            label_matrix = load_label_matrix(emo_csv, num_classes=28, is_emotion=True)
            weights = compute_class_weights(label_matrix)
            result[f"emotion_{subdir_name}"] = weights.tolist()

    # save JSON
    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)

    print(f"\nSaved class weights → {output_path}")

def load_class_weights(task: str, sampled: bool = False):
    """
    Load class weights for a given task from weights.json.

    Args:
        task (str): "toxicity" or "emotion"
        sampled (bool): Whether to load sampled weights (True) or processed (False)

    Returns:
        torch.Tensor or None
    """

    import json
    weights_path = Path(__file__).resolve().parent.parent / "data" / "weights.json"

    if not weights_path.exists():
        raise FileNotFoundError(
            f"weights.json not found at {weights_path}. "
            "Run compute_weights.py first!"
        )

    with open(weights_path, "r") as f:
        weights_dict = json.load(f)

    key = f"{task}_{'processed_sampled' if sampled else 'processed'}"

    if key not in weights_dict:
        raise KeyError(f"Weight key '{key}' not found in weights.json.")

    weights = torch.tensor(weights_dict[key], dtype=torch.float32)
    return weights

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    compute_all_weights(root)
