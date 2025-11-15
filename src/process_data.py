import os
import re
import unicodedata
import pandas as pd
from pathlib import Path

def clean_comment(text: str) -> str:
    """Normalize and clean a single comment for toxicity/emotion modeling."""
    if not isinstance(text, str):
        return ""

    # Normalize unicode accents (e.g., doesn´t -> doesn't)
    text = unicodedata.normalize("NFKD", text)

    # Lowercase
    text = text.lower()

    # Replace underscores and section references (Hubble_Space_Telescope#Impact)
    text = text.replace("_", " ")
    text = re.sub(r"#\S+", "", text)

    # Remove redirect or Wikipedia metadata lines
    text = re.sub(r"^\s*redirect.*", "", text, flags=re.MULTILINE)

    # Replace newlines/tabs with a single space
    text = re.sub(r"\s+", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove Wikipedia references (e.g., WP:RS)
    text = re.sub(r"\bwp:[a-zA-Z0-9]+\b", "", text)

    # Remove stray HTML entities
    text = re.sub(r"&\w+;", " ", text)

    # Clean extra spaces around punctuation
    text = re.sub(r'\s+([?.!,])', r'\1', text)

    # Trim whitespace
    return text.strip()


def process_file(file_path: Path, processed_dir: Path):
    """Load a CSV/TSV, clean comment_text, normalize labels, and save processed version."""
    sep = "\t" if file_path.suffix == ".tsv" else ","
    print(f"Processing {file_path.name} ...")

    # STEP 1 — Load file
    try:
        # TSV = no headers; CSV = infer headers
        df = pd.read_csv(file_path, sep=sep, header=None if file_path.suffix == ".tsv" else "infer")
    except Exception as e:
        print(f"Failed to read {file_path.name}: {e}")
        return

    # CASE A — TSV FILES (GoEmotions-style)
    if file_path.suffix == ".tsv":
        # name the columns explicitly
        df.columns = ["comment_text", "label_encoded", "drop_col"]

        # drop the third column
        df = df.drop(columns=["drop_col"])

        # load emotion map from file
        map_path = file_path.parent / "emotion_map.txt"
        with open(map_path, "r", encoding="utf-8") as f:
            emotion_list = [line.strip() for line in f]

        #  FIX: MULTI-LABEL HANDLING 

        # split "3,10" -> ["3","10"]
        df["label_encoded"] = df["label_encoded"].astype(str).str.split(",")

        # convert each to int
        df["label_encoded"] = df["label_encoded"].apply(
            lambda lst: [int(x) for x in lst]
        )

        # decode to label text
        df["label_text"] = df["label_encoded"].apply(
            lambda lst: [emotion_list[x] for x in lst]
        )

    # CASE B — CSV FILES (Jigsaw Toxicity)
    # Format: id, comment_text, one-hot toxicity labels
    # We convert the one-hot row to a single encoded label
    else:
        # toxicity train columns
        train_cols = ["id", "comment_text",
                    "toxic","severe_toxic","obscene",
                    "threat","insult","identity_hate"]

        if all(col in df.columns for col in train_cols):
            # TRAIN FILE
            one_hot = df[["toxic","severe_toxic","obscene","threat","insult","identity_hate"]]

            # integer encoding
            df["label_encoded"] = one_hot.values.argmax(axis=1)

            # integer → label string
            toxicity_labels = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
            df["label_text"] = df["label_encoded"].apply(lambda x: toxicity_labels[x])

            # keep relevant columns
            df = df[["id","comment_text","label_encoded","label_text"]]

        elif all(col in df.columns for col in ["id", "comment_text"]):
            # TEST FILE (no labels)
            df["label_encoded"] = None
            df["label_text"] = None

            df = df[["id","comment_text","label_encoded","label_text"]]

        else:
            print(f"Skipped {file_path.name}: unknown CSV format.")
            return

    # STEP 3 — Cleaning the text
    df["clean_text"] = df["comment_text"].apply(clean_comment)

    # STEP 4 — Save processed file
    output_file = processed_dir / f"{file_path.stem}_processed.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved cleaned file to: {output_file}\n")


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / "data"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    files = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.tsv"))
    if not files:
        print("No CSV or TSV files found in data/ directory.")
    else:
        print(f"Found {len(files)} file(s) to process.\n")

    for file_path in files:
        if not file_path.name.endswith("_processed.csv"):
            process_file(file_path, processed_dir)

    print("All files processed.")
