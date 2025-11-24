# process_data.py
import os
import re
import unicodedata
import pandas as pd
from pathlib import Path
import argparse     # added argparse

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

def combine_emotion_tsvs(data_dir: Path) -> Path:
    """Combine all emotion_raw*.tsv files into one master TSV."""
    emotion_files = sorted(data_dir.glob("emotion_raw*.tsv"))
    if not emotion_files or len(emotion_files) == 1:
        # Nothing to combine
        return None

    print(f"Combining {len(emotion_files)} emotion TSV files...")

    # Load & concat
    dfs = []
    for f in emotion_files:
        df = pd.read_csv(f, sep="\t", header=None)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Save combined file
    combined_path = data_dir / "emotion_raw_all.tsv"
    combined.to_csv(combined_path, sep="\t", header=False, index=False)

    print(f"Saved combined emotion TSV to: {combined_path}")
    return combined_path


def process_file(file_path: Path, processed_dir: Path, sample: bool):
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

        # DOWN-SAMPLE "neutral" ONLY ROWS #
        if sample:
            neutral_alone_mask = df["label_text"].apply(lambda x: x == ["neutral"])
            neutral_alone = df[neutral_alone_mask]

            # Decide how many to keep
            target_size = min(len(neutral_alone), 5000)
            print(f"Down-sampling 'neutral' only rows from {len(neutral_alone)} to {target_size}.")

            # Random sample
            neutral_down = neutral_alone.sample(n=target_size, random_state=42)

            # Keep all non-neutral rows
            non_neutral = df[~neutral_alone_mask]

            # Recombine
            df = pd.concat([non_neutral, neutral_down], ignore_index=True)

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

            # DOWN-SAMPLE "toxic" CLASS #
            if sample:
                toxic_mask = df["label_text"] == "toxic"
                toxic_df = df[toxic_mask]

                # choose target size (12k recommended)
                target_size = min(len(toxic_df), 12_000)
                print(f"Down-sampling toxic from {len(toxic_df)} → {target_size}")

                toxic_down = toxic_df.sample(n=target_size, random_state=42)

                # keep all minority classes untouched
                minorities = df[~toxic_mask]

                # recombine
                df = pd.concat([minorities, toxic_down], ignore_index=True)

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
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=str, default="False",
                        help="True/False — whether to down-sample datasets.")
    args = parser.parse_args()

    # normalize string to bool
    sample_flag = args.sample.lower() == "true"

    data_dir = Path(__file__).resolve().parent.parent / "data"

    # choose processed output directory based on flag
    processed_dir = data_dir / ("processed_sampled" if sample_flag else "processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("Scanning for raw files...\n")

    ## Combine emotion_raw_train.tsv + emotion_raw_test.tsv ##
    combined_emotion_path = combine_emotion_tsvs(data_dir)

    # Collect all files for processing
    files = [
        f for f in (list(data_dir.glob("*.csv")) + list(data_dir.glob("*.tsv")))
        if not f.name.endswith("_processed.csv")
        and not f.name.endswith("_processed_sampled.csv")
    ]

    # If combined file exists, remove the original separate TSVs from list
    if combined_emotion_path:
        files = [
            f for f in files
            if not (f.name.startswith("emotion_raw") and f.suffix == ".tsv")
        ]
        files.append(combined_emotion_path)

    print(f"Found {len(files)} file(s) to process.\n")

    ## Process each file normally ##
    for file_path in files:
        process_file(file_path, processed_dir, sample_flag)

    print("All files processed.")
