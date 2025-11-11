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
    """Load a CSV/TSV, clean comment_text, and save processed version."""
    sep = "\t" if file_path.suffix == ".tsv" else ","
    print(f"Processing {file_path.name} ...")

    try:
        df = pd.read_csv(file_path, sep=sep, header=None if file_path.suffix == ".tsv" else 'infer')
    except Exception as e:
        print(f"Failed to read {file_path.name}: {e}")
        return

    # Handle files without proper headers (this is for the emotion data)
    if all(str(col).isdigit() for col in df.columns):
        # Guess the likely structure, assume comment text is in the first column
        df.columns = ["comment_text"] + [f"label_{i}" for i in range(1, len(df.columns))]
        print(f"File {file_path.name} had no headers. Assigned default column names.")
    elif "comment_text" not in df.columns:
        print(f"Skipped {file_path.name}: no 'comment_text' column found.")
        return

    # Apply cleaning
    df["clean_text"] = df["comment_text"].apply(clean_comment)

    # Save output
    output_file = processed_dir / f"{file_path.stem}_processed.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved cleaned file to: {output_file}\n")


if __name__ == "__main__":
    data_dir = Path("../data")
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    files = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.tsv"))
    if not files:
        print("No CSV or TSV files found in data/ directory.")
    else:
        print(f"Found {len(files)} file(s) to process.\n")

    for file_path in files:
        process_file(file_path, processed_dir)

    print("All files processed.")
