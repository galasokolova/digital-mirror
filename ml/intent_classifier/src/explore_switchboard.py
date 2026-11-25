from pathlib import Path

import pandas as pd


# ==================================================
# Configuration
# ==================================================

ML_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    ML_ROOT
    / "data"
    / "raw"
    / "switchboard"
    / "swda"
)

COLUMNS = [
    "swda_filename",
    "conversation_no",
    "transcript_index",
    "act_tag",
    "caller",
    "utterance_index",
    "subutterance_index",
    "text",
]


# ==================================================
# Load dataset
# ==================================================

def load_dataset():
    files = sorted(DATASET_DIR.rglob("*.utt.csv"))

    if not files:
        raise FileNotFoundError(
            f"No transcript files found in {DATASET_DIR}"
        )

    print(f"Transcript files found: {len(files)}")

    frames = []

    for file in files:
        df = pd.read_csv(
            file,
            usecols=COLUMNS,
            dtype={"act_tag": str, "text": str}
        )

        frames.append(df)

    return pd.concat(frames, ignore_index=True)


# ==================================================
# Dataset overview
# ==================================================

def explore_dataset(df):
    print("\n=== DATASET OVERVIEW ===")

    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")

    print(
        f"Unique conversations: "
        f"{df['conversation_no'].nunique():,}"
    )

    print(
        f"Unique raw labels: "
        f"{df['act_tag'].nunique():,}"
    )

    print("\n=== COLUMN TYPES ===")
    print(df.dtypes)

    print("\n=== MISSING VALUES ===")
    print(df.isna().sum())

    print("\n=== DUPLICATE ROWS ===")
    print(df.duplicated().sum())

    print("\n=== EMPTY TEXTS ===")

    empty_texts = (
        df["text"]
        .fillna("")
        .str.strip()
        .eq("")
        .sum()
    )

    print(empty_texts)


# ==================================================
# Label distribution
# ==================================================

def explore_labels(df):
    print("\n=== RAW LABEL DISTRIBUTION ===")

    counts = df["act_tag"].value_counts(dropna=False)

    distribution = pd.DataFrame({
        "count": counts,
        "percentage": (
            counts / len(df) * 100
        ).round(3)
    })

    print(distribution.to_string())

    print("\n=== CONTINUATION ROWS ===")

    continuations = df["act_tag"].eq("+").sum()

    print(f"Count: {continuations:,}")
    print(f"Percentage: {continuations / len(df):.2%}")


# ==================================================
# Text length analysis
# ==================================================

def explore_text_lengths(df):
    print("\n=== RAW TEXT LENGTH STATISTICS ===")

    texts = df["text"].dropna()

    word_counts = texts.str.split().str.len()

    print(word_counts.describe().to_string())

    print("\n=== SHORT TEXTS ===")
    print(f"One token: {(word_counts == 1).sum():,}")
    print(f"Two tokens: {(word_counts == 2).sum():,}")
    print(f"Three tokens: {(word_counts == 3).sum():,}")


# ==================================================
# Main
# ==================================================

def main():
    print("=== E01: SWITCHBOARD DATASET EXPLORATION ===")
    print(f"Dataset directory: {DATASET_DIR}")

    df = load_dataset()

    explore_dataset(df)
    explore_labels(df)
    explore_text_lengths(df)

    print("\n=== EXPLORATION COMPLETE ===")


if __name__ == "__main__":
    main()