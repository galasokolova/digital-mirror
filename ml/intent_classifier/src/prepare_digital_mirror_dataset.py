
from pathlib import Path

import pandas as pd

from taxonomy.intent_mapping import INTENT_CLASSES

from preprocessing.intent_text_normalizer import (
    normalize_text,
    remove_punctuation,
)


ML_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ML_ROOT
    / "data"
    / "raw"
    / "digital_mirror"
    / "digital_mirror_intents_v1.csv"
)

OUTPUT_DIR = (
    ML_ROOT
    / "data"
    / "processed"
    / "digital_mirror_v1"
)

OUTPUT_PATH = OUTPUT_DIR / "train.csv"

EXPECTED_ROWS = 600


def main():

    print("=== E09.04.3: DIGITAL MIRROR PREPROCESSING ===")

    df = pd.read_csv(
        INPUT_PATH,
        dtype=str,
        keep_default_na=False
    )

    original_count = len(df)

    print(f"\nInput records: {original_count:,}")

    # Apply the existing text normalizer

    df["normalized_text"] = df["text"].apply(
        normalize_text
    )

    df["model_text"] = df["text"].apply(
        remove_punctuation
    )

    # Analyze text transformations

    normalization_changes = (
        df["text"] != df["normalized_text"]
    ).sum()

    punctuation_changes = (
        df["normalized_text"] != df["model_text"]
    ).sum()

    total_changes = (
        df["text"] != df["model_text"]
    ).sum()

    print("\n=== TEXT TRANSFORMATIONS ===")

    print(
        f"Normalization changes: "
        f"{normalization_changes}"
    )

    print(
        f"Punctuation removal changes: "
        f"{punctuation_changes}"
    )

    print(
        f"Total changed texts: "
        f"{total_changes}"
    )

    # Display examples

    print("\n=== PREPROCESSING EXAMPLES ===")

    samples = df.sample(
        n=12,
        random_state=42
    )

    for _, row in samples.iterrows():

        print(f"\nID: {row['id']}")
        print(f"Intent: {row['intent']}")
        print(f"Original:   {row['text']}")
        print(f"Normalized: {row['normalized_text']}")
        print(f"Model text: {row['model_text']}")

    # Validate the processed dataset

    print("\n=== VALIDATION ===")

    distribution = (
        df["intent"]
        .value_counts()
        .reindex(INTENT_CLASSES, fill_value=0)
    )

    idempotent = (
        df["model_text"].apply(remove_punctuation)
        == df["model_text"]
    ).all()

    checks = {

        "Expected record count":
            len(df) == EXPECTED_ROWS,

        "No records removed":
            len(df) == original_count,

        "Unique IDs":
            df["id"].is_unique,

        "No empty model texts":
            not df["model_text"].eq("").any(),

        "Unique model texts":
            df["model_text"].is_unique,

        "All intent classes present":
            set(df["intent"]) == set(INTENT_CLASSES),

        "Balanced class distribution":
            distribution.eq(50).all(),

        "Normalization is idempotent":
            idempotent,
    }

    for name, passed in checks.items():

        status = "PASS" if passed else "FAIL"

        print(f"{status:4s} | {name}")

    if not all(checks.values()):

        raise ValueError(
            "Preprocessing validation failed."
        )

    # Save the processed dataset

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    print("\n=== OUTPUT DATASET ===")

    print(f"Records: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    print("\nClass distribution:")

    print(distribution.to_string())

    print(f"\nSaved to: {OUTPUT_PATH}")

    print("\n=== PREPROCESSING COMPLETE ===")


if __name__ == "__main__":
    main()
