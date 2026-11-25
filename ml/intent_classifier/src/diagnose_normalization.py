
from pathlib import Path

import pandas as pd

from preprocessing.intent_text_normalizer import (
    normalize_text,
    remove_punctuation,
)


ML_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ML_ROOT
    / "data"
    / "processed"
    / "intent_v1"
    / "train.csv"
)


def main():

    print("=== NORMALIZATION DIAGNOSTICS ===")

    dataset = pd.read_csv(
        DATA_PATH,
        dtype=str,
        keep_default_na=False
    )

    print(f"\nRecords: {len(dataset)}")

    print("\n=== COLUMNS ===")

    for column in dataset.columns:
        print(column)

    # Test A: original text -> model text

    calculated_a = (
        dataset["text"]
        .map(normalize_text)
        .map(remove_punctuation)
    )

    mismatch_a = (
        dataset["model_text"] != calculated_a
    )

    print("\n=== TEST A: TEXT -> MODEL TEXT ===")

    print(f"Matching: {(~mismatch_a).sum()}")
    print(f"Mismatching: {mismatch_a.sum()}")
    print(f"Mismatch rate: {mismatch_a.mean():.4%}")

    columns = [
        column
        for column in [
            "text",
            "normalized_text",
            "model_text",
            "intent",
        ]
        if column in dataset.columns
    ]

    examples = dataset.loc[
        mismatch_a, columns
    ].copy()

    examples["calculated_model_text"] = (
        calculated_a.loc[mismatch_a]
    )

    print("\n=== MISMATCH EXAMPLES ===")

    for index, row in examples.head(20).iterrows():

        print(f"\nRecord index: {index}")

        for column, value in row.items():
            print(f"{column}: {value!r}")

    # Test B: normalized text -> model text

    if "normalized_text" in dataset.columns:

        calculated_b = (
            dataset["normalized_text"]
            .map(remove_punctuation)
        )

        mismatch_b = (
            dataset["model_text"] != calculated_b
        )

        print(
            "\n=== TEST B: "
            "NORMALIZED TEXT -> MODEL TEXT ==="
        )

        print(f"Matching: {(~mismatch_b).sum()}")
        print(f"Mismatching: {mismatch_b.sum()}")

        print(
            f"Mismatch rate: "
            f"{mismatch_b.mean():.4%}"
        )

        # Test C: original text -> normalized text

        calculated_c = (
            dataset["text"]
            .map(normalize_text)
        )

        mismatch_c = (
            dataset["normalized_text"]
            != calculated_c
        )

        print(
            "\n=== TEST C: "
            "TEXT -> NORMALIZED TEXT ==="
        )

        print(f"Matching: {(~mismatch_c).sum()}")
        print(f"Mismatching: {mismatch_c.sum()}")

        print(
            f"Mismatch rate: "
            f"{mismatch_c.mean():.4%}"
        )

    print("\n=== EMPTY MODEL TEXTS ===")

    empty = (
        dataset["model_text"]
        .str.strip()
        .eq("")
        .sum()
    )

    print(f"Empty model texts: {empty}")

    print("\nSource dataset modified: NO")
    print("Model loaded: NO")
    print("Model trained: NO")

    print("\n=== DIAGNOSTICS FINISHED ===")


if __name__ == "__main__":
    main()
