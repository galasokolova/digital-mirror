
from pathlib import Path

import pandas as pd

from taxonomy.intent_mapping import INTENT_CLASSES

from preprocessing.intent_text_normalizer import (
    normalize_text,
    remove_punctuation,
)


ML_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    ML_ROOT
    / "data"
    / "evaluation"
    / "digital_mirror_final"
)

INPUT_PATH = DATASET_DIR / "test.csv"

OUTPUT_PATH = DATASET_DIR / "test_processed.csv"

EXPECTED_RECORDS = 240

EXPECTED_COLUMNS = [
    "id",
    "text",
    "intent",
    "group",
    "source",
]

OUTPUT_COLUMNS = [
    *EXPECTED_COLUMNS,
    "normalized_text",
    "model_text",
]


def load_dataset():

    dataset = pd.read_csv(
        INPUT_PATH,
        dtype=str,
        keep_default_na=False
    )

    if dataset.columns.tolist() != EXPECTED_COLUMNS:

        raise ValueError(
            "Unexpected dataset columns."
        )

    return dataset


def preprocess_dataset(dataset):

    processed = dataset.copy()

    processed["normalized_text"] = (
        processed["text"]
        .apply(normalize_text)
    )

    processed["model_text"] = (
        processed["normalized_text"]
        .apply(remove_punctuation)
    )

    return processed


def analyze_changes(dataset):

    print("\n=== PREPROCESSING STATISTICS ===")

    total = len(dataset)

    normalized_changes = (
        dataset["text"]
        != dataset["normalized_text"]
    ).sum()

    punctuation_changes = (
        dataset["normalized_text"]
        != dataset["model_text"]
    ).sum()

    total_changes = (
        dataset["text"]
        != dataset["model_text"]
    ).sum()

    changes = {

        "Normalization changes":
            normalized_changes,

        "Punctuation removal changes":
            punctuation_changes,

        "Total changed records":
            total_changes,
    }

    for name, count in changes.items():

        percentage = count / total * 100

        print(
            f"{name}: {count} "
            f"({percentage:.2f}%)"
        )


def print_examples(dataset):

    print("\n=== PREPROCESSING EXAMPLES ===")

    for intent in INTENT_CLASSES:

        subset = dataset.loc[
            dataset["intent"] == intent
        ]

        example = subset.sample(
            n=1,
            random_state=42
        ).iloc[0]

        print(f"\nIntent: {intent}")

        print(
            f"Original:   {example['text']}"
        )

        print(
            f"Normalized: "
            f"{example['normalized_text']}"
        )

        print(
            f"Model text: {example['model_text']}"
        )


def validate_preprocessing(original, processed):

    print("\n=== PREPROCESSING VALIDATION ===")

    repeated_normalization = (
        processed["model_text"]
        .apply(remove_punctuation)
    )

    checks = {

        "Expected input record count":
            len(original) == EXPECTED_RECORDS,

        "Expected output record count":
            len(processed) == EXPECTED_RECORDS,

        "No records removed":
            len(original) == len(processed),

        "Original columns preserved":
            original.equals(
                processed[EXPECTED_COLUMNS]
            ),

        "Unique IDs":
            processed["id"].is_unique,

        "No empty normalized texts":
            not processed["normalized_text"]
            .eq("").any(),

        "No empty model texts":
            not processed["model_text"]
            .eq("").any(),

        "Unique normalized texts":
            processed["normalized_text"].is_unique,

        "Unique model texts":
            processed["model_text"].is_unique,

        "All intent classes preserved":
            set(processed["intent"])
            == set(INTENT_CLASSES),

        "Class distribution preserved":
            original["intent"].value_counts().equals(
                processed["intent"].value_counts()
            ),

        "Group distribution preserved":
            original["group"].value_counts().equals(
                processed["group"].value_counts()
            ),

        "Preprocessing is idempotent":
            repeated_normalization.equals(
                processed["model_text"]
            ),

        "Expected output columns":
            processed.columns.tolist()
            == OUTPUT_COLUMNS,
    }

    for name, passed in checks.items():

        status = "PASS" if passed else "FAIL"

        print(f"{status} | {name}")

    print(
        f"\nChecks passed: "
        f"{sum(checks.values())} / {len(checks)}"
    )

    if not all(checks.values()):

        raise ValueError(
            "Preprocessing validation failed."
        )


def save_dataset(dataset):

    dataset[OUTPUT_COLUMNS].to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    print("\n=== OUTPUT DATASET ===")

    print(f"Saved to: {OUTPUT_PATH}")

    print(f"Records: {len(dataset)}")

    print(f"Columns: {len(OUTPUT_COLUMNS)}")


def main():

    print(
        "=== E09.06.2c: FINAL INSTALLATION "
        "DATASET PREPROCESSING ==="
    )

    original = load_dataset()

    print("\n=== INPUT DATASET ===")

    print(f"Path: {INPUT_PATH}")

    print(f"Records: {len(original)}")

    processed = preprocess_dataset(
        original
    )

    analyze_changes(processed)

    print_examples(processed)

    validate_preprocessing(
        original,
        processed
    )

    save_dataset(processed)

    print("\nOriginal dataset modified: NO")
    print("Model loaded: NO")
    print("Model training performed: NO")

    print("\n=== PREPROCESSING COMPLETE ===")


if __name__ == "__main__":
    main()
