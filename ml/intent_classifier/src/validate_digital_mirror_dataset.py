
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

REQUIRED_COLUMNS = {
    "id",
    "text",
    "intent",
    "source",
}

EXPECTED_ROWS = 600
EXPECTED_PER_CLASS = 50
RANDOM_STATE = 42


def main():

    print("=== E09.04.2: DIGITAL MIRROR DATASET VALIDATION ===")

    df = pd.read_csv(
        INPUT_PATH,
        dtype=str,
        keep_default_na=False
    )

    print("\n=== DATASET OVERVIEW ===")

    print(f"Records: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # Create temporary normalized representations

    df["normalized_text"] = df["text"].apply(
        normalize_text
    )

    df["model_text"] = df["text"].apply(
        remove_punctuation
    )

    # Basic structural validation

    print("\n=== STRUCTURAL VALIDATION ===")

    missing_values = df.isna().sum().sum()

    empty_fields = (
        df[sorted(REQUIRED_COLUMNS)]
        .apply(lambda column: column.str.strip().eq(""))
        .sum()
    )

    duplicate_ids = df["id"].duplicated().sum()

    duplicate_rows = df[
        sorted(REQUIRED_COLUMNS)
    ].duplicated().sum()

    empty_model_texts = df["model_text"].eq("").sum()

    invalid_labels = sorted(
        set(df["intent"]) - set(INTENT_CLASSES)
    )

    missing_labels = sorted(
        set(INTENT_CLASSES) - set(df["intent"])
    )

    print(f"Missing values: {missing_values}")

    print("\nEmpty required fields:")
    print(empty_fields.to_string())

    print(f"\nDuplicate IDs: {duplicate_ids}")
    print(f"Duplicate rows: {duplicate_rows}")
    print(f"Empty model texts: {empty_model_texts}")
    print(f"Invalid labels: {invalid_labels}")
    print(f"Missing labels: {missing_labels}")

    # Class distribution

    print("\n=== CLASS DISTRIBUTION ===")

    distribution = (
        df["intent"]
        .value_counts()
        .reindex(INTENT_CLASSES, fill_value=0)
    )

    distribution_table = pd.DataFrame({
        "records": distribution,
        "percentage": (
            distribution / len(df) * 100
        ).round(2),
    })

    print(distribution_table.to_string())

    print(f"\nTotal classes: {df['intent'].nunique()}")

    # Duplicate text analysis

    print("\n=== DUPLICATE TEXT ANALYSIS ===")

    raw_duplicates = df["text"].duplicated().sum()

    normalized_duplicates = (
        df["normalized_text"].duplicated().sum()
    )

    model_duplicates = (
        df["model_text"].duplicated().sum()
    )

    print(f"Raw text duplicates: {raw_duplicates}")

    print(
        f"Normalized text duplicates: "
        f"{normalized_duplicates}"
    )

    print(
        f"Model text duplicates: "
        f"{model_duplicates}"
    )

    duplicate_records = df[
        df["model_text"].duplicated(keep=False)
    ]

    if not duplicate_records.empty:

        print("\nDuplicate records:")

        print(
            duplicate_records[
                ["id", "text", "model_text", "intent"]
            ]
            .sort_values("model_text")
            .to_string(index=False)
        )

    # Conflicting intent labels

    print("\n=== LABEL CONFLICT ANALYSIS ===")

    label_counts = (
        df.groupby("model_text")["intent"]
        .nunique()
    )

    conflicting_texts = label_counts[
        label_counts > 1
    ].index

    conflicting_records = df[
        df["model_text"].isin(conflicting_texts)
    ]

    print(
        f"Conflicting unique texts: "
        f"{len(conflicting_texts)}"
    )

    print(
        f"Conflicting records: "
        f"{len(conflicting_records)}"
    )

    if not conflicting_records.empty:

        print(
            conflicting_records[
                ["id", "text", "model_text", "intent"]
            ]
            .sort_values("model_text")
            .to_string(index=False)
        )

    # Text length analysis

    print("\n=== TEXT LENGTH ANALYSIS ===")

    df["word_count"] = (
        df["model_text"].str.split().str.len()
    )

    print(
        df["word_count"]
        .describe()
        .round(2)
        .to_string()
    )

    short_texts = df["word_count"].between(1, 3)

    print(
        f"\nShort texts (1-3 words): "
        f"{short_texts.sum()}"
    )

    print(
        f"Short text percentage: "
        f"{short_texts.mean() * 100:.2f}%"
    )

    print("\nText length by intent:")

    length_table = (
        df.groupby("intent")["word_count"]
        .agg(["min", "max", "mean", "median"])
        .reindex(INTENT_CLASSES)
        .round(2)
    )

    print(length_table.to_string())

    # Vocabulary analysis

    print("\n=== VOCABULARY ANALYSIS ===")

    vocabulary = set(
        " ".join(df["model_text"]).split()
    )

    print(f"Unique words: {len(vocabulary)}")

    vocabulary_by_class = []

    for intent in INTENT_CLASSES:

        subset = df[df["intent"] == intent]

        words = " ".join(
            subset["model_text"]
        ).split()

        unique_words = set(words)

        vocabulary_by_class.append({
            "intent": intent,
            "total_words": len(words),
            "unique_words": len(unique_words),
        })

    print(
        pd.DataFrame(vocabulary_by_class)
        .to_string(index=False)
    )

    # Source validation

    print("\n=== DATA SOURCES ===")

    print(df["source"].value_counts().to_string())

    # Examples for manual semantic review

    print("\n=== SAMPLE UTTERANCES ===")

    for intent in INTENT_CLASSES:

        subset = df[df["intent"] == intent]

        samples = subset.sample(
            n=min(10, len(subset)),
            random_state=RANDOM_STATE
        )

        print(f"\n--- {intent} ---")

        for _, row in samples.iterrows():

            print(f"{row['id']}: {row['text']}")

    # Final validation checks

    print("\n=== FINAL VALIDATION ===")

    checks = {
        "Expected record count":
            len(df) == EXPECTED_ROWS,

        "Required columns present":
            REQUIRED_COLUMNS.issubset(df.columns),

        "No missing values":
            missing_values == 0,

        "No empty required fields":
            empty_fields.sum() == 0,

        "Unique IDs":
            duplicate_ids == 0,

        "No duplicate rows":
            duplicate_rows == 0,

        "No empty model texts":
            empty_model_texts == 0,

        "No invalid labels":
            len(invalid_labels) == 0,

        "All intent classes present":
            len(missing_labels) == 0,

        "Balanced class distribution":
            distribution.eq(EXPECTED_PER_CLASS).all(),

        "No raw text duplicates":
            raw_duplicates == 0,

        "No normalized text duplicates":
            normalized_duplicates == 0,

        "No model text duplicates":
            model_duplicates == 0,

        "No conflicting labels":
            len(conflicting_texts) == 0,

        "Correct dataset source":
            df["source"].eq(
                "digital_mirror_synthetic_v1"
            ).all(),
    }

    for name, passed in checks.items():

        status = "PASS" if passed else "FAIL"

        print(f"{status:4s} | {name}")

    if not all(checks.values()):

        raise ValueError(
            "Dataset validation failed."
        )

    print("\n=== ALL VALIDATION CHECKS PASSED ===")


if __name__ == "__main__":
    main()
