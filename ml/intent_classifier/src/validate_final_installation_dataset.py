
from pathlib import Path

import pandas as pd

from taxonomy.intent_mapping import INTENT_CLASSES

from preprocessing.intent_text_normalizer import (
    normalize_text,
    remove_punctuation,
)


ML_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    ML_ROOT
    / "data"
    / "evaluation"
    / "digital_mirror_final"
    / "test.csv"
)

EXPECTED_COLUMNS = [
    "id",
    "text",
    "intent",
    "group",
    "source",
]

EXPECTED_GROUPS = [
    "natural",
    "short",
    "installation",
    "complex",
]

EXPECTED_SOURCE = "digital_mirror_synthetic_test_v1"

EXPECTED_RECORDS = 240
EXPECTED_PER_CLASS = 20
EXPECTED_PER_CLASS_GROUP = 5


def load_dataset():

    dataset = pd.read_csv(
        DATASET_PATH,
        dtype=str,
        keep_default_na=False
    )

    return dataset


def analyze_structure(dataset):

    print("\n=== DATASET STRUCTURE ===")

    print(f"Records: {len(dataset)}")
    print(f"Columns: {len(dataset.columns)}")

    print("\nColumn names:")
    print(dataset.columns.tolist())

    print("\nMissing values:")
    print(dataset.isna().sum().to_string())

    print("\nEmpty values:")

    empty_values = dataset.apply(
        lambda column: column.str.strip().eq("").sum()
    )

    print(empty_values.to_string())

    print("\nSource distribution:")

    print(
        dataset["source"]
        .value_counts()
        .to_string()
    )


def prepare_temporary_texts(dataset):

    dataset = dataset.copy()

    dataset["normalized_text"] = (
        dataset["text"]
        .apply(normalize_text)
    )

    dataset["model_text"] = (
        dataset["text"]
        .apply(remove_punctuation)
    )

    return dataset


def analyze_duplicates(dataset):

    print("\n=== DUPLICATE ANALYSIS ===")

    columns = [
        "id",
        "text",
        "normalized_text",
        "model_text",
    ]

    for column in columns:

        duplicates = dataset[column].duplicated().sum()

        print(f"{column}: {duplicates}")

        if duplicates > 0:

            mask = dataset[column].duplicated(
                keep=False
            )

            print(
                dataset.loc[
                    mask,
                    ["id", "text", "intent", "group"]
                ].to_string(index=False)
            )

    print(
        "\nDuplicate complete rows:",
        dataset[EXPECTED_COLUMNS].duplicated().sum()
    )

    # Identical model texts with different labels

    label_counts = (
        dataset.groupby("model_text")["intent"]
        .nunique()
    )

    conflicting_texts = label_counts[
        label_counts > 1
    ].index

    print(
        f"\nConflicting model texts: "
        f"{len(conflicting_texts)}"
    )

    if len(conflicting_texts) > 0:

        print(
            dataset.loc[
                dataset["model_text"].isin(
                    conflicting_texts
                ),
                [
                    "id",
                    "text",
                    "model_text",
                    "intent",
                    "group",
                ]
            ].to_string(index=False)
        )

    return len(conflicting_texts)


def analyze_distribution(dataset):

    print("\n=== INTENT DISTRIBUTION ===")

    distribution = (
        dataset["intent"]
        .value_counts()
        .reindex(INTENT_CLASSES, fill_value=0)
    )

    distribution_df = pd.DataFrame({
        "records": distribution,
        "percentage": (
            distribution / len(dataset) * 100
        ).round(2),
    })

    print(distribution_df.to_string())

    print("\n=== GROUP DISTRIBUTION ===")

    print(
        dataset["group"]
        .value_counts()
        .reindex(EXPECTED_GROUPS, fill_value=0)
        .to_string()
    )

    print("\n=== INTENT / GROUP DISTRIBUTION ===")

    cross_table = pd.crosstab(
        dataset["intent"],
        dataset["group"]
    ).reindex(
        index=INTENT_CLASSES,
        columns=EXPECTED_GROUPS,
        fill_value=0
    )

    cross_table["total"] = (
        cross_table.sum(axis=1)
    )

    print(cross_table.to_string())

    return distribution, cross_table


def analyze_text_statistics(dataset):

    print("\n=== TEXT STATISTICS ===")

    dataset["word_count"] = (
        dataset["model_text"]
        .str.split()
        .str.len()
    )

    print(
        dataset["word_count"]
        .describe()
        .round(2)
        .to_string()
    )

    short_mask = (
        dataset["word_count"].between(1, 3)
    )

    short_count = short_mask.sum()

    print(
        f"\nShort utterances (1-3 words): "
        f"{short_count}"
    )

    print(
        f"Short utterance percentage: "
        f"{short_count / len(dataset) * 100:.2f}%"
    )

    vocabulary = set(
        " ".join(dataset["model_text"]).split()
    )

    print(f"Unique words: {len(vocabulary)}")

    print("\n=== TEXT LENGTH BY INTENT ===")

    print(
        dataset.groupby("intent")["word_count"]
        .agg(["count", "min", "mean", "max"])
        .reindex(INTENT_CLASSES)
        .round(2)
        .to_string()
    )

    print("\n=== TEXT LENGTH BY GROUP ===")

    print(
        dataset.groupby("group")["word_count"]
        .agg(["count", "min", "mean", "max"])
        .reindex(EXPECTED_GROUPS)
        .round(2)
        .to_string()
    )


def print_examples(dataset):

    print("\n=== SAMPLE UTTERANCES ===")

    for intent in INTENT_CLASSES:

        print(f"\n--- {intent} ---")

        subset = dataset.loc[
            dataset["intent"] == intent
        ]

        for group in EXPECTED_GROUPS:

            group_subset = subset.loc[
                subset["group"] == group
            ]

            if group_subset.empty:
                continue

            example = group_subset.sample(
                n=1,
                random_state=42
            ).iloc[0]

            print(f"[{group}] {example['text']}")


def validate_dataset(
    dataset,
    distribution,
    cross_table,
    conflicting_count
):

    print("\n=== FINAL VALIDATION CHECKS ===")

    empty_values = dataset[
        EXPECTED_COLUMNS
    ].apply(
        lambda column: column.str.strip().eq("")
    )

    checks = {

        "Expected record count":
            len(dataset) == EXPECTED_RECORDS,

        "Expected original columns":
            dataset.columns[:5].tolist()
            == EXPECTED_COLUMNS,

        "No missing values":
            not dataset[EXPECTED_COLUMNS]
            .isna().any().any(),

        "No empty values":
            not empty_values.any().any(),

        "Unique IDs":
            dataset["id"].is_unique,

        "No duplicate rows":
            not dataset[EXPECTED_COLUMNS]
            .duplicated().any(),

        "Unique raw texts":
            dataset["text"].is_unique,

        "Unique normalized texts":
            dataset["normalized_text"].is_unique,

        "Unique model texts":
            dataset["model_text"].is_unique,

        "No conflicting labels":
            conflicting_count == 0,

        "Valid intent labels":
            dataset["intent"]
            .isin(INTENT_CLASSES).all(),

        "Valid group labels":
            dataset["group"]
            .isin(EXPECTED_GROUPS).all(),

        "All 12 intent classes present":
            set(dataset["intent"])
            == set(INTENT_CLASSES),

        "Balanced intent distribution":
            distribution.eq(
                EXPECTED_PER_CLASS
            ).all(),

        "Balanced intent-group distribution":
            cross_table[EXPECTED_GROUPS]
            .eq(EXPECTED_PER_CLASS_GROUP)
            .all().all(),

        "No empty model texts":
            not dataset["model_text"]
            .eq("").any(),

        "Expected source":
            dataset["source"]
            .eq(EXPECTED_SOURCE).all(),
    }

    for name, passed in checks.items():

        status = "PASS" if passed else "FAIL"

        print(f"{status} | {name}")

    passed_count = sum(checks.values())

    print(
        f"\nChecks passed: "
        f"{passed_count} / {len(checks)}"
    )

    if not all(checks.values()):

        raise ValueError(
            "Final test dataset validation failed."
        )

    print("\nDataset validation: PASSED")


def main():

    print(
        "=== E09.06.2b: FINAL INSTALLATION "
        "DATASET VALIDATION ==="
    )

    print(f"\nDataset path: {DATASET_PATH}")

    dataset = load_dataset()

    if dataset.columns.tolist() != EXPECTED_COLUMNS:

        raise ValueError(
            "Unexpected dataset columns."
        )

    if dataset.empty:

        raise ValueError(
            "Dataset is empty."
        )

    analyze_structure(dataset)

    dataset = prepare_temporary_texts(dataset)

    conflicting_count = analyze_duplicates(
        dataset
    )

    distribution, cross_table = (
        analyze_distribution(dataset)
    )

    analyze_text_statistics(dataset)

    print_examples(dataset)

    validate_dataset(
        dataset,
        distribution,
        cross_table,
        conflicting_count
    )

    print("\nOriginal dataset modified: NO")
    print("Model loaded: NO")
    print("Model training performed: NO")

    print("\n=== DATASET VALIDATION COMPLETE ===")


if __name__ == "__main__":
    main()
