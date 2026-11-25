from pathlib import Path

import pandas as pd

from taxonomy.intent_mapping import (
    INTENT_CLASSES,
    TAXONOMY_VERSION,
    map_intent,
)


ML_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ML_ROOT / "data" / "processed"

INPUT_PATH = DATA_DIR / "switchboard_cleaned.csv"

OUTPUT_PATH = DATA_DIR / "intent_dataset_v1.csv"


def main():

    print("=== E08a: INTENT DATASET PREPARATION ===")

    print(f"Taxonomy version: {TAXONOMY_VERSION}")

    df = pd.read_csv(
        INPUT_PATH,
        keep_default_na=False
    )

    original_count = len(df)

    print(f"Input rows: {original_count:,}")

    # Exclude empty texts
    empty_mask = df["clean_text"].str.strip().eq("")

    empty_count = empty_mask.sum()

    df = df.loc[~empty_mask].copy()

    print(f"Empty texts excluded: {empty_count:,}")

    # Map DAMSL labels to custom intents
    df["intent"] = df["damsl_tag"].apply(map_intent)

    excluded = df[df["intent"].isna()].copy()

    dataset = df[df["intent"].notna()].copy()

    print("\n=== DATASET STATISTICS ===")

    print(f"Included rows: {len(dataset):,}")

    print(f"Excluded unmapped rows: {len(excluded):,}")

    print(
        f"Retained percentage: "
        f"{len(dataset) / original_count:.2%}"
    )

    print(
        f"Intent classes: "
        f"{dataset['intent'].nunique()}"
    )

    print(
        f"Conversations: "
        f"{dataset['conversation_no'].nunique():,}"
    )

    print("\n=== INTENT DISTRIBUTION ===")

    counts = dataset["intent"].value_counts()

    for intent, count in counts.items():

        print(
            f"{intent:25s} "
            f"{count:8,} "
            f"{count / len(dataset):7.2%}"
        )

    print("\n=== CLASS IMBALANCE ===")

    print(f"Largest class: {counts.idxmax()}")

    print(f"Largest class size: {counts.max():,}")

    print(f"Smallest class: {counts.idxmin()}")

    print(f"Smallest class size: {counts.min():,}")

    print(
        f"Imbalance ratio: "
        f"{counts.max() / counts.min():.2f}"
    )

    print("\n=== EXCLUDED DAMSL LABELS ===")

    print(
        excluded["damsl_tag"]
        .value_counts()
        .to_string()
    )

    print("\n=== INTENT MAPPING DISTRIBUTION ===")

    mapping_distribution = pd.crosstab(
        dataset["intent"],
        dataset["damsl_tag"]
    )

    print(mapping_distribution.to_string())

    print("\n=== VALIDATION ===")

    assert (
        len(dataset)
        + len(excluded)
        + empty_count
        == original_count
    )

    assert dataset["intent"].notna().all()

    assert dataset["clean_text"].str.strip().ne("").all()

    assert set(dataset["intent"]) == set(INTENT_CLASSES)

    print("Row accounting: PASSED")
    print("Missing intents: 0")
    print("Empty texts: 0")
    print("Intent class validation: PASSED")

    # Preserve source information for reproducibility
    dataset.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    print("\n=== OUTPUT ===")

    print(f"Saved to: {OUTPUT_PATH}")

    print("\n=== PREPARATION COMPLETE ===")


if __name__ == "__main__":
    main()