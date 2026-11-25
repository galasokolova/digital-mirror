from pathlib import Path
from itertools import combinations

import pandas as pd

from sklearn.model_selection import GroupShuffleSplit

from preprocessing.intent_text_normalizer import (
    normalize_text,
    remove_punctuation,
)

from taxonomy.intent_mapping import INTENT_CLASSES


ML_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ML_ROOT / "data" / "processed"

INPUT_PATH = DATA_DIR / "intent_dataset_v1.csv"

OUTPUT_DIR = DATA_DIR / "intent_v1"

RANDOM_STATE = 42


def split_dataset(df):

    # First split: 80% train, 20% temporary
    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    train_indices, temp_indices = next(
        first_split.split(
            df,
            groups=df["conversation_no"]
        )
    )

    train = df.iloc[train_indices].copy()

    temp = df.iloc[temp_indices].copy()

    # Second split: 10% validation, 10% test
    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=RANDOM_STATE
    )

    val_indices, test_indices = next(
        second_split.split(
            temp,
            groups=temp["conversation_no"]
        )
    )

    validation = temp.iloc[val_indices].copy()

    test = temp.iloc[test_indices].copy()

    return {
        "train": train,
        "validation": validation,
        "test": test,
    }


def print_split_statistics(splits, total_rows):

    print("\n=== SPLIT STATISTICS ===")

    for name, df in splits.items():

        print(f"\n{name.upper()}")

        print(f"Rows: {len(df):,}")

        print(
            f"Percentage: "
            f"{len(df) / total_rows:.2%}"
        )

        print(
            f"Conversations: "
            f"{df['conversation_no'].nunique():,}"
        )

        print(
            f"Intent classes: "
            f"{df['intent'].nunique()}"
        )


def analyze_class_distribution(splits):

    print("\n=== CLASS DISTRIBUTION ===")

    distribution = pd.DataFrame({
        name: df["intent"].value_counts()
        for name, df in splits.items()
    })

    distribution = (
        distribution
        .reindex(INTENT_CLASSES)
        .fillna(0)
        .astype(int)
    )

    distribution["total"] = distribution.sum(axis=1)

    print(distribution.to_string())

    print("\n=== RARE CLASS WARNINGS ===")

    for name in splits:

        for intent in INTENT_CLASSES:

            count = distribution.loc[intent, name]

            if count < 10:

                print(
                    f"WARNING: {name} / {intent}: "
                    f"{count} examples"
                )


def analyze_overlap(splits):

    print("\n=== SPLIT OVERLAP ===")

    for first, second in combinations(splits, 2):

        first_df = splits[first]
        second_df = splits[second]

        shared_conversations = (
            set(first_df["conversation_no"])
            & set(second_df["conversation_no"])
        )

        shared_texts = (
            set(first_df["model_text"])
            & set(second_df["model_text"])
        )

        print(f"\n{first} <-> {second}")

        print(
            f"Shared conversations: "
            f"{len(shared_conversations):,}"
        )

        print(
            f"Shared normalized texts: "
            f"{len(shared_texts):,}"
        )

        assert not shared_conversations


def validate_splits(splits, original_df):

    print("\n=== VALIDATION ===")

    total_rows = sum(
        len(df) for df in splits.values()
    )

    assert total_rows == len(original_df)

    print("Row accounting: PASSED")

    combined = pd.concat(splits.values())

    assert combined.index.is_unique

    assert set(combined.index) == set(original_df.index)

    print("Row overlap: 0")

    for name, df in splits.items():

        assert df["model_text"].str.strip().ne("").all()

        missing_classes = (
            set(INTENT_CLASSES)
            - set(df["intent"])
        )

        print(
            f"{name}: missing classes = "
            f"{sorted(missing_classes)}"
        )

        if missing_classes:

            raise ValueError(
                f"{name} is missing intent classes"
            )

    print("Class coverage: PASSED")

    print("Empty model texts: 0")

    print("Conversation isolation: PASSED")


def save_splits(splits):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n=== OUTPUT FILES ===")

    for name, df in splits.items():

        path = OUTPUT_DIR / f"{name}.csv"

        df.to_csv(
            path,
            index=False,
            encoding="utf-8"
        )

        print(f"Saved: {path}")


def main():

    print("=== E08a.3: CONVERSATION-BASED SPLIT ===")

    print(f"Random state: {RANDOM_STATE}")

    df = pd.read_csv(
        INPUT_PATH,
        keep_default_na=False
    )

    print(f"Input rows: {len(df):,}")

    print(
        f"Conversations: "
        f"{df['conversation_no'].nunique():,}"
    )

    # Preserve both text representations
    df["normalized_text"] = (
        df["clean_text"].apply(normalize_text)
    )

    df["model_text"] = (
        df["clean_text"].apply(remove_punctuation)
    )

    splits = split_dataset(df)

    print_split_statistics(splits, len(df))

    analyze_class_distribution(splits)

    analyze_overlap(splits)

    validate_splits(splits, df)

    save_splits(splits)

    print("\n=== SPLIT COMPLETE ===")


if __name__ == "__main__":
    main()