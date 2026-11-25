from pathlib import Path
from itertools import combinations
import json

import pandas as pd

from taxonomy.emotion_mapping import (
    EMOTION_CLASSES,
    TAXONOMY_VERSION,
    map_emotions,
)

from preprocessing.intent_text_normalizer import (
    normalize_text,
    remove_punctuation,
)


ML_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    ML_ROOT / "data" / "raw" / "goemotions"
)

OUTPUT_DIR = (
    ML_ROOT / "data" / "processed" / "emotion_v1"
)

SPLIT_FILES = {
    "train": "train.tsv",
    "validation": "dev.tsv",
    "test": "test.tsv",
}


def load_emotion_names():

    path = INPUT_DIR / "emotions.txt"

    return [
        line.strip()
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def load_split(filename, emotion_names):

    df = pd.read_csv(
        INPUT_DIR / filename,
        sep="\t",
        header=None,
        names=["text", "label_ids", "comment_id"],
        dtype=str,
        keep_default_na=False,
        quoting=3,
    )

    def decode_labels(value):

        ids = [
            int(item)
            for item in value.split(",")
        ]

        assert all(
            0 <= item < len(emotion_names)
            for item in ids
        )

        return [
            emotion_names[item]
            for item in ids
        ]

    df["original_emotions"] = (
        df["label_ids"].apply(decode_labels)
    )

    return df


def prepare_split(df):

    df = df.copy()

    df["emotions"] = (
        df["original_emotions"].apply(map_emotions)
    )

    df["normalized_text"] = (
        df["text"].apply(normalize_text)
    )

    df["model_text"] = (
        df["text"].apply(remove_punctuation)
    )

    df["emotion_count"] = (
        df["emotions"].apply(len)
    )

    return df


def print_statistics(name, df):

    print("\n" + "=" * 60)
    print(name.upper())
    print("=" * 60)

    print(f"Rows: {len(df):,}")

    no_target = df["emotion_count"].eq(0).sum()

    single = df["emotion_count"].eq(1).sum()

    multiple = df["emotion_count"].gt(1).sum()

    print(f"No target emotions: {no_target:,}")
    print(f"Single target emotion: {single:,}")
    print(f"Multiple target emotions: {multiple:,}")

    print(
        f"Multilabel percentage: "
        f"{multiple / len(df):.2%}"
    )

    print("\nEmotion distribution:")

    counts = (
        df["emotions"]
        .explode()
        .value_counts()
        .reindex(EMOTION_CLASSES, fill_value=0)
    )

    print(counts.to_string())

    print("\nTarget label count distribution:")

    print(
        df["emotion_count"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    neutral_with_others = df["emotions"].apply(
        lambda labels:
            "NEUTRAL" in labels
            and len(labels) > 1
    ).sum()

    print(
        f"\nNeutral with other target emotions: "
        f"{neutral_with_others:,}"
    )


def analyze_overlap(splits):

    print("\n=== SPLIT OVERLAP ===")

    for first, second in combinations(splits, 2):

        first_df = splits[first]
        second_df = splits[second]

        shared_ids = (
            set(first_df["comment_id"])
            & set(second_df["comment_id"])
        )

        shared_raw_texts = (
            set(first_df["text"])
            & set(second_df["text"])
        )

        shared_model_texts = (
            set(first_df["model_text"])
            & set(second_df["model_text"])
        )

        print(f"\n{first} <-> {second}")

        print(f"Shared IDs: {len(shared_ids):,}")

        print(
            f"Shared raw texts: "
            f"{len(shared_raw_texts):,}"
        )

        print(
            f"Shared model texts: "
            f"{len(shared_model_texts):,}"
        )

        assert not shared_ids


def validate_splits(splits):

    print("\n=== VALIDATION ===")

    allowed = set(EMOTION_CLASSES)

    for name, df in splits.items():

        assert df["comment_id"].is_unique

        assert df["comment_id"].str.strip().ne("").all()

        assert df["text"].str.strip().ne("").all()

        assert df["emotions"].apply(
            lambda labels: set(labels).issubset(allowed)
        ).all()

        present = set(
            df["emotions"].explode().dropna()
        )

        missing = allowed - present

        assert not missing, (
            f"{name}: missing emotions {missing}"
        )

        empty_model_texts = (
            df["model_text"].str.strip().eq("").sum()
        )

        print(f"{name}: validation PASSED")

        print(
            f"{name}: empty model texts = "
            f"{empty_model_texts:,}"
        )

    print("Comment ID isolation: PASSED")
    print("Emotion class coverage: PASSED")


def save_splits(splits):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n=== OUTPUT FILES ===")

    for name, df in splits.items():

        output = df.copy()

        # Store lists as valid JSON inside CSV fields
        for column in ["original_emotions", "emotions"]:

            output[column] = (
                output[column].apply(json.dumps)
            )

        path = OUTPUT_DIR / f"{name}.csv"

        output.to_csv(
            path,
            index=False,
            encoding="utf-8"
        )

        print(f"Saved: {path}")


def main():

    print("=== E08b: EMOTION DATASET PREPARATION ===")

    print(f"Taxonomy version: {TAXONOMY_VERSION}")

    emotion_names = load_emotion_names()

    print(
        f"Original emotion categories: "
        f"{len(emotion_names)}"
    )

    print(
        f"Target emotion categories: "
        f"{len(EMOTION_CLASSES)}"
    )

    splits = {}

    for name, filename in SPLIT_FILES.items():

        raw = load_split(filename, emotion_names)

        prepared = prepare_split(raw)

        assert len(raw) == len(prepared)

        splits[name] = prepared

        print_statistics(name, prepared)

    print(
        f"\nTotal rows: "
        f"{sum(len(df) for df in splits.values()):,}"
    )

    analyze_overlap(splits)

    validate_splits(splits)

    save_splits(splits)

    print("\n=== PREPARATION COMPLETE ===")


if __name__ == "__main__":
    main()