from pathlib import Path
from collections import Counter
from itertools import combinations

import pandas as pd


ML_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    ML_ROOT / "data" / "raw" / "goemotions"
)

SPLITS = ["train", "dev", "test"]

RANDOM_STATE = 42

TARGET_EMOTIONS = [
    "amusement",
    "annoyance",
    "approval",
    "confusion",
    "curiosity",
    "disapproval",
    "gratitude",
    "realization",
    "surprise",
    "neutral",
]


def load_emotions():

    path = DATASET_DIR / "emotions.txt"

    emotions = path.read_text(
        encoding="utf-8"
    ).splitlines()

    return [
        emotion.strip()
        for emotion in emotions
        if emotion.strip()
    ]


def load_dataset():

    frames = []

    for split in SPLITS:

        path = DATASET_DIR / f"{split}.tsv"

        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=["text", "label_ids", "comment_id"],
            dtype=str,
            keep_default_na=False,
            quoting=3
        )

        df["split"] = split

        df["labels"] = df["label_ids"].apply(
            lambda value: [
                int(label)
                for label in value.split(",")
            ]
        )

        frames.append(df)

    return pd.concat(
        frames,
        ignore_index=True
    )


def print_overview(df, emotions):

    print("\n=== DATASET OVERVIEW ===")

    print(f"Total rows: {len(df):,}")
    print(f"Emotion categories: {len(emotions)}")

    print("\n=== SPLIT DISTRIBUTION ===")

    print(
        df["split"]
        .value_counts()
        .to_string()
    )

    print("\n=== DATA QUALITY ===")

    print("Missing values:")

    print(df.isna().sum().to_string())

    empty = df["text"].str.strip().eq("")

    print(f"\nEmpty texts: {empty.sum():,}")

    print(
        f"Duplicate comment IDs: "
        f"{df['comment_id'].duplicated().sum():,}"
    )

    print(
        f"Duplicate texts: "
        f"{df['text'].duplicated().sum():,}"
    )

    invalid_ids = sorted({
        label
        for labels in df["labels"]
        for label in labels
        if not 0 <= label < len(emotions)
    })

    print(f"Invalid label IDs: {invalid_ids}")

    assert not invalid_ids


def analyze_label_distribution(df, emotions):

    print("\n=== EMOTION DISTRIBUTION ===")

    counts = Counter(
        label
        for labels in df["labels"]
        for label in set(labels)
    )

    for label, count in counts.most_common():

        print(
            f"{label:2d} "
            f"{emotions[label]:20s} "
            f"{count:7,} "
            f"{count / len(df):7.2%}"
        )

    print("\n=== EMOTION DISTRIBUTION BY SPLIT ===")

    expanded = df.explode("labels").copy()

    expanded["emotion"] = expanded["labels"].map(
        dict(enumerate(emotions))
    )

    distribution = pd.crosstab(
        expanded["emotion"],
        expanded["split"]
    )

    print(distribution.to_string())


def analyze_multilabel(df, emotions):

    print("\n=== MULTILABEL ANALYSIS ===")

    label_counts = df["labels"].apply(
        lambda labels: len(set(labels))
    )

    single = label_counts.eq(1).sum()
    multiple = label_counts.gt(1).sum()

    print(f"Single-label rows: {single:,}")
    print(f"Multilabel rows: {multiple:,}")

    print(
        f"Multilabel percentage: "
        f"{multiple / len(df):.2%}"
    )

    print(
        f"Average labels per text: "
        f"{label_counts.mean():.4f}"
    )

    print("\n=== LABEL COUNT DISTRIBUTION ===")

    print(
        label_counts.value_counts()
        .sort_index()
        .to_string()
    )

    print("\n=== MOST FREQUENT EMOTION PAIRS ===")

    pairs = Counter()

    for labels in df["labels"]:

        pairs.update(
            combinations(sorted(set(labels)), 2)
        )

    for (first, second), count in pairs.most_common(30):

        print(
            f"{emotions[first]:18s} + "
            f"{emotions[second]:18s} "
            f"{count:,}"
        )

    neutral_id = emotions.index("neutral")

    neutral_with_other = df["labels"].apply(
        lambda labels:
            neutral_id in labels
            and len(set(labels)) > 1
    )

    print(
        f"\nNeutral with other emotions: "
        f"{neutral_with_other.sum():,}"
    )


def analyze_text_length(df):

    print("\n=== TEXT LENGTH STATISTICS ===")

    token_counts = (
        df["text"].str.split().str.len()
    )

    print(token_counts.describe().to_string())

    print("\n=== SHORT TEXT DISTRIBUTION ===")

    print(
        token_counts.value_counts()
        .sort_index()
        .head(11)
        .to_string()
    )

    short = token_counts.between(1, 3)

    print(f"\nTexts with 1-3 tokens: {short.sum():,}")

    print(
        f"Short text percentage: "
        f"{short.mean():.2%}"
    )


def analyze_split_overlap(df):

    print("\n=== SPLIT OVERLAP ===")

    for first, second in combinations(SPLITS, 2):

        first_df = df[df["split"] == first]
        second_df = df[df["split"] == second]

        shared_ids = (
            set(first_df["comment_id"])
            & set(second_df["comment_id"])
        )

        shared_texts = (
            set(first_df["text"])
            & set(second_df["text"])
        )

        print(f"\n{first} <-> {second}")
        print(f"Shared comment IDs: {len(shared_ids):,}")
        print(f"Shared exact texts: {len(shared_texts):,}")


def explore_emotion_examples(df, emotions):

    print("\n=== TARGET EMOTION EXAMPLES ===")

    # Qualitative exploration of training data only
    train_df = df[df["split"] == "train"]

    for emotion in TARGET_EMOTIONS:

        label_id = emotions.index(emotion)

        subset = train_df[
            train_df["labels"].apply(
                lambda labels: label_id in labels
            )
        ]

        print("\n" + "=" * 70)
        print(f"EMOTION: {emotion}")
        print(f"TRAIN ROWS: {len(subset):,}")
        print("=" * 70)

        samples = subset.sample(
            n=min(15, len(subset)),
            random_state=RANDOM_STATE
        )

        for index, (_, row) in enumerate(
            samples.iterrows(),
            start=1
        ):

            names = [
                emotions[label]
                for label in row["labels"]
            ]

            print(f"{index:02d}. {row['text']}")
            print(f"    Labels: {', '.join(names)}")


def main():

    print("=== E07: GOEMOTIONS EXPLORATION ===")

    emotions = load_emotions()

    df = load_dataset()

    print_overview(df, emotions)

    analyze_label_distribution(df, emotions)

    analyze_multilabel(df, emotions)

    analyze_text_length(df)

    analyze_split_overlap(df)

    explore_emotion_examples(df, emotions)

    print("\n=== EXPLORATION COMPLETE ===")


if __name__ == "__main__":
    main()