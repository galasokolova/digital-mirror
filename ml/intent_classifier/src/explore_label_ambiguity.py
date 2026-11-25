from pathlib import Path

import pandas as pd


ML_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ML_ROOT
    / "data"
    / "processed"
    / "switchboard_cleaned.csv"
)

TARGET_TAGS = [
    "aa",
    "aap_am",
    "ar",
    "arp_nd",
    "b",
    "ba",
    "bh",
    "br",
    "h",
    "ny",
    "nn",
    "na",
    "ng",
    "qy",
    "qw",
    "qy^d",
    "qh",
    "fp",
    "fc",
    "ft",
    "sd",
    "sv",
]


def analyze_ambiguity(df, title):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    label_counts = (
        df.groupby(["comparison_text", "damsl_tag"])
        .size()
        .unstack(fill_value=0)
    )

    text_counts = label_counts.sum(axis=1)

    labels_per_text = (
        label_counts.gt(0).sum(axis=1)
    )

    repeated_mask = text_counts.gt(1)

    ambiguous_mask = labels_per_text.gt(1)

    total_rows = len(df)

    unique_texts = len(label_counts)

    ambiguous_texts = ambiguous_mask.sum()

    ambiguous_rows = (
        text_counts[ambiguous_mask].sum()
    )

    print(f"Total rows: {total_rows:,}")

    print(f"Unique texts: {unique_texts:,}")

    print(
        f"Repeated unique texts: "
        f"{repeated_mask.sum():,}"
    )

    print(
        f"Ambiguous unique texts: "
        f"{ambiguous_texts:,}"
    )

    print(
        f"Ambiguous text percentage: "
        f"{ambiguous_texts / unique_texts:.2%}"
    )

    print(f"Ambiguous rows: {ambiguous_rows:,}")

    print(
        f"Ambiguous row percentage: "
        f"{ambiguous_rows / total_rows:.2%}"
    )

    # Empirical maximum for a deterministic
    # classifier using this text representation
    best_possible_correct = (
        label_counts.max(axis=1).sum()
    )

    ceiling = (
        best_possible_correct / total_rows
    )

    print(
        f"Empirical text-only ceiling: "
        f"{ceiling:.4f}"
    )

    print("\n=== MOST FREQUENT AMBIGUOUS TEXTS ===")

    top_ambiguous = (
        text_counts[ambiguous_mask]
        .sort_values(ascending=False)
        .head(30)
    )

    for text, count in top_ambiguous.items():

        distribution = label_counts.loc[text]

        distribution = (
            distribution[distribution.gt(0)]
            .sort_values(ascending=False)
        )

        print("\n" + "-" * 60)

        print(f"Text: {text}")
        print(f"Total occurrences: {count}")
        print(f"Different labels: {len(distribution)}")

        for tag, frequency in distribution.items():

            print(
                f"  {tag}: {frequency:,} "
                f"({frequency / count:.2%})"
            )

    print("\n=== AMBIGUITY BY LABEL ===")

    for tag in label_counts.columns:

        total = label_counts[tag].sum()

        ambiguous = (
            label_counts.loc[ambiguous_mask, tag]
            .sum()
        )

        print(
            f"{tag:20s} "
            f"total={total:7,} "
            f"ambiguous={ambiguous:7,} "
            f"rate={ambiguous / total:7.2%}"
        )

    print("\n=== LABEL OVERLAP ===")

    presence = label_counts.gt(0).astype("int64")

    overlap = presence.T @ presence

    pairs = []

    for i, first in enumerate(overlap.columns):

        for second in overlap.columns[i + 1:]:

            shared = overlap.loc[first, second]

            if shared > 0:

                pairs.append(
                    (first, second, shared)
                )

    pairs.sort(
        key=lambda item: item[2],
        reverse=True
    )

    for first, second, shared in pairs[:30]:

        print(
            f"{first:12s} <-> {second:12s} "
            f"{shared:,} shared texts"
        )


def main():

    print("=== E06: LABEL AMBIGUITY ANALYSIS ===")

    df = pd.read_csv(
        INPUT_PATH,
        keep_default_na=False
    )

    print(f"Input rows: {len(df):,}")

    # Exclude empty texts only for this experiment
    df = df[df["clean_text"].ne("")].copy()

    print(f"Non-empty rows: {len(df):,}")

    # Normalize case and whitespace for comparison.
    # Preserve punctuation and negation.
    df["comparison_text"] = (
        df["clean_text"]
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    analyze_ambiguity(
        df,
        "EXPERIMENT A: ALL DAMSL LABELS"
    )

    target_df = df[
        df["damsl_tag"].isin(TARGET_TAGS)
    ].copy()

    analyze_ambiguity(
        target_df,
        "EXPERIMENT B: SELECTED DAMSL LABELS"
    )

    print("\n=== ANALYSIS COMPLETE ===")


if __name__ == "__main__":
    main()