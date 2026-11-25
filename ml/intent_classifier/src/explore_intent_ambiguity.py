from pathlib import Path
import unicodedata

import pandas as pd


ML_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ML_ROOT
    / "data"
    / "processed"
    / "intent_dataset_v1.csv"
)


def normalize_text(text: str) -> str:

    return " ".join(text.lower().split())


def remove_punctuation(text: str) -> str:

    text = normalize_text(text)

    # Normalize apostrophe variants
    text = text.replace("’", "'")

    # Preserve apostrophes inside contractions
    text = "".join(
        char
        if char == "'"
        or not unicodedata.category(char).startswith("P")
        else " "
        for char in text
    )

    return " ".join(text.split())


def analyze_ambiguity(df, text_column, title):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    empty_mask = df[text_column].str.strip().eq("")

    print(
        f"Empty texts after normalization: "
        f"{empty_mask.sum():,}"
    )

    df = df.loc[~empty_mask].copy()

    label_counts = (
        df.groupby([text_column, "intent"])
        .size()
        .unstack(fill_value=0)
    )

    text_counts = label_counts.sum(axis=1)

    labels_per_text = label_counts.gt(0).sum(axis=1)

    ambiguous_mask = labels_per_text.gt(1)

    repeated_mask = text_counts.gt(1)

    total_rows = len(df)

    unique_texts = len(label_counts)

    ambiguous_texts = ambiguous_mask.sum()

    ambiguous_rows = (
        text_counts[ambiguous_mask].sum()
    )

    best_possible_correct = (
        label_counts.max(axis=1).sum()
    )

    ceiling = best_possible_correct / total_rows

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

    print(
        f"Empirical text-only ceiling: "
        f"{ceiling:.4f}"
    )

    print("\n=== MOST FREQUENT AMBIGUOUS TEXTS ===")

    top_ambiguous = (
        text_counts[ambiguous_mask]
        .sort_values(ascending=False)
        .head(20)
    )

    for text, count in top_ambiguous.items():

        distribution = label_counts.loc[text]

        distribution = (
            distribution[distribution.gt(0)]
            .sort_values(ascending=False)
        )

        print("\n" + "-" * 60)

        print(f"Text: {text}")
        print(f"Total occurrences: {count:,}")
        print(f"Different intents: {len(distribution)}")

        for intent, frequency in distribution.items():

            print(
                f"  {intent:25s} "
                f"{frequency:7,} "
                f"({frequency / count:.2%})"
            )

    print("\n=== AMBIGUITY BY INTENT ===")

    for intent in label_counts.columns:

        total = label_counts[intent].sum()

        ambiguous = (
            label_counts.loc[ambiguous_mask, intent]
            .sum()
        )

        print(
            f"{intent:25s} "
            f"total={total:7,} "
            f"ambiguous={ambiguous:7,} "
            f"rate={ambiguous / total:7.2%}"
        )

    print("\n=== INTENT OVERLAP ===")

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

    for first, second, shared in pairs[:20]:

        print(
            f"{first:25s} <-> "
            f"{second:25s} "
            f"{shared:,} shared texts"
        )

    return {
        "experiment": title,
        "rows": total_rows,
        "unique_texts": unique_texts,
        "ambiguous_texts": ambiguous_texts,
        "ambiguous_rows_pct": (
            ambiguous_rows / total_rows * 100
        ),
        "ceiling_pct": ceiling * 100,
    }


def main():

    print("=== E08a.2: CUSTOM INTENT AMBIGUITY ===")

    df = pd.read_csv(
        INPUT_PATH,
        keep_default_na=False
    )

    print(f"Input rows: {len(df):,}")

    print(
        f"Intent classes: "
        f"{df['intent'].nunique()}"
    )

    df["normalized_text"] = (
        df["clean_text"].apply(normalize_text)
    )

    df["no_punctuation_text"] = (
        df["clean_text"].apply(remove_punctuation)
    )

    results = []

    results.append(
        analyze_ambiguity(
            df,
            "clean_text",
            "A: ORIGINAL CLEAN TEXT"
        )
    )

    results.append(
        analyze_ambiguity(
            df,
            "normalized_text",
            "B: LOWERCASE TEXT"
        )
    )

    results.append(
        analyze_ambiguity(
            df,
            "no_punctuation_text",
            "C: WITHOUT PUNCTUATION"
        )
    )

    print("\n=== EXPERIMENT COMPARISON ===")

    summary = pd.DataFrame(results)

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}"
        )
    )

    print("\n=== ANALYSIS COMPLETE ===")


if __name__ == "__main__":
    main()