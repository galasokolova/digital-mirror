from explore_switchboard import load_dataset
from preprocessing.damsl_normalizer import normalize_damsl_tag


def main():

    print("=== E03: DAMSL TAG NORMALIZATION ===")

    df = load_dataset()

    print("\n=== BEFORE NORMALIZATION ===")

    print(f"Total rows: {len(df):,}")
    print(f"Raw unique tags: {df['act_tag'].nunique()}")

    # Preserve original labels
    df["damsl_tag"] = df["act_tag"].apply(
        normalize_damsl_tag
    )

    print("\n=== AFTER NORMALIZATION ===")

    print(f"Total rows: {len(df):,}")

    print(
        f"Normalized unique tags: "
        f"{df['damsl_tag'].nunique()}"
    )

    print("\n=== NORMALIZED LABEL DISTRIBUTION ===")

    counts = df["damsl_tag"].value_counts()

    print(counts.to_string())

    print("\n=== NORMALIZATION STATISTICS ===")

    changed = (
        df["act_tag"] != df["damsl_tag"]
    ).sum()

    unchanged = len(df) - changed

    print(f"Changed labels: {changed:,}")
    print(f"Unchanged labels: {unchanged:,}")

    print(
        f"Changed percentage: "
        f"{changed / len(df):.2%}"
    )

    print("\n=== RAW TO DAMSL MAPPING ===")

    mapping = (
        df.groupby(["act_tag", "damsl_tag"])
        .size()
        .reset_index(name="count")
        .sort_values(
            ["damsl_tag", "count"],
            ascending=[True, False]
        )
    )

    print(mapping.to_string(index=False))

    print("\n=== EMPTY NORMALIZED TAGS ===")

    empty_tags = df["damsl_tag"].str.strip().eq("")

    print(f"Count: {empty_tags.sum():,}")

    print("\n=== CONTINUATION ROWS ===")

    continuations = df["damsl_tag"].eq("+").sum()

    print(f"Count: {continuations:,}")

    print("\n=== EXPLORATION COMPLETE ===")


if __name__ == "__main__":
    main()