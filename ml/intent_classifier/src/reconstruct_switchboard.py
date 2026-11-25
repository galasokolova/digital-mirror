from pathlib import Path

from explore_switchboard import load_dataset

from preprocessing.damsl_normalizer import (
    normalize_damsl_tag
)

from preprocessing.utterance_reconstructor import (
    reconstruct_utterances
)


ML_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ML_ROOT / "data" / "processed"


def main():

    print("=== E04: UTTERANCE RECONSTRUCTION ===")

    df = load_dataset()

    df["damsl_tag"] = df["act_tag"].apply(
        normalize_damsl_tag
    )

    original_count = len(df)

    continuation_count = df["damsl_tag"].eq("+").sum()

    reconstructed, orphans, merged_count = (
        reconstruct_utterances(df)
    )

    print("\n=== RECONSTRUCTION STATISTICS ===")

    print(f"Original rows: {original_count:,}")

    print(f"Continuation rows: {continuation_count:,}")

    print(f"Merged continuations: {merged_count:,}")

    print(f"Orphan continuations: {len(orphans):,}")

    print(f"Reconstructed rows: {len(reconstructed):,}")

    print("\n=== VALIDATION ===")

    accounted_rows = (
        reconstructed["segment_count"].sum()
        + len(orphans)
    )

    remaining_continuations = (
        reconstructed["damsl_tag"].eq("+").sum()
    )

    assert accounted_rows == original_count

    assert (
        merged_count + len(orphans)
        == continuation_count
    )

    assert remaining_continuations == 0

    assert (
        len(reconstructed)
        == original_count - continuation_count
    )

    print("Row accounting: PASSED")
    print("Continuation accounting: PASSED")
    print("Remaining continuation tags: 0")
    print("Output row count: PASSED")

    print("\n=== SEGMENT COUNT DISTRIBUTION ===")

    print(
        reconstructed["segment_count"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\n=== LABEL DISTRIBUTION ===")

    print(
        reconstructed["damsl_tag"]
        .value_counts()
        .to_string()
    )

    print("\n=== RECONSTRUCTED EXAMPLES ===")

    merged = reconstructed[
        reconstructed["segment_count"] > 1
    ]

    samples = merged.sample(
        n=min(15, len(merged)),
        random_state=42
    )

    for _, row in samples.iterrows():

        print("\n" + "=" * 70)

        print(f"Conversation: {row['conversation_no']}")
        print(f"Caller: {row['caller']}")
        print(f"Tag: {row['damsl_tag']}")
        print(f"Segments: {row['segment_count']}")
        print(f"Source indices: {row['source_indices']}")
        print(f"Source tags: {row['source_tags']}")
        print(f"Text: {row['text']}")

    print("\n=== LONGEST RECONSTRUCTIONS ===")

    longest = reconstructed.nlargest(
        5, "segment_count"
    )

    for _, row in longest.iterrows():

        print("\n" + "=" * 70)

        print(f"Conversation: {row['conversation_no']}")
        print(f"Caller: {row['caller']}")
        print(f"Segments: {row['segment_count']}")
        print(f"Tag: {row['damsl_tag']}")
        print(f"Text: {row['text']}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        OUTPUT_DIR / "switchboard_reconstructed.csv"
    )

    reconstructed.to_csv(
        output_path,
        index=False
    )

    if not orphans.empty:

        orphans.to_csv(
            OUTPUT_DIR / "orphan_continuations.csv",
            index=False
        )

    print("\n=== OUTPUT ===")

    print(f"Saved to: {output_path}")

    print("\n=== RECONSTRUCTION COMPLETE ===")


if __name__ == "__main__":
    main()