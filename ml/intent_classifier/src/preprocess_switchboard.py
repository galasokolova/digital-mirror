from pathlib import Path

import pandas as pd

from preprocessing.text_cleaner import clean_text


ML_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ML_ROOT / "data" / "processed"

INPUT_PATH = (
    DATA_DIR / "switchboard_reconstructed.csv"
)

OUTPUT_PATH = (
    DATA_DIR / "switchboard_cleaned.csv"
)


def main():

    print("=== E05: TEXT PREPROCESSING ===")

    df = pd.read_csv(
        INPUT_PATH,
        keep_default_na=False
    )

    original_count = len(df)

    print(f"Input rows: {original_count:,}")

    # Preserve reconstructed raw text
    df["clean_text"] = df["text"].apply(
        clean_text
    )

    # Token counts before and after cleaning
    df["raw_token_count"] = (
        df["text"].str.split().str.len()
    )

    df["clean_token_count"] = (
        df["clean_text"].str.split().str.len()
    )

    changed = df["text"].ne(df["clean_text"])

    empty = df["clean_text"].eq("")

    print("\n=== CLEANING STATISTICS ===")

    print(f"Changed texts: {changed.sum():,}")
    print(f"Unchanged texts: {(~changed).sum():,}")
    print(f"Empty texts: {empty.sum():,}")

    print(
        f"Empty percentage: "
        f"{empty.mean():.2%}"
    )

    print("\n=== RAW TOKEN STATISTICS ===")

    print(
        df["raw_token_count"]
        .describe()
        .to_string()
    )

    print("\n=== CLEAN TOKEN STATISTICS ===")

    print(
        df["clean_token_count"]
        .describe()
        .to_string()
    )

    print("\n=== EMPTY TEXTS BY LABEL ===")

    print(
        df.loc[empty, "damsl_tag"]
        .value_counts()
        .to_string()
    )

    print("\n=== SHORT TEXT DISTRIBUTION ===")

    print(
        df["clean_token_count"]
        .value_counts()
        .sort_index()
        .head(11)
        .to_string()
    )

    print("\n=== CLEANING EXAMPLES ===")

    changed_rows = df[changed]

    samples = changed_rows.sample(
        n=min(30, len(changed_rows)),
        random_state=42
    )

    for _, row in samples.iterrows():

        print("\n" + "=" * 70)

        print(f"Tag: {row['damsl_tag']}")
        print(f"Raw: {row['text']}")
        print(f"Clean: {row['clean_text']}")

    print("\n=== EMPTY TEXT EXAMPLES ===")

    empty_rows = df[empty]

    samples = empty_rows.sample(
        n=min(15, len(empty_rows)),
        random_state=42
    )

    for _, row in samples.iterrows():

        print("\n" + "=" * 70)

        print(f"Tag: {row['damsl_tag']}")
        print(f"Raw: {row['text']}")

    print("\n=== VALIDATION ===")

    assert len(df) == original_count

    assert df["clean_text"].notna().all()

    assert df["damsl_tag"].ne("+").all()

    # Check that cleaning is idempotent
    assert df["clean_text"].apply(
        clean_text
    ).equals(df["clean_text"])

    print("Row accounting: PASSED")
    print("Missing clean texts: 0")
    print("Continuation tags: 0")
    print("Cleaning idempotence: PASSED")

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n=== OUTPUT ===")

    print(f"Saved to: {OUTPUT_PATH}")

    print("\n=== PREPROCESSING COMPLETE ===")


if __name__ == "__main__":
    main()   