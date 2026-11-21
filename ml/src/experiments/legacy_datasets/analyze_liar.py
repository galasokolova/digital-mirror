import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "raw" / "liar"

# LIAR has no header row in its TSV files.
COLUMNS = [
    "id",
    "label",
    "statement",
    "subject",
    "speaker",
    "speaker_job",
    "state",
    "party",
    "barely_true_count",
    "false_count",
    "half_true_count",
    "mostly_true_count",
    "pants_fire_count",
    "context",
]

SPLITS = ["train", "valid", "test"]

STRICT_LABELS = {
    "true": "TRUE",
    "false": "FALSE",
}

all_splits = {}

for split in SPLITS:
    path = DATA_DIR / f"{split}.tsv"

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=COLUMNS,
        dtype=str,
        keep_default_na=False,
    )

    all_splits[split] = df

    print(f"\n=== {split.upper()} ===")
    print("Total records:", len(df))

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    binary = df[df["label"].isin(STRICT_LABELS)]

    print("\nStrict binary subset:")
    print(binary["label"].value_counts())
    print("Total binary:", len(binary))

    print("\nMissing/empty statements:",
          df["statement"].str.strip().eq("").sum())

    print("\nDuplicate statements:",
          df["statement"].duplicated().sum())

    print("\nExamples:")
    for label in ["true", "false"]:
        examples = df.loc[
            df["label"] == label, "statement"
        ].head(3)

        print(f"\n{label.upper()}:")
        for statement in examples:
            print("-", statement[:250])


print("\n=== CROSS-SPLIT DUPLICATES ===")

for i, split_a in enumerate(SPLITS):
    for split_b in SPLITS[i + 1:]:
        a = set(
            all_splits[split_a]["statement"]
            .str.strip()
            .str.lower()
        )
        b = set(
            all_splits[split_b]["statement"]
            .str.strip()
            .str.lower()
        )

        print(
            f"{split_a} vs {split_b}:",
            len(a & b)
        )

print("\nAnalysis complete.")
