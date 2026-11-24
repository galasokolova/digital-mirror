import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw" / "liar"
OUTPUT_DIR = ROOT / "data" / "processed" / "liar"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

LABEL_MAP = {
    "true": 1,
    "false": 0,
}

SPLITS = ["train", "valid", "test"]

data = {}

for split in SPLITS:
    df = pd.read_csv(
        RAW_DIR / f"{split}.tsv",
        sep="\t",
        header=None,
        names=COLUMNS,
        dtype=str,
        keep_default_na=False,
    )

    df = df[df["label"].isin(LABEL_MAP)].copy()

    df["text"] = (
        df["statement"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    df["text_key"] = df["text"].str.casefold()

    df["target"] = df["label"].map(LABEL_MAP)

    df = df[df["text"] != ""]

    data[split] = df

# Identify contradictory labels before deduplication.
combined = pd.concat(data.values(), ignore_index=True)

conflicts = (
    combined.groupby("text_key")["target"]
    .nunique()
)

conflict_keys = set(
    conflicts[conflicts > 1].index
)

print("Contradictory statements:", len(conflict_keys))

# Exclude contradictions from every split.
for split in SPLITS:
    data[split] = data[split][
        ~data[split]["text_key"].isin(conflict_keys)
    ].copy()

# Preserve evaluation sets preferentially.
seen = set()

for split in ["test", "valid", "train"]:
    df = data[split]

    before = len(df)

    df = df.drop_duplicates(
        subset=["text_key"],
        keep="first"
    )

    df = df[~df["text_key"].isin(seen)].copy()

    seen.update(df["text_key"])

    data[split] = df

    print(
        f"{split}: {before} -> {len(df)} "
        f"(removed {before - len(df)})"
    )

for split in SPLITS:
    df = data[split][
        ["id", "text", "target", "label"]
    ].reset_index(drop=True)

    path = OUTPUT_DIR / f"{split}.csv"
    df.to_csv(path, index=False)

    print(f"\n{split.upper()}")
    print(df["label"].value_counts())
    print("Saved:", path)

print("\nPreparation complete.")
