from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data" / "processed" / "fever"

for split in ["train", "dev"]:
    df = pd.read_csv(DATA_DIR / f"{split}_verified.csv")

    print(f"\n=== {split.upper()} ===")
    print("Evidence rows:", len(df))
    print("Unique claims:", df["claim_id"].nunique())

    print("\nRows by label:")
    print(df["label"].value_counts())

    one_per_claim = df.drop_duplicates(subset=["claim_id"])

    print("\nUnique claims by label:")
    print(one_per_claim["label"].value_counts())

    print("\nEvidence sets per claim:")
    print(
        df.groupby("claim_id")
        .size()
        .describe()
        .round(2)
    )

    print("\nMissing values:")
    print(df.isna().sum())
