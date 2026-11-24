from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / "data/processed/claims_review.csv"

df = pd.read_csv(INPUT)

df["checkable"] = (
    df["checkable"]
    .astype("string")
    .str.strip()
    .str.upper()
)

print("=== CLAIM REVIEW SUMMARY ===")
print("Total claims:", len(df))

print("\nCheckability:")
print(df["checkable"].value_counts(dropna=False))

print("\nMissing labels:")
print(df["checkable"].isna().sum())

print("\nUnexpected labels:")
valid = {"YES", "NO", "UNCERTAIN"}
unexpected = df[
    df["checkable"].notna()
    & ~df["checkable"].isin(valid)
]
print(unexpected[["claim", "checkable"]].to_string(index=False))

print("\nCheckable percentage:")
print(f"{df['checkable'].eq('YES').sum() / len(df):.1%}")
