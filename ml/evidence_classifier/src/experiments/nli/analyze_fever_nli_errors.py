from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
FILE = (
    BASE_DIR / "data" / "experiments"
    / "fever" / "e03_nli_sample.csv"
)

df = pd.read_csv(FILE)

errors = df[df["label"] != df["prediction"]].copy()

print("=== E03 ERROR ANALYSIS ===")
print("Total claims:", len(df))
print("Total errors:", len(errors))

print("\nError types:")
print(
    errors.groupby(["label", "prediction"])
    .size()
    .sort_values(ascending=False)
)

print("\nAverage confidence:")
print(
    df.groupby("correct")["confidence"]
    .mean()
    .round(4)
)

# Inspect incorrect predictions with high confidence.
errors = errors.sort_values(
    "confidence", ascending=False
)

print("\n=== TOP 10 HIGH-CONFIDENCE ERRORS ===")

for row in errors.head(10).itertuples():
    print("\n" + "=" * 70)
    print("CLAIM ID:", row.claim_id)
    print("CLAIM:", row.claim)
    print("EVIDENCE:", row.evidence[:1000])
    print("EXPECTED:", row.label)
    print("PREDICTED:", row.prediction)
    print(f"CONFIDENCE: {row.confidence:.4f}")

print("\n=== SUPPORTS PREDICTED AS NEUTRAL ===")

subset = errors[
    (errors["label"] == "SUPPORTS")
    & (errors["prediction"] == "NOT ENOUGH INFO")
]

for row in subset.head(5).itertuples():
    print("\n" + "=" * 70)
    print("CLAIM:", row.claim)
    print("EVIDENCE:", row.evidence[:1000])
    print(f"CONFIDENCE: {row.confidence:.4f}")
