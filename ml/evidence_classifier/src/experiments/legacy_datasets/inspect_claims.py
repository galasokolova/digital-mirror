from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

INPUT = ROOT / "data/processed/euvsdisinfo_false.csv"
OUTPUT = ROOT / "data/processed/claims_review.csv"

df = pd.read_csv(INPUT)

# Reproducible random sample
sample = df.sample(n=min(50, len(df)), random_state=42).copy()

# Columns for manual review
sample["checkable"] = ""
sample["notes"] = ""

sample = sample[
    [
        "claim",
        "original_text",
        "factcheck_text",
        "factcheck_url",
        "checkable",
        "notes",
    ]
]

sample.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print("Sample size:", len(sample))
print("Saved to:", OUTPUT)

print("\nFirst 10 claims:")
for i, claim in enumerate(sample["claim"].head(10), start=1):
    print(f"{i}. {claim}")
