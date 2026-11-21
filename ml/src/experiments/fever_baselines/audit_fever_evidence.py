from pathlib import Path
import re
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
INPUT = (
    BASE_DIR / "data" / "experiments"
    / "fever" / "e03_nli_sample.csv"
)

df = pd.read_csv(INPUT)

pronoun_pattern = re.compile(
    r"^\s*(it|he|she|they|this|that|these|those|"
    r"his|her|its|their)\b",
    re.IGNORECASE,
)

df["starts_with_pronoun"] = (
    df["evidence"].fillna("").apply(
        lambda text: bool(pronoun_pattern.search(text))
    )
)

df["is_error"] = df["label"] != df["prediction"]

print("=== E03b: EVIDENCE CONTEXT AUDIT ===")
print("Total examples:", len(df))
print("Evidence starts with pronoun:",
      int(df["starts_with_pronoun"].sum()))

print("\nError rates:")
for flag in [True, False]:
    subset = df[df["starts_with_pronoun"] == flag]
    if len(subset):
        print(
            f"Starts with pronoun={flag}: "
            f"{subset['is_error'].mean():.2%} "
            f"({subset['is_error'].sum()}/{len(subset)})"
        )

print("\nError rates by label and pronoun:")
print(
    df.groupby(["label", "starts_with_pronoun"])
    ["is_error"]
    .agg(["count", "sum", "mean"])
    .round(4)
)

print("\n=== PRONOUN-START ERRORS ===")
subset = df[
    df["starts_with_pronoun"] & df["is_error"]
]

for row in subset.head(10).itertuples():
    print("\n" + "=" * 60)
    print("CLAIM:", row.claim)
    print("EVIDENCE:", row.evidence[:500])
    print("EXPECTED:", row.label)
    print("PREDICTED:", row.prediction)

print("\nAudit completed.")
