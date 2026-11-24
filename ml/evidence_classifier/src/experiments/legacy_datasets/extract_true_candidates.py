import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / "data/raw/claim_reviews.json"
OUTPUT = ROOT / "data/processed/claimreview_true_candidates.csv"

START_DATE = "2022-02-24"

UKRAINE_PATTERN = re.compile(
    r"\b(?:ukrain\w*|kyiv|kiev|zelensk\w*|"
    r"mariupol|kharkiv|kherson|bucha|"
    r"donbas|donbass|crimea|"
    r"russia\w*|russian\w*|putin)\b",
    re.IGNORECASE,
)

with INPUT.open("r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

for record in data:

    if record.get("label") != "credible":
        continue

    checker = record.get("fact_checker") or {}

    if checker.get("language") != "English":
        continue

    claims = record.get("claim_text") or []

    if isinstance(claims, str):
        claims = [claims]

    reviews = record.get("reviews") or []

    recent_reviews = [
        review
        for review in reviews
        if isinstance(review.get("date_published"), str)
        and review["date_published"][:10] >= START_DATE
        and review.get("label") == "credible"
    ]

    if not recent_reviews:
        continue

    for claim in claims:

        if not isinstance(claim, str):
            continue

        claim = " ".join(claim.split())

        if not UKRAINE_PATTERN.search(claim):
            continue

        for review in recent_reviews:
            rows.append({
                "claim": claim,
                "label": "TRUE",
                "date": review["date_published"][:10],
                "original_rating": review.get("original_label"),
                "fact_checker": checker.get("name"),
                "factcheck_url": record.get("review_url"),
            })

df = pd.DataFrame(rows, columns=[
    "claim", "label", "date",
    "original_rating", "fact_checker",
    "factcheck_url"
])

df = df.drop_duplicates(subset=["claim"])
df = df.sort_values("date").reset_index(drop=True)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False, encoding="utf-8")

print("=== TRUE CANDIDATE EXTRACTION ===")
print("Candidates:", len(df))
print("Unique fact-checkers:", df["fact_checker"].nunique())

print("\nDate range:")
print(df["date"].min() if not df.empty else "N/A")
print(df["date"].max() if not df.empty else "N/A")

print("\nTop fact-checkers:")
print(df["fact_checker"].value_counts().head(10))

print("\nFirst 10 claims:")
for i, claim in enumerate(df["claim"].head(10), 1):
    print(f"{i}. {claim[:200]}")

print("\nSaved to:", OUTPUT)
