import json
import re
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / "data/raw/claim_reviews.json"
OUTPUT = ROOT / "data/processed/relations_candidates.csv"

START_DATE = "2014-01-01"
END_DATE = "2026-11-20"

UKRAINE_PATTERN = re.compile(
    r"\b(?:"
    r"ukrain\w*|kyiv|kiev|zelensk\w*|"
    r"crimea\w*|donbas\w*|donbass|"
    r"donetsk|luhansk|lugansk|"
    r"mariupol|kharkiv|kherson|bucha|"
    r"yanukovych"
    r")\b",
    re.IGNORECASE,
)

RUSSIA_PATTERN = re.compile(
    r"\b(?:"
    r"russia\w*|putin|kremlin|"
    r"moscow|russian-backed|"
    r"russian-occupied"
    r")\b",
    re.IGNORECASE,
)

RELATIONS_PATTERN = re.compile(
    r"\b(?:"
    r"minsk agreements?|"
    r"annexation of crimea|"
    r"war in ukraine|"
    r"invasion of ukraine|"
    r"russo-ukrainian"
    r")\b",
    re.IGNORECASE,
)

def is_relevant(claim):
    return (
        (
            UKRAINE_PATTERN.search(claim)
            and RUSSIA_PATTERN.search(claim)
        )
        or RELATIONS_PATTERN.search(claim)
    )


rows = []
labels = Counter()
sources = Counter()

with INPUT.open("r", encoding="utf-8") as f:
    data = json.load(f)

for record in data:
    checker = record.get("fact_checker") or {}

    if checker.get("language") != "English":
        continue

    claims = record.get("claim_text") or []
    if isinstance(claims, str):
        claims = [claims]

    reviews = record.get("reviews") or []

    recent_reviews = [
        review for review in reviews
        if isinstance(review.get("date_published"), str)
        and START_DATE <= review["date_published"][:10] <= END_DATE
    ]

    if not recent_reviews:
        continue

    for claim in claims:
        if not isinstance(claim, str):
            continue

        claim = " ".join(claim.split())

        if not is_relevant(claim):
            continue

        label = record.get("label")


        review = recent_reviews[0]

        rows.append({
            "claim": claim,
            "label": label,
            "date": review["date_published"][:10],
            "original_rating": review.get("original_label"),
            "fact_checker": checker.get("name"),
            "factcheck_url": record.get("review_url"),
        })

        labels[label] += 1
        sources[(label, checker.get("name"))] += 1

import pandas as pd

df = pd.DataFrame(rows)
df = df.drop_duplicates(subset=["claim"])
df = df.sort_values("date").reset_index(drop=True)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False, encoding="utf-8")

print("=== RUSSIA–UKRAINE RELATIONS ===")
print("Total unique claims:", len(df))

print("\nLABEL DISTRIBUTION:")
print(df["label"].value_counts(dropna=False))

print("\nTRUE CANDIDATES BY SOURCE:")
true_df = df[df["label"] == "credible"]
print(true_df["fact_checker"].value_counts().head(15))

print("\nFIRST 15 TRUE CANDIDATES:")
for i, claim in enumerate(true_df["claim"].head(15), 1):
    print(f"{i}. {claim[:220]}")

print("\nSaved to:", OUTPUT)
