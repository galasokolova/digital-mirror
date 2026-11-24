import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / "data/raw/claim_reviews.json"

START_DATE = "2022-02-24"

PATTERN = re.compile(
    r"\b(?:ukrain\w*|kyiv|kiev|zelensk\w*|"
    r"mariupol|kharkiv|kherson|bucha|"
    r"donbas|donbass|crimea)\b",
    re.IGNORECASE,
)

labels = Counter()
sources = Counter()
examples = {}

with INPUT.open("r", encoding="utf-8") as f:
    data = json.load(f)

for record in data:
    checker = record.get("fact_checker") or {}

    if checker.get("language") != "English":
        continue

    claims = record.get("claim_text") or []
    if isinstance(claims, str):
        claims = [claims]

    matching_claims = [
        " ".join(claim.split())
        for claim in claims
        if isinstance(claim, str) and PATTERN.search(claim)
    ]

    if not matching_claims:
        continue

    reviews = record.get("reviews") or []

    recent_reviews = [
        review for review in reviews
        if isinstance(review.get("date_published"), str)
        and review["date_published"][:10] >= START_DATE
    ]

    if not recent_reviews:
        continue

    label = record.get("label")
    labels[label] += 1
    sources[(label, checker.get("name"))] += 1

    if label not in examples:
        examples[label] = matching_claims[0]

print("=== UKRAINE CLAIMREVIEW ANALYSIS ===")
print("Total records:", sum(labels.values()))

print("\nLABEL DISTRIBUTION:")
for label, count in labels.most_common():
    print(f"{str(label):25} {count}")

print("\nTOP SOURCES BY LABEL:")
for label in labels:
    print(f"\n{label}:")
    for (source_label, source), count in sources.most_common():
        if source_label == label:
            print(f"  {str(source):30} {count}")
            if sum(
                1 for (l, s), c in sources.most_common()
                if l == label and c > count
            ) >= 9:
                break

print("\nEXAMPLES:")
for label, claim in examples.items():
    print(f"\n{label}: {claim[:250]}")
