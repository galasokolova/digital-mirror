import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / "data/raw/claim_reviews.json"

with INPUT.open("r", encoding="utf-8") as f:
    data = json.load(f)

labels = Counter()
languages = Counter()
years = Counter()
original_labels = Counter()
label_pairs = Counter()

for record in data:
    label = record.get("label")
    labels[label] += 1

    checker = record.get("fact_checker") or {}
    language = checker.get("language")
    languages[language] += 1

    for review in record.get("reviews") or []:
        original = review.get("original_label")
        normalized = review.get("label")

        original_labels[original] += 1
        label_pairs[(original, normalized)] += 1

        date = review.get("date_published")
        if isinstance(date, str) and len(date) >= 4:
            years[date[:4]] += 1

print("=== LABEL DISTRIBUTION ===")
for label, count in labels.most_common():
    print(f"{str(label):25} {count}")

print("\n=== TOP 15 LANGUAGES ===")
for language, count in languages.most_common(15):
    print(f"{str(language):25} {count}")

print("\n=== REVIEWS BY YEAR ===")
for year, count in sorted(years.items()):
    print(year, count)

print("\n=== TOP 20 ORIGINAL LABELS ===")
for label, count in original_labels.most_common(20):
    print(f"{str(label):35} {count}")

print("\n=== EXAMPLES OF LABEL MAPPINGS ===")
for pair, count in label_pairs.most_common(30):
    print(f"{str(pair):65} {count}")
