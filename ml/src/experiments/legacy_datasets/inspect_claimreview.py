import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / "data/raw/claim_reviews.json"

print("=== CLAIMREVIEW DATASET INSPECTION ===")

with INPUT.open("r", encoding="utf-8") as f:
    first_char = f.read(1)

print("First character:", repr(first_char))

with INPUT.open("r", encoding="utf-8") as f:
    data = json.load(f)

print("Root type:", type(data).__name__)
print("Total records:", len(data))

if isinstance(data, list):
    print("\nFirst record:")
    print(json.dumps(data[0], indent=2, ensure_ascii=False)[:4000])

elif isinstance(data, dict):
    print("\nTop-level keys:", list(data.keys()))

    for key, value in data.items():
        print(
            f"{key}: {type(value).__name__}",
            f"length={len(value)}"
            if hasattr(value, "__len__") else ""
        )

    print("\nFirst available sample:")
    for key, value in data.items():
        if isinstance(value, list) and value:
            print(json.dumps(value[0], indent=2, ensure_ascii=False)[:4000])
            break
