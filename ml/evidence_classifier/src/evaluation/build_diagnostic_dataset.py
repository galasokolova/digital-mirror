import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]

CLAIMS_PATH = (
    ROOT / "orchestrator" / "src" / "main"
    / "resources" / "data" / "claims.json"
)

OUTPUT_PATH = (
    ROOT / "ml" / "evidence_classifier" / "data" / "evaluation"
    / "diagnostic_claims.json"
)

PAIRS = [
    ("fact-001", "fact-002", "entity"),
    ("fact-003", "fact-004", "number"),
    ("fact-005", "fact-006", "entity"),
    ("fact-007", "fact-008", "entity"),
    ("fact-009", "fact-010", "date"),
    ("fact-011", "fact-012", "negation"),
]


def main():
    with CLAIMS_PATH.open(encoding="utf-8") as file:
        claims = json.load(file)

    by_id = {claim["id"]: claim for claim in claims}

    dataset = []

    for supports_id, refutes_id, category in PAIRS:
        positive = by_id[supports_id]
        negative = by_id[refutes_id]

        # Both claims must use the same evidence.
        evidence = positive["evidence"]
        wiki_title = positive.get("wikiTitle")

        for claim, label in [
            (positive, "SUPPORTS"),
            (negative, "REFUTES"),
        ]:
            dataset.append({
                "id": claim["id"],
                "pairId": supports_id,
                "category": category,
                "claim": claim["claim"],
                "evidence": evidence,
                "wikiTitle": wiki_title,
                "verifiedLabel": label,
            })

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(dataset, file, indent=2, ensure_ascii=False)

    print(f"Created: {OUTPUT_PATH}")
    print(f"Total claims: {len(dataset)}")
    print(f"Pairs: {len(PAIRS)}")


if __name__ == "__main__":
    main()
