import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "experiments" / "fever"

SAMPLE_FILE = DATA_DIR / "e04_random_titles.csv"

FEVER_FILE = (
    BASE_DIR / "data" / "raw"
    / "fever" / "shared_task_dev.jsonl"
)

OUTPUT_FILE = DATA_DIR / "e04_integrity_audit.csv"


def normalize_title(page_id):
    return (
        page_id.replace("_", " ")
        .replace("-LRB-", "(")
        .replace("-RRB-", ")")
    )


def normalize_text(value):
    return re.sub(r"\s+", " ", str(value)).strip()


def get_original_references(record, set_id):
    evidence_sets = record.get("evidence", [])

    if set_id < 0 or set_id >= len(evidence_sets):
        return None

    references = []

    for item in evidence_sets[set_id]:
        if len(item) < 4:
            continue

        page_id = item[2]
        sentence_id = item[3]

        if page_id is not None and sentence_id is not None:
            references.append((str(page_id), int(sentence_id)))

    return references


def main():
    df = pd.read_csv(SAMPLE_FILE)

    wanted_ids = set(df["claim_id"].astype(int))

    records = {}

    with open(FEVER_FILE, encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            claim_id = int(record["id"])

            if claim_id in wanted_ids:
                records[claim_id] = record

    results = []

    for row in df.itertuples():
        claim_id = int(row.claim_id)
        set_id = int(row.evidence_set_id)

        record = records.get(claim_id)

        if record is None:
            results.append({
                "claim_id": claim_id,
                "status": "MISSING_RECORD",
            })
            continue

        original_refs = get_original_references(
            record, set_id
        )

        if original_refs is None:
            status = "INVALID_SET_ID"
            original_refs = []
        else:
            stored_refs = json.loads(
                row.evidence_references
            )

            stored_refs = [
                (str(page), int(sentence))
                for page, sentence in stored_refs
            ]

            original_titles = "; ".join(
                dict.fromkeys(
                    normalize_title(page)
                    for page, _ in original_refs
                )
            )

            references_match = (
                original_refs == stored_refs
            )

            titles_match = (
                normalize_text(original_titles)
                == normalize_text(row.wiki_titles)
            )

            if references_match and titles_match:
                status = "MATCH"
            elif not references_match:
                status = "REFERENCE_MISMATCH"
            else:
                status = "TITLE_MISMATCH"

        results.append({
            "claim_id": claim_id,
            "evidence_set_id": set_id,
            "status": status,
            "claim": row.claim,
            "wiki_titles": row.wiki_titles,
            "evidence": row.evidence,
        })

    result_df = pd.DataFrame(results)

    print("\n=== E04c: EVIDENCE INTEGRITY AUDIT ===")

    print("Claims checked:", len(result_df))

    counts = Counter(result_df["status"])

    for status, count in counts.items():
        print(f"{status}: {count}")

    mismatches = result_df[
        result_df["status"] != "MATCH"
    ]

    print("\n=== MISMATCH EXAMPLES ===")

    for row in mismatches.head(10).itertuples():
        print("\nCLAIM:", row.claim)
        print("STATUS:", row.status)
        print("TITLE:", row.wiki_titles)

    result_df.to_csv(OUTPUT_FILE, index=False)

    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
