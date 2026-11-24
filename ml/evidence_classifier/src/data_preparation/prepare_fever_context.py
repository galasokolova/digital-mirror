import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

SAMPLE_FILE = (
    BASE_DIR / "data" / "experiments"
    / "fever" / "e03_nli_sample.csv"
)

FEVER_FILE = (
    BASE_DIR / "data" / "raw"
    / "fever" / "shared_task_dev.jsonl"
)

OUTPUT_FILE = (
    BASE_DIR / "data" / "experiments"
    / "fever" / "e03_context_sample.csv"
)


def extract_evidence_sets(record):
    """Extract complete evidence sets from FEVER JSONL."""

    result = []

    for evidence_set in record.get("evidence", []):
        references = []

        for item in evidence_set:
            if len(item) < 4:
                continue

            page_id = item[2]
            sentence_id = item[3]

            if page_id is not None and sentence_id is not None:
                references.append(
                    (str(page_id), int(sentence_id))
                )

        if references:
            result.append(references)

    return result


def main():
    df = pd.read_csv(SAMPLE_FILE)

    wanted_ids = set(df["claim_id"].astype(int))

    print("Sample claims:", len(df))

    records = {}

    with open(FEVER_FILE, "r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)

            claim_id = int(record["id"])

            if claim_id in wanted_ids:
                records[claim_id] = record

    print("Original FEVER records found:", len(records))

    titles = []
    evidence_refs = []
    evidence_set_counts = []

    for row in df.itertuples():
        record = records.get(int(row.claim_id))

        if record is None:
            titles.append("")
            evidence_refs.append("")
            evidence_set_counts.append(0)
            continue

        sets = extract_evidence_sets(record)

        evidence_set_counts.append(len(sets))

        # Use the evidence set index retained by preparation.
        index = int(row.evidence_set_id)

        if index < 0 or index >= len(sets):
            titles.append("")
            evidence_refs.append("")
            continue

        references = sets[index]

        unique_pages = list(
            dict.fromkeys(page_id for page_id, _ in references)
        )

        readable_titles = [
            page.replace("_", " ")
                .replace("-LRB-", "(")
                .replace("-RRB-", ")")
            for page in unique_pages
        ]

        titles.append("; ".join(readable_titles))

        evidence_refs.append(
            json.dumps(references, ensure_ascii=False)
        )

    df["wiki_titles"] = titles
    df["evidence_references"] = evidence_refs
    df["available_evidence_sets"] = evidence_set_counts

    df["context_evidence"] = df.apply(
        lambda row: (
            f"Wikipedia article: {row['wiki_titles']}. "
            f"{row['evidence']}"
            if row["wiki_titles"]
            else row["evidence"]
        ),
        axis=1,
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print("\n=== CONTEXT PREPARATION ===")
    print("Rows:", len(df))
    print("Rows with titles:", (df["wiki_titles"] != "").sum())
    print("Rows without titles:", (df["wiki_titles"] == "").sum())

    print("\nExamples:")

    for row in df.head(5).itertuples():
        print("\nCLAIM:", row.claim)
        print("WIKIPEDIA:", row.wiki_titles)
        print("EVIDENCE:", row.evidence[:200])

    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
