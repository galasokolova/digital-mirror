import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = BASE_DIR / "data" / "raw" / "fever"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "fever"


def readable_title(page_id):
    return (
        page_id.replace("_", " ")
        .replace("-LRB-", "(")
        .replace("-RRB-", ")")
        .replace("-LSB-", "[")
        .replace("-RSB-", "]")
    )


def load_titles(jsonl_path, required_claim_ids):
    titles = {}

    with open(jsonl_path, encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            claim_id = int(record["id"])

            if claim_id not in required_claim_ids:
                continue

            for set_id, evidence_set in enumerate(
                record.get("evidence", [])
            ):
                pages = []

                for item in evidence_set:
                    if len(item) < 4:
                        continue

                    page_id = item[2]
                    sentence_id = item[3]

                    if page_id is None or sentence_id is None:
                        continue

                    if page_id not in pages:
                        pages.append(page_id)

                if pages:
                    titles[(claim_id, set_id)] = "; ".join(
                        readable_title(page)
                        for page in pages
                    )

    return titles


def process_split(name, raw_file):
    input_file = PROCESSED_DIR / f"{name}_verified.csv"
    output_file = (
        PROCESSED_DIR / f"{name}_context.csv"
    )

    print(f"\n=== {name.upper()} ===", flush=True)

    df = pd.read_csv(input_file)

    required_ids = set(df["claim_id"].astype(int))

    title_lookup = load_titles(
        RAW_DIR / raw_file,
        required_ids,
    )

    keys = zip(
        df["claim_id"].astype(int),
        df["evidence_set_id"].astype(int),
    )

    df["wiki_titles"] = [
        title_lookup.get(key, "")
        for key in keys
    ]

    df["context_evidence"] = [
        f"Wikipedia article: {title}. {evidence}"
        if title
        else evidence
        for title, evidence in zip(
            df["wiki_titles"],
            df["evidence"],
        )
    ]

    missing = (df["wiki_titles"] == "").sum()

    print("Rows:", len(df))
    print("Rows with titles:", len(df) - missing)
    print("Rows without titles:", missing)

    if missing:
        print(
            "WARNING: Missing titles detected. "
            "Inspect before training."
        )

    df.to_csv(output_file, index=False)

    print("Saved:", output_file)

    print("\nExamples:")

    for row in df.head(3).itertuples():
        print("\nCLAIM:", row.claim)
        print("WIKIPEDIA:", row.wiki_titles)
        print("EVIDENCE:", row.evidence[:200])


def main():
    process_split(
        "train",
        "train.jsonl",
    )

    process_split(
        "dev",
        "shared_task_dev.jsonl",
    )

    print("\nE05 context preparation completed.")


if __name__ == "__main__":
    main()
