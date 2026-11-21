import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = BASE_DIR / "data" / "experiments" / "fever"

INPUT_FILE = DATA_DIR / "e04_random_titles.csv"

WIKI_FILE = (
    BASE_DIR / "data" / "raw"
    / "fever" / "wiki-pages.zip"
)

OUTPUT_FILE = DATA_DIR / "e04_evidence_text_audit.csv"


def normalize(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def parse_wiki_lines(lines):
    sentences = {}

    for line in lines.split("\n"):
        if not line.strip():
            continue

        parts = line.split("\t")

        if len(parts) < 2:
            continue

        try:
            sentence_id = int(parts[0])
        except ValueError:
            continue

        sentences[sentence_id] = parts[1]

    return sentences



def load_required_wikipedia_sentences(required_refs):
    pages = {}

    skipped_empty = 0
    skipped_invalid = 0

    with zipfile.ZipFile(WIKI_FILE) as archive:
        jsonl_files = [
            name for name in archive.namelist()
            if name.endswith(".jsonl")
            and not name.startswith("__MACOSX/")
        ]

        print("Wikipedia files:", len(jsonl_files), flush=True)

        for index, filename in enumerate(jsonl_files, 1):
            with archive.open(filename) as file:
                for line_number, raw_line in enumerate(file, 1):

                    # Ignore empty or whitespace-only lines
                    if not raw_line.strip():
                        skipped_empty += 1
                        continue

                    try:
                        record = json.loads(raw_line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        skipped_invalid += 1

                        if skipped_invalid <= 5:
                            print(
                                f"Invalid JSON: {filename}, "
                                f"line {line_number}, "
                                f"preview={raw_line[:100]!r}"
                            )

                        continue

                    if not isinstance(record, dict):
                        continue

                    page_id = record.get("id")

                    if page_id not in required_refs:
                        continue

                    all_sentences = parse_wiki_lines(
                        record.get("lines", "")
                    )

                    pages[page_id] = {
                        sentence_id: all_sentences.get(sentence_id)
                        for sentence_id in required_refs[page_id]
                    }

            if index % 10 == 0 or index == len(jsonl_files):
                print(
                    f"Scanned Wikipedia files: "
                    f"{index}/{len(jsonl_files)}",
                    flush=True,
                )

    print("\n=== ARCHIVE DIAGNOSTICS ===")
    print("Empty lines skipped:", skipped_empty)
    print("Invalid JSON lines skipped:", skipped_invalid)
    print("Required pages:", len(required_refs))
    print("Pages found:", len(pages))

    return pages



def main():
    df = pd.read_csv(INPUT_FILE)

    required_refs = {}

    for references_json in df["evidence_references"]:
        references = json.loads(references_json)

        for page_id, sentence_id in references:
            required_refs.setdefault(page_id, set()).add(
                int(sentence_id)
            )

    print("=== E04d: EVIDENCE TEXT AUDIT ===")
    print("Claims:", len(df))
    print("Required Wikipedia pages:", len(required_refs))

    pages = load_required_wikipedia_sentences(required_refs)

    results = []

    for row in df.itertuples():
        references = json.loads(row.evidence_references)

        sentences = []
        missing_pages = []
        missing_sentences = []

        for page_id, sentence_id in references:
            sentence_id = int(sentence_id)

            if page_id not in pages:
                missing_pages.append(page_id)
                continue

            sentence = pages[page_id].get(sentence_id)

            if sentence is None:
                missing_sentences.append(
                    f"{page_id}:{sentence_id}"
                )
                continue

            sentences.append(sentence)

        reconstructed = " ".join(sentences)

        if missing_pages:
            status = "MISSING_PAGE"
        elif missing_sentences:
            status = "MISSING_SENTENCE"
        elif normalize(reconstructed) == normalize(row.evidence):
            status = "EXACT_MATCH"
        else:
            status = "TEXT_MISMATCH"

        results.append({
            "claim_id": row.claim_id,
            "evidence_set_id": row.evidence_set_id,
            "status": status,
            "claim": row.claim,
            "stored_evidence": row.evidence,
            "reconstructed_evidence": reconstructed,
            "missing_pages": "; ".join(missing_pages),
            "missing_sentences": "; ".join(missing_sentences),
        })

    result_df = pd.DataFrame(results)

    print("\n=== RESULTS ===")

    counts = Counter(result_df["status"])

    for status in [
        "EXACT_MATCH",
        "TEXT_MISMATCH",
        "MISSING_PAGE",
        "MISSING_SENTENCE",
    ]:
        print(f"{status}: {counts[status]}")

    print("\n=== MISMATCH EXAMPLES ===")

    mismatches = result_df[
        result_df["status"] != "EXACT_MATCH"
    ]

    for row in mismatches.head(10).itertuples():
        print("\n" + "=" * 60)
        print("CLAIM:", row.claim)
        print("STATUS:", row.status)
        print("STORED:", row.stored_evidence[:500])
        print("RECONSTRUCTED:", row.reconstructed_evidence[:500])
        if row.missing_pages:
            print("MISSING PAGES:", row.missing_pages)
        if row.missing_sentences:
            print("MISSING SENTENCES:", row.missing_sentences)

    result_df.to_csv(OUTPUT_FILE, index=False)

    print("\nSaved:", OUTPUT_FILE)
    print("E04d completed.")


if __name__ == "__main__":
    main()
