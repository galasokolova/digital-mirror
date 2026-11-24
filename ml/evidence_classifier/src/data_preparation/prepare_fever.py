import csv
import json
import zipfile
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw" / "fever"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "fever"

SPLITS = {
    "train": RAW_DIR / "train.jsonl",
    "dev": RAW_DIR / "shared_task_dev.jsonl",
}

WIKI_ZIP = RAW_DIR / "wiki-pages.zip"


def load_claims(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def get_evidence_sets(claim):
    evidence_sets = []

    for group in claim["evidence"]:
        references = []

        for item in group:
            page_id = item[2]
            sentence_id = item[3]

            if page_id is not None and sentence_id is not None:
                reference = (page_id, int(sentence_id))

                if reference not in references:
                    references.append(reference)

        if references:
            evidence_sets.append(references)

    return evidence_sets


def collect_references(datasets):
    references = set()

    for claims in datasets.values():
        for claim in claims:
            if claim["label"] == "NOT ENOUGH INFO":
                continue

            for group in get_evidence_sets(claim):
                references.update(group)

    return references


def extract_evidence(references):
    needed_pages = {page_id for page_id, _ in references}
    found = {}

    with zipfile.ZipFile(WIKI_ZIP) as archive:
        wiki_files = sorted(
            name for name in archive.namelist()
            if name.startswith("wiki-pages/")
            and name.endswith(".jsonl")
        )

        print(f"Wikipedia files: {len(wiki_files)}", flush=True)

        for index, filename in enumerate(wiki_files, start=1):
            with archive.open(filename) as file:
                for line in file:
                    page = json.loads(line)

                    page_id = page.get("id")

                    if page_id not in needed_pages:
                        continue

                    for sentence_line in page.get("lines", "").split("\n"):
                        parts = sentence_line.split("\t", 2)

                        if len(parts) < 2:
                            continue

                        try:
                            sentence_id = int(parts[0])
                        except ValueError:
                            continue

                        key = (page_id, sentence_id)

                        if key in references and parts[1].strip():
                            found[key] = parts[1].strip()

            print(
                f"[{index}/{len(wiki_files)}] "
                f"Found {len(found)}/{len(references)}",
                flush=True
            )

            if len(found) == len(references):
                break

    return found


def save_dataset(split, claims, sentences):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    verified_path = OUTPUT_DIR / f"{split}_verified.csv"
    nei_path = OUTPUT_DIR / f"{split}_nei.csv"

    stats = Counter()

    with (
        verified_path.open("w", newline="", encoding="utf-8") as verified_file,
        nei_path.open("w", newline="", encoding="utf-8") as nei_file
    ):
        verified_writer = csv.DictWriter(
            verified_file,
            fieldnames=[
                "claim_id",
                "claim",
                "evidence_set_id",
                "evidence",
                "label",
                "split",
            ],
        )

        nei_writer = csv.DictWriter(
            nei_file,
            fieldnames=["claim_id", "claim", "label", "split"],
        )

        verified_writer.writeheader()
        nei_writer.writeheader()

        for claim in claims:
            label = claim["label"]
            stats[f"claims_{label}"] += 1

            if label == "NOT ENOUGH INFO":
                nei_writer.writerow({
                    "claim_id": claim["id"],
                    "claim": claim["claim"],
                    "label": label,
                    "split": split,
                })

                stats["nei_rows"] += 1
                continue

            evidence_sets = get_evidence_sets(claim)
            complete_sets = 0

            for set_id, group in enumerate(evidence_sets):
                if not all(ref in sentences for ref in group):
                    stats["incomplete_evidence_sets"] += 1
                    continue

                evidence_text = " ".join(
                    sentences[ref] for ref in group
                )

                verified_writer.writerow({
                    "claim_id": claim["id"],
                    "claim": claim["claim"],
                    "evidence_set_id": set_id,
                    "evidence": evidence_text,
                    "label": label,
                    "split": split,
                })

                stats["verified_rows"] += 1
                complete_sets += 1

            if complete_sets:
                stats["claims_with_evidence"] += 1
            else:
                stats["claims_without_complete_evidence"] += 1

    print(f"\n=== {split.upper()} ===")

    for key, value in sorted(stats.items()):
        print(f"{key}: {value}")

    print(f"Saved: {verified_path}")
    print(f"Saved: {nei_path}")


def main():
    print("Loading FEVER claims...", flush=True)

    datasets = {
        split: load_claims(path)
        for split, path in SPLITS.items()
    }

    for split, claims in datasets.items():
        print(f"{split}: {len(claims)} claims")

    references = collect_references(datasets)

    print(f"\nUnique evidence references: {len(references)}")
    print("Extracting Wikipedia evidence...", flush=True)

    sentences = extract_evidence(references)

    print(f"\nEvidence found: {len(sentences)}/{len(references)}")

    for split, claims in datasets.items():
        save_dataset(split, claims, sentences)

    print("\nFEVER preparation completed.")


if __name__ == "__main__":
    main()
