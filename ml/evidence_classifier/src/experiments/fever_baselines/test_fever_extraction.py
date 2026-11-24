import json
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
RAW_DIR = BASE_DIR / "data" / "raw" / "fever"

TRAIN_FILE = RAW_DIR / "train.jsonl"
WIKI_ZIP = RAW_DIR / "wiki-pages.zip"


def load_claims(path, limit=5):
    claims = []

    with open(path, encoding="utf-8") as file:
        for line in file:
            claim = json.loads(line)

            if claim["label"] == "NOT ENOUGH INFO":
                continue

            if not any(
                item[2] is not None
                for group in claim["evidence"]
                for item in group
            ):
                continue

            claims.append(claim)

            if len(claims) >= limit:
                break

    return claims


def get_evidence_references(claim):
    references = set()

    for group in claim["evidence"]:
        for item in group:
            page_id = item[2]
            sentence_id = item[3]

            if page_id is not None and sentence_id is not None:
                references.add((page_id, sentence_id))

    return references


def extract_sentences(zip_path, references):
    needed_pages = {page_id for page_id, _ in references}
    found = {}

    with zipfile.ZipFile(zip_path) as archive:
        wiki_files = [
            name for name in archive.namelist()
            if name.startswith("wiki-pages/")
            and name.endswith(".jsonl")
        ]

        for filename in wiki_files:
            with archive.open(filename) as file:
                for line in file:
                    page = json.loads(line)

                    if page["id"] not in needed_pages:
                        continue

                    for sentence_line in page["lines"].split("\n"):
                        parts = sentence_line.split("\t", 2)

                        if len(parts) < 2:
                            continue

                        try:
                            sentence_id = int(parts[0])
                        except ValueError:
                            continue

                        key = (page["id"], sentence_id)

                        if key in references and parts[1].strip():
                            found[key] = parts[1].strip()

            if len(found) == len(references):
                break

    return found


def main():
    claims = load_claims(TRAIN_FILE)
    references = set()

    for claim in claims:
        references.update(get_evidence_references(claim))

    print(f"Claims selected: {len(claims)}")
    print(f"Evidence references: {len(references)}")

    sentences = extract_sentences(WIKI_ZIP, references)

    print(f"Evidence sentences found: {len(sentences)}")

    for claim in claims:
        print("\n" + "=" * 60)
        print("CLAIM:", claim["claim"])
        print("LABEL:", claim["label"])

        for reference in sorted(get_evidence_references(claim)):
            print("EVIDENCE:", sentences.get(reference, "[NOT FOUND]"))


if __name__ == "__main__":
    main()
