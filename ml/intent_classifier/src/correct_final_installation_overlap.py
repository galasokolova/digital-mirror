
from pathlib import Path
import shutil

import pandas as pd

from analyze_final_installation_overlap import (
    load_reference_datasets,
)

from preprocessing.intent_text_normalizer import (
    remove_punctuation,
)


ML_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    ML_ROOT
    / "data"
    / "evaluation"
    / "digital_mirror_final"
)

DATASET_PATH = DATASET_DIR / "test.csv"

BACKUP_PATH = DATASET_DIR / "test_original_v1.csv"

CANDIDATE_PATH = DATASET_DIR / "test_candidate_v2.csv"

EXPERIMENTS_DIR = ML_ROOT / "experiments"

OVERLAP_PATH = (
    EXPERIMENTS_DIR
    / "E09_06_02d_overlap_details.csv"
)

CHANGES_PATH = (
    EXPERIMENTS_DIR
    / "E09_06_02d_01_replacements.csv"
)


REPLACEMENTS = {

    "DMT-0007": "Greetings to you",

    "DMT-0008": "Lovely morning",

    "DMT-0009": "Hello, good morning",

    "DMT-0010": "Hiya there",

    "DMT-0011":
        "Hello, interactive mirror, nice to meet you.",

    "DMT-0023":
        "I hope our paths cross again.",

    "DMT-0026": "Farewell for now",

    "DMT-0027": "Goodbye for today",

    "DMT-0029": "Bye for today",

    "DMT-0046": "Really appreciate that",

    "DMT-0047": "Thanks for everything",

    "DMT-0083":
        "My view differs considerably from yours.",

    "DMT-0106": "Possibly so",

    "DMT-0109": "Difficult to judge",

    "DMT-0122":
        "I've followed your explanation so far.",

    "DMT-0126": "Understood so far",

    "DMT-0127": "Duly noted",

    "DMT-0128": "Information received",

    "DMT-0146": "Yes, I shall",

    "DMT-0170": "No, not today",

    "DMT-0189": "How many participants?",
}


def load_dataset(path):

    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False
    )


def validate_replacement_ids(dataset):

    print("\n=== REPLACEMENT ID VALIDATION ===")

    overlap_details = load_dataset(
        OVERLAP_PATH
    )

    overlapping_ids = set(
        overlap_details["id"]
    )

    replacement_ids = set(
        REPLACEMENTS
    )

    if overlapping_ids != replacement_ids:

        raise ValueError(
            "Replacement IDs do not match "
            "the overlap report."
        )

    if not replacement_ids.issubset(
        set(dataset["id"])
    ):

        raise ValueError(
            "Unknown replacement IDs."
        )

    # Ensure that the overlap report matches
    # the current dataset.

    current_texts = (
        dataset.set_index("id")["text"]
        .apply(remove_punctuation)
    )

    for row in overlap_details.itertuples(index=False):

        if current_texts.loc[row.id] != row.model_text:

            raise ValueError(
                f"Outdated overlap report: {row.id}"
            )

    print(f"Overlapping IDs: {len(overlapping_ids)}")

    print(f"Replacement IDs: {len(replacement_ids)}")

    print("Replacement ID validation: PASSED")


def create_candidate(original):

    candidate = original.copy()

    changes = []

    for record_id, replacement in REPLACEMENTS.items():

        mask = candidate["id"].eq(
            record_id
        )

        if mask.sum() != 1:

            raise ValueError(
                f"Invalid record ID: {record_id}"
            )

        row = candidate.loc[mask].iloc[0]

        changes.append({

            "id": record_id,

            "intent": row["intent"],

            "group": row["group"],

            "original_text": row["text"],

            "replacement_text": replacement,
        })

        candidate.loc[mask, "text"] = replacement

    return candidate, pd.DataFrame(changes)


def validate_candidate(original, candidate):

    print("\n=== CANDIDATE VALIDATION ===")

    model_texts = candidate["text"].apply(
        remove_punctuation
    )

    changed_mask = (
        original["text"] != candidate["text"]
    )

    changed_ids = set(
        candidate.loc[changed_mask, "id"]
    )

    short_mask = candidate["group"].eq(
        "short"
    )

    short_lengths = (
        model_texts.loc[short_mask]
        .str.split()
        .str.len()
    )

    checks = {

        "Expected record count":
            len(candidate) == 240,

        "Exactly 21 changed records":
            changed_mask.sum() == 21,

        "Only overlapping IDs changed":
            changed_ids == set(REPLACEMENTS),

        "All non-text columns preserved":
            original.drop(columns="text").equals(
                candidate.drop(columns="text")
            ),

        "Unique IDs":
            candidate["id"].is_unique,

        "Unique model texts":
            model_texts.is_unique,

        "No empty model texts":
            not model_texts.eq("").any(),

        "Short group contains 1-3 words":
            short_lengths.between(1, 3).all(),

        "Balanced intent-group distribution":
            pd.crosstab(
                candidate["intent"],
                candidate["group"]
            ).eq(5).all().all(),
    }

    for name, passed in checks.items():

        status = "PASS" if passed else "FAIL"

        print(f"{status} | {name}")

    if not all(checks.values()):

        raise ValueError(
            "Candidate validation failed."
        )


def check_reference_overlap(candidate):

    print("\n=== REPLACEMENT OVERLAP CHECK ===")

    references = load_reference_datasets()

    model_texts = candidate["text"].apply(
        remove_punctuation
    )

    all_overlapping_ids = set()

    for name, reference in references.items():

        reference_texts = set(
            reference["model_text"]
        )

        mask = model_texts.isin(
            reference_texts
        )

        matches = candidate.loc[mask]

        all_overlapping_ids.update(
            matches["id"]
        )

        print(f"{name}: {len(matches)}")

        if not matches.empty:

            print(
                matches[
                    ["id", "text", "intent", "group"]
                ].to_string(index=False)
            )

    print(
        "\nUnique remaining overlaps: "
        f"{len(all_overlapping_ids)}"
    )

    return len(all_overlapping_ids) == 0


def main():

    print(
        "=== E09.06.2d.1: DATASET "
        "OVERLAP CORRECTION ==="
    )

    original = load_dataset(
        DATASET_PATH
    )

    validate_replacement_ids(
        original
    )

    candidate, changes = create_candidate(
        original
    )

    print("\n=== PROPOSED REPLACEMENTS ===")

    print(
        changes.to_string(index=False)
    )

    validate_candidate(
        original,
        candidate
    )

    overlap_free = check_reference_overlap(
        candidate
    )

    # Save the candidate for inspection.

    candidate.to_csv(
        CANDIDATE_PATH,
        index=False,
        encoding="utf-8"
    )

    changes.to_csv(
        CHANGES_PATH,
        index=False,
        encoding="utf-8"
    )

    if not overlap_free:

        print("\nSTATUS: REVIEW REQUIRED")

        print("Original dataset modified: NO")

        print(
            "Some replacement texts "
            "still overlap with reference data."
        )

        raise SystemExit(1)

    # Preserve the original dataset.

    if BACKUP_PATH.exists():

        raise FileExistsError(
            f"Backup already exists: {BACKUP_PATH}"
        )

    shutil.copy2(
        DATASET_PATH,
        BACKUP_PATH
    )

    # Replace the dataset only after validation.

    candidate.to_csv(
        DATASET_PATH,
        index=False,
        encoding="utf-8"
    )

    print("\n=== FINAL CORRECTION STATUS ===")

    print(f"Original records: {len(original)}")

    print(f"Corrected records: {len(changes)}")

    print(
        f"Unchanged records: "
        f"{len(original) - len(changes)}"
    )

    print(f"Final records: {len(candidate)}")

    print("Remaining exact overlaps: 0")

    print(f"Backup: {BACKUP_PATH}")

    print(f"Updated dataset: {DATASET_PATH}")

    print("Original labels modified: NO")
    print("Original groups modified: NO")
    print("Model loaded: NO")
    print("Model training performed: NO")

    print("\nSTATUS: PASSED")

    print("\n=== OVERLAP CORRECTION COMPLETE ===")


if __name__ == "__main__":
    main()
