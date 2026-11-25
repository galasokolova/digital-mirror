
import hashlib
from pathlib import Path

import pandas as pd

from preprocessing.intent_text_normalizer import (
    normalize_text,
    remove_punctuation,
)

from taxonomy.intent_mapping import INTENT_CLASSES


ML_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = (
    ML_ROOT
    / "data"
    / "raw"
    / "digital_mirror_contrastive"
)

OUTPUT_DIR = (
    ML_ROOT
    / "data"
    / "processed"
    / "digital_mirror_contrastive_v1"
)

EXPERIMENTS_DIR = ML_ROOT / "experiments"

INPUT_PATH = (
    RAW_DIR / "contrastive_intents_v1.csv"
)

OUTPUT_PATH = OUTPUT_DIR / "train.csv"

EXPERIMENT = "E09_07_06_04_01"

EXPECTED_RECORDS = 1200

ORIGINAL_COLUMNS = [
    "id",
    "text",
    "intent",
    "contrast_group",
    "topic",
    "style",
    "context",
    "source",
]

OUTPUT_COLUMNS = ORIGINAL_COLUMNS + [
    "normalized_text",
    "model_text",
]


def calculate_sha256(path):

    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def save_table(table, name):

    EXPERIMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    path = (
        EXPERIMENTS_DIR
        / f"{EXPERIMENT}_{name}.csv"
    )

    table.to_csv(
        path,
        index=False,
        encoding="utf-8"
    )

    print(f"Saved: {path.name}")


def load_dataset():

    dataset = pd.read_csv(
        INPUT_PATH,
        dtype=str,
        keep_default_na=False
    )

    print("\n=== INPUT DATASET ===")

    print(f"Records: {len(dataset)}")
    print(f"Columns: {len(dataset.columns)}")

    if list(dataset.columns) != ORIGINAL_COLUMNS:

        raise ValueError(
            "Unexpected input columns."
        )

    if len(dataset) != EXPECTED_RECORDS:

        raise ValueError(
            "Unexpected dataset size."
        )

    if not dataset["id"].is_unique:

        raise ValueError(
            "Duplicate record IDs."
        )

    if not dataset["text"].str.strip().ne("").all():

        raise ValueError(
            "Empty input texts."
        )

    if set(dataset["intent"]) != set(INTENT_CLASSES):

        raise ValueError(
            "Unexpected intent classes."
        )

    return dataset


def preprocess_dataset(dataset):

    processed = dataset.copy()

    processed["normalized_text"] = (
        processed["text"]
        .map(normalize_text)
    )

    processed["model_text"] = (
        processed["normalized_text"]
        .map(remove_punctuation)
    )

    return processed


def validate_preprocessing(original, processed):

    print("\n=== PREPROCESSING VALIDATION ===")

    original_counts = (
        original["intent"]
        .value_counts()
        .sort_index()
    )

    processed_counts = (
        processed["intent"]
        .value_counts()
        .sort_index()
    )

    checks = {
        "Expected output columns":
            list(processed.columns)
            == OUTPUT_COLUMNS,

        "Record count preserved":
            len(original) == len(processed),

        "Original columns preserved":
            processed[ORIGINAL_COLUMNS]
            .equals(original),

        "Record IDs preserved":
            original["id"]
            .equals(processed["id"]),

        "Unique record IDs":
            processed["id"].is_unique,

        "Original text preserved":
            original["text"]
            .equals(processed["text"]),

        "Intent labels preserved":
            original["intent"]
            .equals(processed["intent"]),

        "Class distribution preserved":
            original_counts.equals(
                processed_counts
            ),

        "Contrastive groups preserved":
            original["contrast_group"]
            .equals(
                processed["contrast_group"]
            ),

        "Context preserved":
            original["context"]
            .equals(processed["context"]),

        "No empty normalized texts":
            processed["normalized_text"]
            .str.strip()
            .ne("")
            .all(),

        "No empty model texts":
            processed["model_text"]
            .str.strip()
            .ne("")
            .all(),

        "Normalization reproducible":
            processed["normalized_text"]
            .equals(
                processed["text"]
                .map(normalize_text)
            ),

        "Model text reproducible":
            processed["model_text"]
            .equals(
                processed["normalized_text"]
                .map(remove_punctuation)
            ),
    }

    for name, passed in checks.items():

        status = "PASS" if passed else "FAIL"

        print(f"{status} | {name}")

    print(
        f"\nChecks passed: "
        f"{sum(checks.values())} / "
        f"{len(checks)}"
    )

    save_table(
        pd.DataFrame([
            {
                "check": name,
                "passed": bool(passed)
            }
            for name, passed in checks.items()
        ]),
        "preprocessing_checks"
    )

    if not all(checks.values()):

        raise ValueError(
            "Preprocessing validation failed."
        )


def analyze_transformations(dataset):

    print("\n=== TEXT TRANSFORMATIONS ===")

    normalization_changed = (
        dataset["text"]
        != dataset["normalized_text"]
    )

    punctuation_changed = (
        dataset["normalized_text"]
        != dataset["model_text"]
    )

    model_text_changed = (
        dataset["text"]
        != dataset["model_text"]
    )

    total = len(dataset)

    print(f"Total records: {total}")

    print(
        "Changed during normalization: "
        f"{normalization_changed.sum()} "
        f"({normalization_changed.mean():.2%})"
    )

    print(
        "Changed during punctuation removal: "
        f"{punctuation_changed.sum()} "
        f"({punctuation_changed.mean():.2%})"
    )

    print(
        "Changed from original to model text: "
        f"{model_text_changed.sum()} "
        f"({model_text_changed.mean():.2%})"
    )

    changes = dataset[[
        "id",
        "text",
        "normalized_text",
        "model_text",
        "intent",
        "style",
    ]].copy()

    changes["normalization_changed"] = (
        normalization_changed
    )

    changes["punctuation_changed"] = (
        punctuation_changed
    )

    changes["model_text_changed"] = (
        model_text_changed
    )

    save_table(
        changes,
        "text_transformations"
    )

    print("\n=== TRANSFORMATION EXAMPLES ===")

    print(
        changes[
            changes["model_text_changed"]
        ].head(20).to_string(index=False)
    )


def analyze_text_lengths(dataset):

    print("\n=== MODEL TEXT LENGTH ===")

    lengths = (
        dataset["model_text"]
        .str.split()
        .str.len()
    )

    print(lengths.describe().to_string())

    print("\n=== SHORT EXPRESSIONS ===")

    for maximum in [1, 2, 3, 5]:

        count = lengths.le(maximum).sum()

        print(
            f"At most {maximum} words: "
            f"{count} "
            f"({count / len(dataset):.2%})"
        )

    statistics = dataset.assign(
        word_count=lengths
    )

    style_stats = (
        statistics.groupby("style")["word_count"]
        .agg(
            records="size",
            mean="mean",
            median="median",
            minimum="min",
            maximum="max"
        )
        .reset_index()
    )

    print("\n=== LENGTH BY STYLE ===")

    print(
        style_stats.to_string(index=False)
    )

    save_table(
        style_stats,
        "model_text_length_by_style"
    )


def analyze_internal_collisions(dataset):

    print("\n=== INTERNAL COLLISION PREVIEW ===")

    duplicated = dataset[
        dataset["model_text"].duplicated(
            keep=False
        )
    ].copy()

    unique_model_texts = (
        dataset["model_text"].nunique()
    )

    intent_counts = (
        dataset.groupby("model_text")["intent"]
        .nunique()
    )

    conflicting_texts = intent_counts[
        intent_counts > 1
    ].index

    conflicts = dataset[
        dataset["model_text"].isin(
            conflicting_texts
        )
    ].copy()

    print(
        f"Unique model texts: "
        f"{unique_model_texts}"
    )

    print(
        f"Duplicate model text excess: "
        f"{len(dataset) - unique_model_texts}"
    )

    print(
        f"Records involved in duplicates: "
        f"{len(duplicated)}"
    )

    print(
        f"Conflicting unique model texts: "
        f"{len(conflicting_texts)}"
    )

    print(
        f"Records involved in conflicts: "
        f"{len(conflicts)}"
    )

    save_table(
        duplicated.sort_values("model_text"),
        "internal_duplicate_preview"
    )

    save_table(
        conflicts.sort_values("model_text"),
        "internal_conflict_preview"
    )

    print("Records removed: 0")


def save_processed_dataset(dataset):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    dataset.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    reloaded = pd.read_csv(
        OUTPUT_PATH,
        dtype=str,
        keep_default_na=False
    )

    if not reloaded.equals(dataset):

        raise RuntimeError(
            "Saved dataset verification failed."
        )

    print("\n=== OUTPUT DATASET ===")

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Records: {len(reloaded)}")
    print(f"Columns: {len(reloaded.columns)}")

    print(
        f"Output SHA256: "
        f"{calculate_sha256(OUTPUT_PATH)}"
    )

    print("Saved dataset verification: PASS")


def main():

    print(
        "=== E09.07.6.4.1: "
        "CONTRASTIVE DATASET PREPROCESSING ==="
    )

    original_hash = calculate_sha256(
        INPUT_PATH
    )

    print(
        f"\nInput SHA256: {original_hash}"
    )

    try:

        original = load_dataset()

        processed = preprocess_dataset(
            original
        )

        validate_preprocessing(
            original,
            processed
        )

        analyze_transformations(processed)

        analyze_text_lengths(processed)

        analyze_internal_collisions(
            processed
        )

        save_processed_dataset(processed)

        print("\n=== PREPROCESSING SUMMARY ===")

        print("Preprocessing: COMPLETE")
        print("Records removed: 0")
        print("Intent labels changed: 0")
        print("External overlap analysis: PENDING")
        print("Dataset ready for training: NO")
        print("Model loaded: NO")
        print("Model trained: NO")
        print("Evaluation data loaded: NO")

    finally:

        final_hash = calculate_sha256(
            INPUT_PATH
        )

        unchanged = original_hash == final_hash

        print("\n=== FILE INTEGRITY ===")

        print(f"Original SHA256: {original_hash}")
        print(f"Current SHA256:  {final_hash}")

        print(
            "Source dataset unchanged: "
            + ("YES" if unchanged else "NO")
        )

        if not unchanged:

            raise RuntimeError(
                "Source dataset was modified."
            )

    print("\n=== PREPROCESSING FINISHED ===")


if __name__ == "__main__":
    main()
