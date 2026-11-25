
from pathlib import Path

import pandas as pd

from taxonomy.intent_mapping import INTENT_CLASSES

from preprocessing.intent_text_normalizer import (
    remove_punctuation,
)


ML_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ML_ROOT / "data" / "processed"

EXPERIMENTS_DIR = ML_ROOT / "experiments"

SWITCHBOARD_DIR = PROCESSED_DIR / "intent_v1"

DOMAIN_DIR = PROCESSED_DIR / "digital_mirror_v1"

FINAL_TEST_DIR = (
    ML_ROOT
    / "data"
    / "evaluation"
    / "digital_mirror_final"
)

FINAL_TEST_PATH = (
    FINAL_TEST_DIR / "test_processed.csv"
)

BENCHMARK_PATH = (
    EXPERIMENTS_DIR
    / "E09_03_installation_predictions.csv"
)

EXPERIMENT = "E09_06_02d"

TEXT_COLUMN = "model_text"
TARGET_COLUMN = "intent"

EXPECTED_GROUPS = [
    "natural",
    "short",
    "installation",
    "complex",
]


def load_dataset(path):

    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False
    )


def load_reference_dataset(path):

    return pd.read_csv(
        path,
        usecols=[
            TEXT_COLUMN,
            TARGET_COLUMN,
        ],
        dtype=str,
        keep_default_na=False
    )


def load_benchmark():

    benchmark = load_dataset(
        BENCHMARK_PATH
    )

    benchmark = (
        benchmark[["text", "expected"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    benchmark = benchmark.rename(
        columns={
            "expected": TARGET_COLUMN
        }
    )

    benchmark[TEXT_COLUMN] = (
        benchmark["text"]
        .apply(remove_punctuation)
    )

    if len(benchmark) != 60:

        raise ValueError(
            "Expected 60 benchmark records."
        )

    if not benchmark[TEXT_COLUMN].is_unique:

        raise ValueError(
            "Duplicate benchmark model texts."
        )

    return benchmark[
        [TEXT_COLUMN, TARGET_COLUMN]
    ]


def load_reference_datasets():

    references = {

        "switchboard_train":
            load_reference_dataset(
                SWITCHBOARD_DIR / "train.csv"
            ),

        "switchboard_validation":
            load_reference_dataset(
                SWITCHBOARD_DIR / "validation.csv"
            ),

        "switchboard_test":
            load_reference_dataset(
                SWITCHBOARD_DIR / "test.csv"
            ),

        "digital_mirror_original_train":
            load_reference_dataset(
                DOMAIN_DIR / "train.csv"
            ),

        "digital_mirror_eligible_train":
            load_reference_dataset(
                DOMAIN_DIR / "train_eligible.csv"
            ),

        "installation_development_benchmark":
            load_benchmark(),
    }

    return references


def validate_datasets(final_test, references):

    print("\n=== DATASET VALIDATION ===")

    expected_sizes = {

        "switchboard_train": 135827,

        "switchboard_validation": 16913,

        "switchboard_test": 17966,

        "digital_mirror_original_train": 600,

        "digital_mirror_eligible_train": 579,

        "installation_development_benchmark": 60,
    }

    checks = {

        "Expected final test size":
            len(final_test) == 240,

        "Unique final test IDs":
            final_test["id"].is_unique,

        "Unique final test model texts":
            final_test[TEXT_COLUMN].is_unique,

        "No empty final test model texts":
            not final_test[TEXT_COLUMN]
            .eq("").any(),

        "All intent classes present":
            set(final_test[TARGET_COLUMN])
            == set(INTENT_CLASSES),

        "Balanced final test classes":
            final_test[TARGET_COLUMN]
            .value_counts().eq(20).all(),

        "Balanced final test groups":
            set(final_test["group"])
            == set(EXPECTED_GROUPS)
            and final_test["group"]
            .value_counts().eq(60).all(),
    }

    for name, expected_size in expected_sizes.items():

        checks[f"Expected size: {name}"] = (
            len(references[name])
            == expected_size
        )

    for name, passed in checks.items():

        status = "PASS" if passed else "FAIL"

        print(f"{status} | {name}")

    if not all(checks.values()):

        raise ValueError(
            "Dataset validation failed."
        )


def determine_overlap_status(
    expected_label,
    reference_labels
):

    if not reference_labels:

        return "NO_OVERLAP"

    if reference_labels == {expected_label}:

        return "SAME_LABEL"

    if expected_label in reference_labels:

        return "AMBIGUOUS_LABELS"

    return "CONFLICTING_LABEL"


def analyze_reference(
    final_test,
    reference,
    reference_name
):

    print(f"\n{'=' * 70}")
    print(f"REFERENCE: {reference_name.upper()}")
    print(f"{'=' * 70}")

    reference_lookup = (
        reference.groupby(TEXT_COLUMN)[TARGET_COLUMN]
        .agg(set)
        .to_dict()
    )

    matches = []

    for row in final_test.itertuples(index=False):

        reference_labels = reference_lookup.get(
            row.model_text,
            set()
        )

        status = determine_overlap_status(
            row.intent,
            reference_labels
        )

        if status == "NO_OVERLAP":
            continue

        matches.append({

            "id": row.id,

            "text": row.text,

            "model_text": row.model_text,

            "intent": row.intent,

            "group": row.group,

            "reference": reference_name,

            "reference_labels": ", ".join(
                sorted(reference_labels)
            ),

            "status": status,
        })

    match_columns = [

        "id",
        "text",
        "model_text",
        "intent",
        "group",
        "reference",
        "reference_labels",
        "status",
    ]

    matches_df = pd.DataFrame(
        matches,
        columns=match_columns
    )

    overlap_count = len(matches_df)

    overlap_percentage = (
        overlap_count / len(final_test) * 100
    )

    print(f"Reference records: {len(reference):,}")

    print(
        f"Unique reference model texts: "
        f"{len(reference_lookup):,}"
    )

    print(
        f"Overlapping final test records: "
        f"{overlap_count}"
    )

    print(
        f"Overlap percentage: "
        f"{overlap_percentage:.2f}%"
    )

    print(
        f"Non-overlapping records: "
        f"{len(final_test) - overlap_count}"
    )

    statuses = [

        "SAME_LABEL",
        "AMBIGUOUS_LABELS",
        "CONFLICTING_LABEL",
    ]

    status_counts = (
        matches_df["status"]
        .value_counts()
        .reindex(statuses, fill_value=0)
    )

    print("\nOverlap categories:")

    print(status_counts.to_string())

    if not matches_df.empty:

        print("\nMatching utterances:")

        print(
            matches_df[
                [
                    "id",
                    "text",
                    "intent",
                    "reference_labels",
                    "status",
                ]
            ].to_string(index=False)
        )

    summary = {

        "reference": reference_name,

        "reference_records": len(reference),

        "unique_reference_texts":
            len(reference_lookup),

        "overlapping_records":
            overlap_count,

        "overlap_percentage":
            overlap_percentage,

        "same_label":
            int(status_counts["SAME_LABEL"]),

        "ambiguous_labels":
            int(status_counts["AMBIGUOUS_LABELS"]),

        "conflicting_label":
            int(status_counts["CONFLICTING_LABEL"]),
    }

    return matches_df, summary


def analyze_total_overlap(
    final_test,
    all_matches
):

    print("\n=== TOTAL OVERLAP ANALYSIS ===")

    overlapping_ids = set(
        all_matches["id"]
    )

    overlap_mask = final_test["id"].isin(
        overlapping_ids
    )

    overlapping = final_test.loc[
        overlap_mask
    ].copy()

    non_overlapping = final_test.loc[
        ~overlap_mask
    ].copy()

    print(f"Original records: {len(final_test)}")

    print(
        f"Unique overlapping records: "
        f"{len(overlapping)}"
    )

    print(
        f"Non-overlapping records: "
        f"{len(non_overlapping)}"
    )

    print(
        f"Total overlap percentage: "
        f"{len(overlapping) / len(final_test) * 100:.2f}%"
    )

    print("\n=== OVERLAP BY INTENT ===")

    intent_summary = pd.DataFrame({

        "original":
            final_test[TARGET_COLUMN]
            .value_counts()
            .reindex(INTENT_CLASSES, fill_value=0),

        "overlapping":
            overlapping[TARGET_COLUMN]
            .value_counts()
            .reindex(INTENT_CLASSES, fill_value=0),

        "non_overlapping":
            non_overlapping[TARGET_COLUMN]
            .value_counts()
            .reindex(INTENT_CLASSES, fill_value=0),
    })

    print(intent_summary.to_string())

    print("\n=== OVERLAP BY GROUP ===")

    group_summary = pd.DataFrame({

        "original":
            final_test["group"]
            .value_counts()
            .reindex(EXPECTED_GROUPS, fill_value=0),

        "overlapping":
            overlapping["group"]
            .value_counts()
            .reindex(EXPECTED_GROUPS, fill_value=0),

        "non_overlapping":
            non_overlapping["group"]
            .value_counts()
            .reindex(EXPECTED_GROUPS, fill_value=0),
    })

    print(group_summary.to_string())

    print("\n=== OVERLAP BY INTENT AND GROUP ===")

    group_class_summary = pd.crosstab(
        overlapping[TARGET_COLUMN],
        overlapping["group"]
    ).reindex(
        index=INTENT_CLASSES,
        columns=EXPECTED_GROUPS,
        fill_value=0
    )

    print(group_class_summary.to_string())

    if not overlapping.empty:

        print("\n=== UNIQUE OVERLAPPING UTTERANCES ===")

        print(
            overlapping[
                ["id", "text", "intent", "group"]
            ].to_string(index=False)
        )

        print("\n=== MATCHES ACROSS REFERENCES ===")

        reference_counts = (
            all_matches.groupby("id")["reference"]
            .nunique()
            .sort_values(ascending=False)
        )

        print(reference_counts.to_string())

    return (
        overlapping,
        non_overlapping,
        intent_summary,
        group_summary,
        group_class_summary,
    )


def main():

    print(
        "=== E09.06.2d: FINAL INSTALLATION "
        "DATASET OVERLAP ANALYSIS ==="
    )

    EXPERIMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    final_test = load_dataset(
        FINAL_TEST_PATH
    )

    references = load_reference_datasets()

    validate_datasets(
        final_test,
        references
    )

    all_matches = []

    summaries = []

    for reference_name, reference in references.items():

        matches, summary = analyze_reference(

            final_test=final_test,

            reference=reference,

            reference_name=reference_name,
        )

        all_matches.append(matches)

        summaries.append(summary)

    all_matches_df = pd.concat(
        all_matches,
        ignore_index=True
    )

    summary_df = pd.DataFrame(
        summaries
    )

    print("\n=== REFERENCE OVERLAP SUMMARY ===")

    print(
        summary_df.round(4)
        .to_string(index=False)
    )

    (
        overlapping,
        non_overlapping,
        intent_summary,
        group_summary,
        group_class_summary,
    ) = analyze_total_overlap(
        final_test,
        all_matches_df
    )

    # Save analysis reports only

    all_matches_df.to_csv(
        EXPERIMENTS_DIR
        / f"{EXPERIMENT}_overlap_details.csv",
        index=False
    )

    summary_df.to_csv(
        EXPERIMENTS_DIR
        / f"{EXPERIMENT}_overlap_summary.csv",
        index=False
    )

    intent_summary.to_csv(
        EXPERIMENTS_DIR
        / f"{EXPERIMENT}_intent_summary.csv"
    )

    group_summary.to_csv(
        EXPERIMENTS_DIR
        / f"{EXPERIMENT}_group_summary.csv"
    )

    group_class_summary.to_csv(
        EXPERIMENTS_DIR
        / f"{EXPERIMENT}_intent_group_summary.csv"
    )

    print("\n=== FINAL DATASET STATUS ===")

    if overlapping.empty:

        print("Exact text overlap: NONE")

        print(
            "Dataset eligible for final evaluation: YES"
        )

    else:

        print(
            f"Exact text overlap: "
            f"{len(overlapping)} records"
        )

        print(
            "Dataset eligible for final evaluation: "
            "REVIEW REQUIRED"
        )

    print("\nOriginal dataset modified: NO")
    print("Processed dataset modified: NO")
    print("Records automatically removed: NO")
    print("Model loaded: NO")
    print("Model training performed: NO")

    print("\n=== OVERLAP ANALYSIS COMPLETE ===")


if __name__ == "__main__":
    main()
