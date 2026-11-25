
import hashlib
from pathlib import Path

import pandas as pd

from preprocessing.intent_text_normalizer import (
    normalize_text,
    remove_punctuation,
)

from taxonomy.intent_mapping import INTENT_CLASSES


ML_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ML_ROOT / "data"

EXPERIMENTS_DIR = ML_ROOT / "experiments"

EXPERIMENT = "E09_07_06_04_02"

CONTRASTIVE_DIR = (
    DATA_DIR
    / "processed"
    / "digital_mirror_contrastive_v1"
)

INPUT_PATH = CONTRASTIVE_DIR / "train.csv"

OUTPUT_PATH = (
    CONTRASTIVE_DIR / "train_eligible.csv"
)

SWITCHBOARD_DIR = (
    DATA_DIR / "processed" / "intent_v1"
)

DOMAIN_DIR = (
    DATA_DIR
    / "processed"
    / "digital_mirror_v1"
)

REFERENCES = {
    "switchboard_train": {
        "path": SWITCHBOARD_DIR / "train.csv",
        "expected": 135827,
        "role": "training",
    },

    "switchboard_validation": {
        "path": SWITCHBOARD_DIR / "validation.csv",
        "expected": 16913,
        "role": "evaluation",
    },

    "switchboard_test": {
        "path": SWITCHBOARD_DIR / "test.csv",
        "expected": 17966,
        "role": "evaluation",
    },

    "digital_mirror_original": {
        "path": DOMAIN_DIR / "train.csv",
        "expected": 600,
        "role": "diagnostic",
    },

    "digital_mirror_eligible": {
        "path": DOMAIN_DIR / "train_eligible.csv",
        "expected": 579,
        "role": "training",
    },

    "installation_development": {
        "path": (
            EXPERIMENTS_DIR
            / "E09_03_installation_predictions.csv"
        ),
        "expected": 60,
        "role": "evaluation",
    },

    "installation_benchmark": {
        "path": (
            DATA_DIR
            / "evaluation"
            / "digital_mirror_final"
            / "test_processed.csv"
        ),
        "expected": 240,
        "role": "evaluation",
    },
}


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


def load_reference(name, config):

    dataset = pd.read_csv(
        config["path"],
        dtype=str,
        keep_default_na=False
    )

    if name == "installation_development":

        dataset = dataset.rename(
            columns={"expected": "intent"}
        )

        dataset = dataset[
            ["text", "intent"]
        ].drop_duplicates()

    required = {"text", "intent"}

    if not required.issubset(dataset.columns):

        raise ValueError(
            f"Missing columns in {name}"
        )

    if name.startswith("switchboard_"):

        calculated_model_text = (
            dataset["normalized_text"]
            .map(remove_punctuation)
        )
    else:
        calculated_model_text = (
            dataset["text"]
            .map(normalize_text)
            .map(remove_punctuation)
        )

    if "model_text" in dataset.columns:

        if not dataset["model_text"].equals(
            calculated_model_text
        ):

            raise ValueError(
                f"Normalization mismatch: {name}"
            )

    else:

        dataset["model_text"] = (
            calculated_model_text
        )

    if len(dataset) != config["expected"]:

        raise ValueError(
            f"Unexpected size for {name}: "
            f"{len(dataset)}"
        )

    if not set(dataset["intent"]).issubset(
        set(INTENT_CLASSES)
    ):

        raise ValueError(
            f"Unexpected intent labels: {name}"
        )

    print(
        f"{name}: {len(dataset)} records | "
        f"{config['role']}"
    )

    return dataset


def load_datasets():

    print("\n=== DATASET LOADING ===")

    contrastive = pd.read_csv(
        INPUT_PATH,
        dtype=str,
        keep_default_na=False
    )

    if len(contrastive) != 1200:

        raise ValueError(
            "Expected 1200 contrastive records."
        )

    if not contrastive["id"].is_unique:

        raise ValueError("Duplicate record IDs.")

    if not contrastive["model_text"].str.strip().ne("").all():

        raise ValueError("Empty model texts.")

    expected_model_text = (
        contrastive["text"]
        .map(normalize_text)
        .map(remove_punctuation)
    )

    if not contrastive["model_text"].equals(
        expected_model_text
    ):

        raise ValueError(
            "Contrastive normalization mismatch."
        )

    print(
        f"Contrastive dataset: {len(contrastive)}"
    )

    references = {
        name: load_reference(name, config)
        for name, config in REFERENCES.items()
    }

    return contrastive, references


def analyze_internal_overlap(dataset):

    print("\n=== INTERNAL OVERLAP ===")

    exclusions = []

    duplicate_records = dataset[
        dataset["model_text"].duplicated(
            keep=False
        )
    ]

    for model_text, group in duplicate_records.groupby(
        "model_text"
    ):

        if group["intent"].nunique() > 1:

            excluded = group

            reason = "internal_label_conflict"

        else:

            excluded = group.iloc[1:]

            reason = "internal_duplicate"

        for record_id in excluded["id"]:

            exclusions.append({
                "id": record_id,
                "reason": reason,
            })

    print(
        f"Records involved in duplicates: "
        f"{len(duplicate_records)}"
    )

    print(
        f"Internal exclusions: {len(exclusions)}"
    )

    save_table(
        duplicate_records,
        "internal_duplicates"
    )

    return exclusions


def analyze_external_overlap(
    contrastive,
    references
):

    print("\n=== EXTERNAL OVERLAP ===")

    summaries = []
    matches = []
    exclusions = []

    for name, reference in references.items():

        role = REFERENCES[name]["role"]

        lookup = (
            reference.groupby("model_text")["intent"]
            .agg(lambda values: set(values))
            .to_dict()
        )

        same = 0
        conflicting = 0
        ambiguous = 0

        for row in contrastive.itertuples(
            index=False
        ):

            labels = lookup.get(row.model_text)

            if labels is None:

                continue

            if labels == {row.intent}:

                relation = "SAME_LABEL"

                same += 1

            elif row.intent in labels:

                relation = "AMBIGUOUS"

                ambiguous += 1

            else:

                relation = "CONFLICTING_LABEL"

                conflicting += 1

            reason = ""

            if role == "evaluation":

                reason = (
                    f"evaluation_overlap:{name}"
                )

            elif role == "training":

                if relation != "SAME_LABEL":

                    reason = (
                        f"training_label_conflict:{name}"
                    )

                elif name == "digital_mirror_eligible":

                    reason = (
                        "existing_domain_duplicate"
                    )

            matches.append({
                "id": row.id,
                "text": row.text,
                "model_text": row.model_text,
                "intent": row.intent,
                "reference": name,
                "reference_role": role,
                "reference_labels": "|".join(
                    sorted(labels)
                ),
                "relation": relation,
                "exclusion_reason": reason,
            })

            if reason:

                exclusions.append({
                    "id": row.id,
                    "reason": reason,
                })

        total = same + conflicting + ambiguous

        summaries.append({
            "reference": name,
            "role": role,
            "reference_records": len(reference),
            "overlapping_records": total,
            "same_label": same,
            "conflicting_label": conflicting,
            "ambiguous_label": ambiguous,
            "overlap_rate": (
                total / len(contrastive)
            ),
        })

    summary = pd.DataFrame(summaries)

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            )
        )
    )

    match_columns = [
        "id",
        "text",
        "model_text",
        "intent",
        "reference",
        "reference_role",
        "reference_labels",
        "relation",
        "exclusion_reason",
    ]

    match_table = pd.DataFrame(
        matches,
        columns=match_columns
    )

    save_table(summary, "overlap_summary")

    save_table(
        match_table,
        "overlap_details"
    )

    return exclusions


def prepare_eligible_dataset(
    dataset,
    exclusions
):

    print("\n=== TRAINING DATASET PREPARATION ===")

    exclusion_table = pd.DataFrame(
        exclusions,
        columns=["id", "reason"]
    ).drop_duplicates()

    reasons = (
        exclusion_table.groupby("id")["reason"]
        .agg(lambda values: "|".join(sorted(values)))
        .to_dict()
    )

    excluded = dataset[
        dataset["id"].isin(reasons)
    ].copy()

    excluded["exclusion_reason"] = (
        excluded["id"].map(reasons)
    )

    eligible = dataset[
        ~dataset["id"].isin(reasons)
    ].copy()

    print(f"Original records: {len(dataset)}")

    print(f"Excluded records: {len(excluded)}")

    print(f"Eligible records: {len(eligible)}")

    print(
        f"Retention rate: "
        f"{len(eligible) / len(dataset):.2%}"
    )

    print("\n=== EXCLUSION REASONS ===")

    print(
        exclusion_table["reason"]
        .value_counts()
        .to_string()
    )

    print(
        "A record may have multiple exclusion reasons."
    )

    save_table(
        exclusion_table,
        "exclusion_reasons"
    )

    save_table(
        excluded,
        "excluded_records"
    )

    return eligible, excluded


def analyze_eligible_distribution(
    original,
    eligible
):

    print("\n=== CLASS DISTRIBUTION ===")

    before = (
        original["intent"]
        .value_counts()
        .reindex(INTENT_CLASSES, fill_value=0)
    )

    after = (
        eligible["intent"]
        .value_counts()
        .reindex(INTENT_CLASSES, fill_value=0)
    )

    distribution = pd.DataFrame({
        "intent": INTENT_CLASSES,
        "before": before.to_numpy(),
        "after": after.to_numpy(),
        "excluded": (before - after).to_numpy(),
    })

    print(
        distribution.to_string(index=False)
    )

    save_table(
        distribution,
        "eligible_class_distribution"
    )

    print("\n=== ELIGIBLE INTENT / STYLE MATRIX ===")

    matrix = pd.crosstab(
        eligible["intent"],
        eligible["style"]
    ).reindex(
        index=INTENT_CLASSES,
        columns=[
            "explicit",
            "implicit",
            "short",
            "complex",
            "challenging",
        ],
        fill_value=0
    )

    print(matrix.to_string())

    save_table(
        matrix.reset_index(),
        "eligible_intent_style_matrix"
    )

    group_stats = (
        original.groupby("contrast_group")
        .size()
        .rename("before")
        .to_frame()
    )

    group_stats["after"] = (
        eligible.groupby("contrast_group")
        .size()
        .reindex(
            group_stats.index,
            fill_value=0
        )
    )

    group_stats["excluded"] = (
        group_stats["before"]
        - group_stats["after"]
    )

    group_stats["remaining_intents"] = (
        eligible.groupby("contrast_group")["intent"]
        .nunique()
        .reindex(
            group_stats.index,
            fill_value=0
        )
    )

    print("\n=== CONTRASTIVE GROUP INTEGRITY ===")

    print(
        f"Unchanged groups: "
        f"{group_stats['excluded'].eq(0).sum()}"
    )

    print(
        f"Modified groups: "
        f"{group_stats['excluded'].gt(0).sum()}"
    )

    print(
        f"Groups with fewer than two intents: "
        f"{group_stats['remaining_intents'].lt(2).sum()}"
    )

    save_table(
        group_stats.reset_index(),
        "eligible_group_statistics"
    )


def validate_eligible_dataset(
    original,
    eligible,
    excluded,
    references
):

    print("\n=== FINAL ELIGIBILITY VALIDATION ===")

    eligible_ids = set(eligible["id"])

    excluded_ids = set(excluded["id"])

    eligible_texts = set(
        eligible["model_text"]
    )

    checks = {
        "Record accounting":
            len(original)
            == len(eligible) + len(excluded),

        "No eligible/excluded ID overlap":
            not eligible_ids.intersection(
                excluded_ids
            ),

        "All original IDs accounted for":
            eligible_ids.union(excluded_ids)
            == set(original["id"]),

        "Original eligible rows preserved":
            eligible.equals(
                original[
                    original["id"].isin(
                        eligible_ids
                    )
                ]
            ),

        "Unique eligible IDs":
            eligible["id"].is_unique,

        "Unique eligible model texts":
            eligible["model_text"].is_unique,

        "No empty model texts":
            eligible["model_text"]
            .str.strip()
            .ne("")
            .all(),

        "All intent classes retained":
            set(eligible["intent"])
            == set(INTENT_CLASSES),

        "No internal conflicting labels":
            eligible.groupby("model_text")["intent"]
            .nunique()
            .le(1)
            .all(),

        "No existing eligible domain duplicates":
            not eligible_texts.intersection(
                references[
                    "digital_mirror_eligible"
                ]["model_text"]
            ),
    }

    for name, reference in references.items():

        role = REFERENCES[name]["role"]

        if role == "evaluation":

            overlap = eligible_texts.intersection(
                reference["model_text"]
            )

            checks[f"No overlap with {name}"] = (
                len(overlap) == 0
            )

        elif role == "training":

            lookup = (
                reference.groupby("model_text")["intent"]
                .agg(lambda values: set(values))
                .to_dict()
            )

            conflicts = [
                row.id
                for row in eligible.itertuples(
                    index=False
                )
                if row.model_text in lookup
                and lookup[row.model_text]
                != {row.intent}
            ]

            checks[
                f"No label conflicts with {name}"
            ] = len(conflicts) == 0

    for name, passed in checks.items():

        status = "PASS" if passed else "FAIL"

        print(f"{status} | {name}")

    print(
        f"\nChecks passed: "
        f"{sum(checks.values())} / {len(checks)}"
    )

    save_table(
        pd.DataFrame([
            {
                "check": name,
                "passed": bool(passed),
            }
            for name, passed in checks.items()
        ]),
        "eligibility_checks"
    )

    if not all(checks.values()):

        raise ValueError(
            "Final eligibility validation failed."
        )


def save_eligible_dataset(dataset):

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

    if not reloaded.equals(
        dataset.reset_index(drop=True)
    ):

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
        "=== E09.07.6.4.2-4: "
        "CONTRASTIVE OVERLAP ANALYSIS ==="
    )

    protected_paths = [
        INPUT_PATH,
        *[
            config["path"]
            for config in REFERENCES.values()
        ],
    ]

    original_hashes = {
        path: calculate_sha256(path)
        for path in protected_paths
    }

    try:

        dataset, references = load_datasets()

        internal_exclusions = (
            analyze_internal_overlap(dataset)
        )

        external_exclusions = (
            analyze_external_overlap(
                dataset,
                references
            )
        )

        eligible, excluded = (
            prepare_eligible_dataset(
                dataset,
                internal_exclusions
                + external_exclusions
            )
        )

        analyze_eligible_distribution(
            dataset,
            eligible
        )

        validate_eligible_dataset(
            dataset,
            eligible,
            excluded,
            references
        )

        save_eligible_dataset(eligible)

        print("\n=== FINAL SUMMARY ===")

        print("Overlap analysis: COMPLETE")

        print(
            f"Original records: {len(dataset)}"
        )

        print(
            f"Excluded records: {len(excluded)}"
        )

        print(
            f"Eligible records: {len(eligible)}"
        )

        print(
            "Exact evaluation text overlap: NONE"
        )

        print(
            "Eligible for controlled training: YES"
        )

        print("Model loaded: NO")

        print("Model trained: NO")

        print(
            "Evaluation data loaded: "
            "YES (overlap analysis only)"
        )

        print("Model evaluation performed: NO")

    finally:

        print("\n=== FILE INTEGRITY ===")

        integrity_results = []

        for path, original_hash in original_hashes.items():

            current_hash = calculate_sha256(path)

            unchanged = (
                original_hash == current_hash
            )

            integrity_results.append(unchanged)

            status = (
                "PASS" if unchanged else "FAIL"
            )

            print(
                f"{status} | "
                f"{path.relative_to(ML_ROOT)}"
            )

        print(
            f"\nIntegrity checks passed: "
            f"{sum(integrity_results)} / "
            f"{len(integrity_results)}"
        )

        if not all(integrity_results):

            raise RuntimeError(
                "Protected files were modified."
            )

    print("\n=== OVERLAP ANALYSIS FINISHED ===")


if __name__ == "__main__":
    main()
