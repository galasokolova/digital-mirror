
from pathlib import Path

import pandas as pd


ML_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ML_ROOT / "data" / "processed"

SWITCHBOARD_DIR = PROCESSED_DIR / "intent_v1"

DIGITAL_MIRROR_PATH = (
    PROCESSED_DIR
    / "digital_mirror_v1"
    / "train.csv"
)

INSTALLATION_TEST_PATH = (
    ML_ROOT
    / "experiments"
    / "E09_03_installation_predictions.csv"
)

EXPERIMENTS_DIR = ML_ROOT / "experiments"

TEXT_COLUMN = "model_text"
TARGET_COLUMN = "intent"


def load_dataset(path, columns=None):

    return pd.read_csv(
        path,
        usecols=columns,
        dtype=str,
        keep_default_na=False
    )


def analyze_training_overlap(domain, train):

    print("\n=== SWITCHBOARD TRAIN OVERLAP ===")

    # Collect all labels associated with each text

    label_map = (
        train.groupby(TEXT_COLUMN)[TARGET_COLUMN]
        .agg(lambda labels: set(labels))
        .to_dict()
    )

    overlap_records = []

    for _, row in domain.iterrows():

        text = row[TEXT_COLUMN]
        intent = row[TARGET_COLUMN]

        existing_labels = label_map.get(text)

        if existing_labels is None:
            continue

        if existing_labels == {intent}:

            status = "SAME_LABEL"

        elif intent in existing_labels:

            status = "AMBIGUOUS_LABELS"

        else:

            status = "CONFLICTING_LABEL"

        overlap_records.append({
            "id": row["id"],
            "text": row["text"],
            "model_text": text,
            "domain_intent": intent,
            "switchboard_intents": ", ".join(
                sorted(existing_labels)
            ),
            "status": status,
        })

    columns = [
        "id",
        "text",
        "model_text",
        "domain_intent",
        "switchboard_intents",
        "status",
    ]

    results = pd.DataFrame(
        overlap_records,
        columns=columns
    )

    print(f"Overlapping records: {len(results)}")

    print(
        f"New unique texts: "
        f"{len(domain) - len(results)}"
    )

    print(
        f"Overlap percentage: "
        f"{len(results) / len(domain) * 100:.2f}%"
    )

    print("\nOverlap categories:")

    print(
        results["status"]
        .value_counts()
        .reindex(
            [
                "SAME_LABEL",
                "AMBIGUOUS_LABELS",
                "CONFLICTING_LABEL",
            ],
            fill_value=0
        )
        .to_string()
    )

    if not results.empty:

        print("\nOverlapping training records:")

        print(
            results.to_string(index=False)
        )

    return results


def analyze_evaluation_overlap(
    domain,
    evaluation_texts,
    name
):

    print(f"\n=== {name} OVERLAP ===")

    mask = domain[TEXT_COLUMN].isin(
        evaluation_texts
    )

    count = int(mask.sum())

    print(f"Overlapping records: {count}")

    print(
        f"Overlap percentage: "
        f"{count / len(domain) * 100:.2f}%"
    )

    return mask


def main():

    print("=== E09.04.4: DATASET OVERLAP ANALYSIS ===")

    EXPERIMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    domain = load_dataset(
        DIGITAL_MIRROR_PATH
    )

    train = load_dataset(
        SWITCHBOARD_DIR / "train.csv",
        [TEXT_COLUMN, TARGET_COLUMN]
    )

    validation = load_dataset(
        SWITCHBOARD_DIR / "validation.csv",
        [TEXT_COLUMN]
    )

    # Read only texts, not test labels

    test = load_dataset(
        SWITCHBOARD_DIR / "test.csv",
        [TEXT_COLUMN]
    )

    installation = load_dataset(
        INSTALLATION_TEST_PATH,
        ["text", "expected"]
    )

    # Use the existing text normalizer

    from preprocessing.intent_text_normalizer import (
        remove_punctuation
    )

    installation[TEXT_COLUMN] = (
        installation["text"]
        .apply(remove_punctuation)
    )

    installation = installation.drop_duplicates(
        subset=[TEXT_COLUMN]
    )

    print("\n=== DATASET OVERVIEW ===")

    print(f"Digital Mirror: {len(domain):,}")
    print(f"Switchboard train: {len(train):,}")
    print(f"Switchboard validation: {len(validation):,}")
    print(f"Switchboard test: {len(test):,}")
    print(f"Installation test: {len(installation):,}")

    if len(installation) != 60:

        raise ValueError(
            "Unexpected installation test size."
        )

    # Analyze training overlap and label conflicts

    train_overlap = analyze_training_overlap(
        domain,
        train
    )

    # Analyze overlap with evaluation datasets

    validation_mask = analyze_evaluation_overlap(
        domain,
        set(validation[TEXT_COLUMN]),
        "SWITCHBOARD VALIDATION"
    )

    test_mask = analyze_evaluation_overlap(
        domain,
        set(test[TEXT_COLUMN]),
        "SWITCHBOARD TEST"
    )

    installation_mask = analyze_evaluation_overlap(
        domain,
        set(installation[TEXT_COLUMN]),
        "INSTALLATION TEST E09.03"
    )

    # Identify records requiring exclusion

    evaluation_overlap_mask = (
        validation_mask
        | test_mask
        | installation_mask
    )

    eligible_mask = ~evaluation_overlap_mask

    # Save training overlap details

    train_overlap.to_csv(
        EXPERIMENTS_DIR
        / "E09_04_04_training_overlap.csv",
        index=False
    )

    # Report only aggregate evaluation overlap

    summary = pd.DataFrame([
        {
            "dataset": "Switchboard Train",
            "overlapping_records": len(train_overlap),
        },
        {
            "dataset": "Switchboard Validation",
            "overlapping_records": int(
                validation_mask.sum()
            ),
        },
        {
            "dataset": "Switchboard Test",
            "overlapping_records": int(
                test_mask.sum()
            ),
        },
        {
            "dataset": "Installation Test E09.03",
            "overlapping_records": int(
                installation_mask.sum()
            ),
        },
    ])

    summary["overlap_percentage"] = (
        summary["overlapping_records"]
        / len(domain) * 100
    ).round(2)

    summary.to_csv(
        EXPERIMENTS_DIR
        / "E09_04_04_overlap_summary.csv",
        index=False
    )

    print("\n=== OVERLAP SUMMARY ===")

    print(summary.to_string(index=False))

    print("\n=== TRAINING ELIGIBILITY ===")

    print(f"Original domain records: {len(domain)}")

    print(
        "Records overlapping evaluation datasets:",
        int(evaluation_overlap_mask.sum())
    )

    print(
        "Eligible domain records:",
        int(eligible_mask.sum())
    )

    print("\n=== ELIGIBLE CLASS DISTRIBUTION ===")

    distribution = pd.DataFrame({
        "original": domain[TARGET_COLUMN].value_counts(),
        "eligible": domain.loc[
            eligible_mask,
            TARGET_COLUMN
        ].value_counts(),
    }).fillna(0).astype(int)

    distribution["excluded"] = (
        distribution["original"]
        - distribution["eligible"]
    )

    print(distribution.to_string())
    
        # Save eligible domain training records

    eligible_domain = domain.loc[
        eligible_mask
    ].copy()

    output_path = (
        DIGITAL_MIRROR_PATH.parent
        / "train_eligible.csv"
    )

    assert len(eligible_domain) == int(
        eligible_mask.sum()
    )

    assert eligible_domain["id"].is_unique

    assert eligible_domain[TEXT_COLUMN].is_unique

    evaluation_texts = (
        set(validation[TEXT_COLUMN])
        | set(test[TEXT_COLUMN])
        | set(installation[TEXT_COLUMN])
    )

    assert not eligible_domain[TEXT_COLUMN].isin(
        evaluation_texts
    ).any()

    eligible_domain.to_csv(
        output_path,
        index=False,
        encoding="utf-8"
    )

    print("\n=== ELIGIBLE DATASET SAVED ===")

    print(f"Records: {len(eligible_domain)}")
    print(f"Saved to: {output_path}")

    print("\n=== OVERLAP ANALYSIS COMPLETE ===")


if __name__ == "__main__":
    main()
