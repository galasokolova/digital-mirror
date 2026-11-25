
import json
from pathlib import Path
from collections import Counter

import pandas as pd

from taxonomy.intent_mapping import INTENT_CLASSES


ML_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    ML_ROOT
    / "data"
    / "raw"
    / "digital_mirror_contrastive"
)

BATCH_DIR = DATA_DIR / "batches"

OUTPUT_PATH = (
    DATA_DIR / "contrastive_intents_v1.csv"
)

SOURCE = "digital_mirror_contrastive_v1"

EXPECTED_BATCHES = {
    "b01_explicit": "explicit",
    "b02_implicit": "implicit",
    "b03_short": "short",
    "b04_complex": "complex",
    "b05_challenging": "challenging",
}

EXPECTED_PER_CLASS = 20

EXPECTED_TOTAL = 1200

VALID_TOPICS = {
    "news_media",
    "evidence",
    "information_sources",
    "video_audio",
    "artificial_intelligence",
    "technology",
    "science",
    "education",
    "everyday_conversation",
    "personal_decisions",
    "art_exhibitions",
    "human_machine_interaction",
}

REQUIRED_FIELDS = {
    "text",
    "intent",
    "context",
}


def normalize_for_check(text):

    return " ".join(
        text.casefold().split()
    )


def load_batch(path):

    print(f"\nLoading: {path.name}")

    with path.open(
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    if not isinstance(data, list):

        raise ValueError(
            f"Expected a list: {path.name}"
        )

    return data


def validate_group(group, path):

    required = {
        "group_id",
        "topic",
        "examples",
    }

    if not required.issubset(group):

        raise ValueError(
            f"Missing group fields: {path.name}"
        )

    group_id = group["group_id"]

    if (
        not isinstance(group_id, str)
        or not group_id.strip()
    ):

        raise ValueError(
            "Invalid group ID."
        )

    if group["topic"] not in VALID_TOPICS:

        raise ValueError(
            f"Invalid topic: {group['topic']}"
        )

    examples = group["examples"]

    if (
        not isinstance(examples, list)
        or len(examples) < 2
    ):

        raise ValueError(
            f"Insufficient examples: {group_id}"
        )

    intents = set()

    for example in examples:

        if not REQUIRED_FIELDS.issubset(
            example
        ):

            raise ValueError(
                f"Missing fields: {group_id}"
            )

        text = example["text"]
        intent = example["intent"]
        context = example["context"]

        if (
            not isinstance(text, str)
            or not text.strip()
        ):

            raise ValueError(
                f"Empty text: {group_id}"
            )

        if not isinstance(context, str):

            raise ValueError(
                f"Invalid context: {group_id}"
            )

        if intent not in INTENT_CLASSES:

            raise ValueError(
                f"Unknown intent: {intent}"
            )

        intents.add(intent)

    if len(intents) < 2:

        raise ValueError(
            f"No intent contrast: {group_id}"
        )


def generate_records():

    records = []

    seen_groups = set()
    seen_texts = {}

    loaded_batches = []

    for batch_name, style in (
        EXPECTED_BATCHES.items()
    ):

        path = BATCH_DIR / f"{batch_name}.json"

        if not path.exists():

            print(f"SKIP | Missing: {path.name}")
            continue

        groups = load_batch(path)

        batch_records = []

        for group in groups:

            validate_group(group, path)

            group_id = group["group_id"]

            if group_id in seen_groups:

                raise ValueError(
                    f"Duplicate group: {group_id}"
                )

            seen_groups.add(group_id)

            for example in group["examples"]:

                text = example["text"].strip()
                intent = example["intent"]

                normalized = normalize_for_check(
                    text
                )

                if normalized in seen_texts:

                    previous = seen_texts[
                        normalized
                    ]

                    if previous != intent:

                        raise ValueError(
                            f"Conflicting labels: {text}"
                        )

                    raise ValueError(
                        f"Duplicate text: {text}"
                    )

                seen_texts[normalized] = intent

                record = {
                    "text": text,
                    "intent": intent,
                    "contrast_group": group_id,
                    "topic": group["topic"],
                    "style": style,
                    "context": example[
                        "context"
                    ].strip(),
                    "source": SOURCE,
                }

                batch_records.append(record)

        distribution = Counter(
            record["intent"]
            for record in batch_records
        )

        print("\nBatch distribution:")

        for intent in INTENT_CLASSES:

            print(
                f"{intent:25s} "
                f"{distribution[intent]:4d}"
            )

        balanced = all(
            distribution[intent]
            == EXPECTED_PER_CLASS
            for intent in INTENT_CLASSES
        )

        print(
            f"Batch records: {len(batch_records)}"
        )

        print(
            f"Target distribution reached: "
            f"{'YES' if balanced else 'NO'}"
        )

        records.extend(batch_records)
        loaded_batches.append(batch_name)

    if not records:

        raise ValueError(
            "No records were generated."
        )

    for index, record in enumerate(
        records,
        start=1
    ):

        record["id"] = f"DMC-{index:04d}"

    return records, loaded_batches


def print_summary(dataset, loaded_batches):

    print("\n=== DATASET SUMMARY ===")

    print(f"Records: {len(dataset)}")

    print(
        f"Loaded batches: "
        f"{len(loaded_batches)}/5"
    )

    print(
        f"Contrastive groups: "
        f"{dataset['contrast_group'].nunique()}"
    )

    print(
        f"Topics: "
        f"{dataset['topic'].nunique()}"
    )

    print("\n=== INTENT DISTRIBUTION ===")

    distribution = (
        dataset["intent"]
        .value_counts()
        .reindex(INTENT_CLASSES, fill_value=0)
    )

    print(distribution.to_string())

    print("\n=== STYLE DISTRIBUTION ===")

    print(
        dataset["style"]
        .value_counts()
        .to_string()
    )

    print("\n=== TOPIC DISTRIBUTION ===")

    print(
        dataset["topic"]
        .value_counts()
        .to_string()
    )

    print("\n=== INTENT / STYLE MATRIX ===")

    matrix = pd.crosstab(
        dataset["intent"],
        dataset["style"]
    ).reindex(
        index=INTENT_CLASSES,
        columns=list(EXPECTED_BATCHES.values()),
        fill_value=0
    )

    print(matrix.to_string())

    lengths = (
        dataset["text"]
        .str.split()
        .str.len()
    )

    print("\n=== TEXT LENGTH ===")

    print(lengths.describe().to_string())

    complete = (
        len(loaded_batches) == 5
        and len(dataset) == EXPECTED_TOTAL
        and matrix.eq(EXPECTED_PER_CLASS)
        .all()
        .all()
    )

    print("\n=== GENERATION STATUS ===")

    print(
        f"Target records: {EXPECTED_TOTAL}"
    )

    print(
        f"Generated records: {len(dataset)}"
    )

    print(
        f"Progress: "
        f"{len(dataset) / EXPECTED_TOTAL:.2%}"
    )

    print(
        "Status: "
        + ("COMPLETE" if complete else "DRAFT")
    )

    return complete


def main():

    print(
        "=== E09.07.6.2: "
        "CONTRASTIVE DATASET GENERATION ==="
    )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    records, loaded_batches = (
        generate_records()
    )

    dataset = pd.DataFrame(records)

    columns = [
        "id",
        "text",
        "intent",
        "contrast_group",
        "topic",
        "style",
        "context",
        "source",
    ]

    dataset = dataset[columns]

    complete = print_summary(
        dataset,
        loaded_batches
    )

    output_path = (
        OUTPUT_PATH
        if complete
        else DATA_DIR / "contrastive_intents_draft.csv"
    )

    dataset.to_csv(
        output_path,
        index=False,
        encoding="utf-8"
    )

    print(f"\nSaved: {output_path}")

    print("Model loaded: NO")
    print("Model trained: NO")
    print("Evaluation data loaded: NO")

    print("\n=== GENERATION FINISHED ===")


if __name__ == "__main__":
    main()
