
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from taxonomy.intent_mapping import INTENT_CLASSES


ML_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ML_ROOT
    / "data"
    / "raw"
    / "digital_mirror_contrastive"
    / "contrastive_intents_v1.csv"
)

EXPERIMENTS_DIR = ML_ROOT / "experiments"

EXPERIMENT = "E09_07_06_03"

EXPECTED_RECORDS = 1200
EXPECTED_GROUPS = 300
EXPECTED_PER_CLASS = 100
EXPECTED_PER_STYLE = 240
EXPECTED_PER_COMBINATION = 20

STYLES = [
    "explicit",
    "implicit",
    "short",
    "complex",
    "challenging",
]

TOPICS = [
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
]

REQUIRED_COLUMNS = [
    "id",
    "text",
    "intent",
    "contrast_group",
    "topic",
    "style",
    "context",
    "source",
]

CONTEXT_SENSITIVE_INTENTS = {
    "AGREEMENT",
    "DISAGREEMENT",
    "ACKNOWLEDGMENT",
    "AFFIRMATIVE_ANSWER",
    "NEGATIVE_ANSWER",
    "CLARIFICATION",
}


def calculate_sha256(path):

    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def tokenize(text):

    return re.findall(
        r"\b\w+(?:['’]\w+)*\b",
        text.lower()
    )


def normalize_for_check(text):

    return " ".join(
        tokenize(text.replace("’", "'"))
    )


def save_table(table, name, index=False):

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
        index=index,
        encoding="utf-8"
    )

    print(f"Saved: {path.name}")


def load_dataset():

    dataset = pd.read_csv(
        DATA_PATH,
        dtype=str,
        keep_default_na=False
    )

    missing_columns = (
        set(REQUIRED_COLUMNS)
        - set(dataset.columns)
    )

    if missing_columns:

        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    print("\n=== DATASET LOADING ===")

    print(f"Records: {len(dataset)}")
    print(f"Columns: {len(dataset.columns)}")

    return dataset


# V01-V06: Structural validation

def validate_structure(dataset):

    print("\n=== STRUCTURAL VALIDATION ===")

    required_nonempty = [
        column
        for column in REQUIRED_COLUMNS
        if column != "context"
    ]

    class_counts = (
        dataset["intent"]
        .value_counts()
        .reindex(INTENT_CLASSES, fill_value=0)
    )

    style_counts = (
        dataset["style"]
        .value_counts()
        .reindex(STYLES, fill_value=0)
    )

    matrix = pd.crosstab(
        dataset["intent"],
        dataset["style"]
    ).reindex(
        index=INTENT_CLASSES,
        columns=STYLES,
        fill_value=0
    )

    group_stats = (
        dataset.groupby("contrast_group")
        .agg(
            records=("id", "size"),
            intents=("intent", "nunique"),
            topics=("topic", "nunique"),
            styles=("style", "nunique")
        )
    )

    normalized = dataset["text"].map(
        normalize_for_check
    )

    label_counts = (
        dataset.assign(check_text=normalized)
        .groupby("check_text")["intent"]
        .nunique()
    )

    checks = {
        "Expected columns":
            list(dataset.columns) == REQUIRED_COLUMNS,

        "Expected dataset size":
            len(dataset) == EXPECTED_RECORDS,

        "Unique record IDs":
            dataset["id"].is_unique,

        "Valid record IDs":
            dataset["id"].str.fullmatch(
                r"DMC-\d{4}"
            ).all(),

        "No empty required fields":
            all(
                dataset[column]
                .str.strip()
                .ne("")
                .all()
                for column in required_nonempty
            ),

        "All intent classes present":
            set(dataset["intent"])
            == set(INTENT_CLASSES),

        "Balanced intent classes":
            class_counts.eq(
                EXPECTED_PER_CLASS
            ).all(),

        "All styles present":
            set(dataset["style"]) == set(STYLES),

        "Balanced styles":
            style_counts.eq(
                EXPECTED_PER_STYLE
            ).all(),

        "Balanced intent-style combinations":
            matrix.eq(
                EXPECTED_PER_COMBINATION
            ).all().all(),

        "All topics present":
            set(dataset["topic"]) == set(TOPICS),

        "No duplicate raw texts":
            dataset["text"].is_unique,

        "No duplicate normalized texts":
            normalized.is_unique,

        "No empty normalized texts":
            normalized.ne("").all(),

        "No conflicting intent labels":
            label_counts.le(1).all(),

        "Expected contrastive group count":
            len(group_stats) == EXPECTED_GROUPS,

        "At least two intents per group":
            group_stats["intents"].ge(2).all(),

        "Single topic per group":
            group_stats["topics"].eq(1).all(),

        "Single style per group":
            group_stats["styles"].eq(1).all(),

        "Valid source":
            dataset["source"].eq(
                "digital_mirror_contrastive_v1"
            ).all(),
    }

    for name, passed in checks.items():

        status = "PASS" if passed else "FAIL"

        print(f"{status} | {name}")

    print(
        f"\nChecks passed: "
        f"{sum(checks.values())} / {len(checks)}"
    )

    print("\n=== GROUP SIZE DISTRIBUTION ===")

    print(
        group_stats["records"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\n=== INTENTS PER GROUP ===")

    print(
        group_stats["intents"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    save_table(
        group_stats.reset_index(),
        "group_statistics"
    )

    save_table(
        pd.DataFrame([
            {
                "check": name,
                "passed": bool(passed)
            }
            for name, passed in checks.items()
        ]),
        "structural_checks"
    )

    return all(checks.values())


# V07: Lexical diversity

def analyze_lexical_diversity(dataset):

    print("\n=== LEXICAL DIVERSITY ===")

    results = []

    for intent in INTENT_CLASSES:

        subset = dataset[
            dataset["intent"] == intent
        ]

        tokens = [
            token
            for text in subset["text"]
            for token in tokenize(text)
        ]

        vocabulary = set(tokens)

        results.append({
            "intent": intent,
            "records": len(subset),
            "total_tokens": len(tokens),
            "unique_tokens": len(vocabulary),
            "type_token_ratio": (
                len(vocabulary) / len(tokens)
                if tokens else 0
            ),
            "mean_words": (
                subset["text"]
                .map(lambda text: len(tokenize(text)))
                .mean()
            ),
        })

    table = pd.DataFrame(results)

    print(
        table.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    save_table(table, "lexical_diversity")


# V08: Repeated opening patterns

def analyze_templates(dataset):

    print("\n=== REPEATED OPENING PATTERNS ===")

    results = []

    for size in [2, 3, 4]:

        for intent in INTENT_CLASSES:

            subset = dataset[
                dataset["intent"] == intent
            ]

            prefixes = subset["text"].map(
                lambda text: " ".join(
                    tokenize(text)[:size]
                )
                if len(tokenize(text)) >= size
                else ""
            )

            counts = (
                prefixes[prefixes.ne("")]
                .value_counts()
            )

            for prefix, count in counts.items():

                if count < 2:
                    continue

                results.append({
                    "intent": intent,
                    "prefix_size": size,
                    "prefix": prefix,
                    "count": count,
                    "class_share": (
                        count / len(subset)
                    ),
                })

    columns = [
        "intent",
        "prefix_size",
        "prefix",
        "count",
        "class_share",
    ]

    table = pd.DataFrame(
        results,
        columns=columns
    ).sort_values(
        ["count", "prefix_size"],
        ascending=False
    )

    print("\nTop 40 repeated patterns:")

    print(
        table.head(40).to_string(
            index=False
        )
    )

    save_table(table, "repeated_prefixes")

    # Remove words shared by several members
    # of the same contrastive group.
    # The remaining text is a rough template proxy.

    template_records = []

    for group_id, group in dataset.groupby(
        "contrast_group"
    ):

        token_sets = [
            set(tokenize(text))
            for text in group["text"]
        ]

        shared = set.intersection(*token_sets)

        for _, row in group.iterrows():

            signature = " ".join(
                "<shared>" if token in shared
                else token
                for token in tokenize(row["text"])
            )

            template_records.append({
                "id": row["id"],
                "intent": row["intent"],
                "style": row["style"],
                "contrast_group": group_id,
                "text": row["text"],
                "signature": signature,
            })

    signatures = pd.DataFrame(template_records)

    counts = (
        signatures.groupby(
            ["intent", "style", "signature"]
        )
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    repeated = counts[counts["count"] > 1]

    print("\n=== SHARED-WORD TEMPLATE PROXY ===")

    print(
        repeated.head(30).to_string(
            index=False
        )
    )

    save_table(signatures, "template_signatures")
    save_table(repeated, "repeated_signatures")


# V09: Topic-intent dependency

def analyze_topic_dependency(dataset):

    print("\n=== INTENT / TOPIC MATRIX ===")

    matrix = pd.crosstab(
        dataset["intent"],
        dataset["topic"]
    ).reindex(
        index=INTENT_CLASSES,
        columns=TOPICS,
        fill_value=0
    )

    print(matrix.to_string())

    save_table(
        matrix,
        "intent_topic_matrix",
        index=True
    )

    print("\n=== TOPIC CONCENTRATION ===")

    results = []

    for intent, row in matrix.iterrows():

        total = row.sum()

        results.append({
            "intent": intent,
            "topics_present": int(row.gt(0).sum()),
            "dominant_topic": row.idxmax(),
            "dominant_topic_records": row.max(),
            "dominant_topic_share": (
                row.max() / total
                if total else 0
            ),
        })

    table = pd.DataFrame(results)

    print(table.to_string(index=False))

    save_table(table, "topic_concentration")

    # Cramer's V: descriptive association
    # between topic and intent.

    observed = matrix.to_numpy(dtype=float)

    observed = observed[
        observed.sum(axis=1) > 0
    ]

    observed = observed[
        :,
        observed.sum(axis=0) > 0
    ]

    total = observed.sum()

    rows, columns = observed.shape

    if total > 0 and min(rows, columns) > 1:

        expected = np.outer(
            observed.sum(axis=1),
            observed.sum(axis=0)
        ) / total

        chi_square = np.sum(
            (observed - expected) ** 2
            / expected
        )

        cramers_v = np.sqrt(
            chi_square
            / (
                total
                * min(rows - 1, columns - 1)
            )
        )

        print(
            f"\nCramer's V: {cramers_v:.6f}"
        )

        print(
            "Interpretation: descriptive association, "
            "not evidence of causality."
        )


# V10: Style and length

def analyze_styles(dataset):

    print("\n=== STYLE LENGTH STATISTICS ===")

    table = (
        dataset.groupby("style")["word_count"]
        .agg(
            records="size",
            mean="mean",
            median="median",
            minimum="min",
            maximum="max",
            std="std"
        )
        .reindex(STYLES)
    )

    print(table.to_string())

    save_table(
        table,
        "style_statistics",
        index=True
    )

    short = dataset[
        dataset["style"] == "short"
    ]

    long_short = short[
        short["word_count"] > 5
    ]

    complex_records = dataset[
        dataset["style"] == "complex"
    ]

    short_complex = complex_records[
        complex_records["word_count"] < 8
    ]

    print("\n=== STYLE REVIEW FLAGS ===")

    print(
        f"Short expressions above 5 words: "
        f"{len(long_short)} / {len(short)}"
    )

    print(
        f"Complex expressions below 8 words: "
        f"{len(short_complex)} / "
        f"{len(complex_records)}"
    )

    print("These are review flags, not errors.")

    save_table(
        pd.concat(
            [long_short, short_complex],
            ignore_index=True
        ),
        "style_review_flags"
    )


# V11: Context availability

def analyze_context(dataset):

    print("\n=== CONTEXT AVAILABILITY ===")

    results = []

    for intent in INTENT_CLASSES:

        subset = dataset[
            dataset["intent"] == intent
        ]

        present = (
            subset["context"]
            .str.strip()
            .ne("")
            .sum()
        )

        results.append({
            "intent": intent,
            "records": len(subset),
            "with_context": present,
            "without_context": (
                len(subset) - present
            ),
            "context_coverage": (
                present / len(subset)
            ),
        })

    table = pd.DataFrame(results)

    print(table.to_string(index=False))

    save_table(table, "context_statistics")

    missing_context = dataset[
        dataset["intent"].isin(
            CONTEXT_SENSITIVE_INTENTS
        )
        & dataset["context"].str.strip().eq("")
    ]

    print(
        "\nContext-sensitive records "
        "without context: "
        f"{len(missing_context)}"
    )

    save_table(
        missing_context,
        "missing_context_review"
    )


# V12: Semantic review preparation

def prepare_semantic_review(dataset):

    print("\n=== SEMANTIC REVIEW PREPARATION ===")

    # Three records from every intent-style
    # combination: 12 * 5 * 3 = 180.

    samples = []

    for _, group in dataset.groupby(
        ["intent", "style"]
    ):

        samples.append(
            group.sample(
                n=min(3, len(group)),
                random_state=42
            )
        )

    sample = pd.concat(
        samples,
        ignore_index=True
    )

    # Hide the assigned intent during review.

    review = sample[[
        "id",
        "text",
        "context",
        "topic",
        "style",
        "contrast_group",
    ]].copy()

    review["reviewed_intent"] = ""
    review["naturalness"] = ""
    review["context_required"] = ""
    review["review_status"] = ""
    review["notes"] = ""

    review = review.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    save_table(
        review,
        "semantic_review_sample"
    )

    save_table(
        sample[["id", "intent"]],
        "semantic_review_answer_key"
    )

    full_review = dataset[[
        "id",
        "text",
        "context",
        "topic",
        "style",
        "contrast_group",
    ]].copy()

    full_review["reviewed_intent"] = ""
    full_review["naturalness"] = ""
    full_review["context_required"] = ""
    full_review["review_status"] = ""
    full_review["notes"] = ""

    full_review = full_review.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    save_table(
        full_review,
        "semantic_review_full"
    )

    save_table(
        dataset[["id", "intent"]],
        "semantic_review_full_answer_key"
    )

    print(f"Review sample: {len(review)}")
    print(f"Full review: {len(full_review)}")
    print("Semantic correctness verified: NO")


def main():

    print(
        "=== E09.07.6.3: "
        "CONTRASTIVE DATASET VALIDATION ==="
    )

    original_hash = calculate_sha256(DATA_PATH)

    print(f"\nOriginal SHA256: {original_hash}")

    try:

        dataset = load_dataset()

        valid = validate_structure(dataset)

        if not valid:

            raise ValueError(
                "Structural validation failed."
            )

        dataset["word_count"] = (
            dataset["text"]
            .map(lambda text: len(tokenize(text)))
        )

        analyze_lexical_diversity(dataset)

        analyze_templates(dataset)

        analyze_topic_dependency(dataset)

        analyze_styles(dataset)

        analyze_context(dataset)

        prepare_semantic_review(dataset)

        print("\n=== VALIDATION SUMMARY ===")

        print("Structural validation: PASS")
        print("Linguistic diagnostics: COMPLETE")
        print("Semantic review: PENDING")
        print("Evaluation overlap analysis: PENDING")
        print("Dataset ready for training: NO")
        print("Model loaded: NO")
        print("Model trained: NO")
        print("Evaluation data loaded: NO")

    finally:

        final_hash = calculate_sha256(DATA_PATH)

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

    print("\n=== VALIDATION FINISHED ===")


if __name__ == "__main__":
    main()
