
from pathlib import Path
from time import perf_counter
import hashlib
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


ML_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ML_ROOT / "src"))

from taxonomy.intent_mapping import INTENT_CLASSES

from preprocessing.intent_text_normalizer import (
    remove_punctuation,
)

from analyze_final_installation_overlap import (
    load_reference_datasets,
)


MODELS_DIR = ML_ROOT / "models"

EXPERIMENTS_DIR = ML_ROOT / "experiments"

DATASET_PATH = (
    ML_ROOT
    / "data"
    / "evaluation"
    / "digital_mirror_final"
    / "test_processed.csv"
)

MODEL_PATH = (
    MODELS_DIR
    / "E09_05a_domain_adapted.joblib"
)

EXPERIMENT = "E09_06_02e"

TEXT_COLUMN = "model_text"

TARGET_COLUMN = "intent"

GROUPS = [
    "natural",
    "short",
    "installation",
    "complex",
]


def calculate_sha256(path):

    sha256 = hashlib.sha256()

    with path.open("rb") as file:

        for chunk in iter(
            lambda: file.read(65536),
            b""
        ):

            sha256.update(chunk)

    return sha256.hexdigest()


def load_dataset():

    return pd.read_csv(
        DATASET_PATH,
        dtype=str,
        keep_default_na=False
    )


def validate_dataset(dataset):

    print("\n=== FINAL DATASET VALIDATION ===")

    raw_dataset = pd.read_csv(
        DATASET_PATH.parent / "test.csv",
        dtype=str,
        keep_default_na=False
    )

    original_columns = [
        "id",
        "text",
        "intent",
        "group",
        "source",
    ]

    checks = {

        "Expected record count":
            len(dataset) == 240,

        "Unique IDs":
            dataset["id"].is_unique,

        "Unique model texts":
            dataset[TEXT_COLUMN].is_unique,

        "No empty model texts":
            not dataset[TEXT_COLUMN]
            .eq("").any(),

        "All intent classes present":
            set(dataset[TARGET_COLUMN])
            == set(INTENT_CLASSES),

        "All groups present":
            set(dataset["group"])
            == set(GROUPS),

        "Balanced intent classes":
            dataset[TARGET_COLUMN]
            .value_counts().eq(20).all(),

        "Balanced groups":
            dataset["group"]
            .value_counts().eq(60).all(),

        "Five records per intent and group":
            pd.crosstab(
                dataset[TARGET_COLUMN],
                dataset["group"]
            ).eq(5).all().all(),

        "Raw and processed datasets match":
            raw_dataset[original_columns].equals(
                dataset[original_columns]
            ),

        "Consistent text preprocessing":
            dataset["text"]
            .apply(remove_punctuation)
            .equals(dataset[TEXT_COLUMN]),
    }

    for name, passed in checks.items():

        status = "PASS" if passed else "FAIL"

        print(f"{status} | {name}")

    if not all(checks.values()):

        raise ValueError(
            "Final dataset validation failed."
        )

    print(
        f"\nChecks passed: "
        f"{sum(checks.values())}/{len(checks)}"
    )


def validate_overlap(dataset):

    print("\n=== FINAL OVERLAP VERIFICATION ===")

    references = load_reference_datasets()

    model_texts = set(dataset[TEXT_COLUMN])

    all_overlaps = set()

    for name, reference in references.items():

        overlap = model_texts.intersection(
            reference[TEXT_COLUMN]
        )

        all_overlaps.update(overlap)

        print(f"{name}: {len(overlap)}")

    print(
        f"Unique overlapping texts: "
        f"{len(all_overlaps)}"
    )

    if all_overlaps:

        raise ValueError(
            "Exact text overlap detected."
        )


def calculate_metrics(
    expected,
    predicted,
    top3_correct
):

    return {

        "records": len(expected),

        "correct": int(
            np.sum(expected == predicted)
        ),

        "accuracy": accuracy_score(
            expected,
            predicted
        ),

        "macro_precision": precision_score(
            expected,
            predicted,
            labels=INTENT_CLASSES,
            average="macro",
            zero_division=0
        ),

        "macro_recall": recall_score(
            expected,
            predicted,
            labels=INTENT_CLASSES,
            average="macro",
            zero_division=0
        ),

        "macro_f1": f1_score(
            expected,
            predicted,
            labels=INTENT_CLASSES,
            average="macro",
            zero_division=0
        ),

        "weighted_f1": f1_score(
            expected,
            predicted,
            labels=INTENT_CLASSES,
            average="weighted",
            zero_division=0
        ),

        "top3_accuracy": float(
            np.mean(top3_correct)
        ),
    }


def print_metrics(metrics):

    for name, value in metrics.items():

        if isinstance(value, float):

            print(f"{name}: {value:.4f}")

        else:

            print(f"{name}: {value}")


def evaluate_model(model, dataset):

    print("\n=== MODEL INFERENCE ===")

    expected = dataset[TARGET_COLUMN].to_numpy()

    classes = np.asarray(model.classes_)

    if set(classes) != set(INTENT_CLASSES):

        raise ValueError(
            "Model classes do not match taxonomy."
        )

    start_time = perf_counter()

    probabilities = model.predict_proba(
        dataset[TEXT_COLUMN]
    )

    inference_seconds = (
        perf_counter() - start_time
    )

    predicted_indices = np.argmax(
        probabilities,
        axis=1
    )

    predicted = classes[predicted_indices]

    top3_indices = np.argsort(
        -probabilities,
        axis=1,
        kind="stable"
    )[:, :3]

    top3_labels = classes[top3_indices]

    top3_correct = np.any(
        top3_labels == expected[:, None],
        axis=1
    )

    predictions = dataset.copy()

    predictions["expected"] = expected

    predictions["predicted"] = predicted

    predictions["correct"] = (
        expected == predicted
    )

    predictions["model_score"] = (
        probabilities[
            np.arange(len(dataset)),
            predicted_indices
        ]
    )

    predictions["top3_correct"] = top3_correct

    for rank in range(3):

        predictions[f"top{rank + 1}_label"] = (
            top3_labels[:, rank]
        )

        predictions[f"top{rank + 1}_score"] = (
            probabilities[
                np.arange(len(dataset)),
                top3_indices[:, rank]
            ]
        )

    metrics = calculate_metrics(
        expected,
        predicted,
        top3_correct
    )

    metrics["inference_seconds"] = (
        inference_seconds
    )

    metrics["milliseconds_per_record"] = (
        inference_seconds
        / len(dataset)
        * 1000
    )

    return predictions, metrics


def analyze_classes(predictions):

    print("\n=== PER-CLASS EVALUATION ===")

    expected = predictions["expected"]

    predicted = predictions["predicted"]

    report = classification_report(
        expected,
        predicted,
        labels=INTENT_CLASSES,
        digits=4,
        zero_division=0
    )

    print(report)

    report_dict = classification_report(
        expected,
        predicted,
        labels=INTENT_CLASSES,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report_dict
    ).transpose()

    class_summary = []

    for intent in INTENT_CLASSES:

        subset = predictions.loc[
            predictions["expected"].eq(intent)
        ]

        class_summary.append({

            "intent": intent,

            "records": len(subset),

            "correct": int(
                subset["correct"].sum()
            ),

            "accuracy": float(
                subset["correct"].mean()
            ),

            "precision":
                report_dict[intent]["precision"],

            "recall":
                report_dict[intent]["recall"],

            "f1":
                report_dict[intent]["f1-score"],

            "top3_accuracy": float(
                subset["top3_correct"].mean()
            ),
        })

    class_summary_df = pd.DataFrame(
        class_summary
    )

    print("\n=== CLASS SUMMARY ===")

    print(
        class_summary_df.round(4)
        .to_string(index=False)
    )

    return report_df, class_summary_df


def analyze_groups(predictions):

    print("\n=== PER-GROUP EVALUATION ===")

    group_results = []

    for group in GROUPS:

        subset = predictions.loc[
            predictions["group"].eq(group)
        ]

        metrics = calculate_metrics(

            subset["expected"].to_numpy(),

            subset["predicted"].to_numpy(),

            subset["top3_correct"].to_numpy()
        )

        metrics["group"] = group

        group_results.append(metrics)

        print(f"\n--- {group.upper()} ---")

        print_metrics(metrics)

    group_results_df = pd.DataFrame(
        group_results
    )

    print("\n=== GROUP COMPARISON ===")

    print(
        group_results_df[
            [
                "group",
                "records",
                "correct",
                "accuracy",
                "macro_f1",
                "top3_accuracy",
            ]
        ].round(4).to_string(index=False)
    )

    return group_results_df


def analyze_intent_group_accuracy(predictions):

    print("\n=== INTENT-GROUP ACCURACY ===")

    accuracy_table = pd.pivot_table(

        predictions,

        index="expected",

        columns="group",

        values="correct",

        aggfunc="mean"
    ).reindex(

        index=INTENT_CLASSES,

        columns=GROUPS
    )

    print(
        accuracy_table.round(4).to_string()
    )

    return accuracy_table


def create_confusion_matrix(predictions):

    print("\n=== CONFUSION MATRIX ===")

    matrix = confusion_matrix(

        predictions["expected"],

        predictions["predicted"],

        labels=INTENT_CLASSES
    )

    matrix_df = pd.DataFrame(

        matrix,

        index=INTENT_CLASSES,

        columns=INTENT_CLASSES
    )

    matrix_df.index.name = "actual"

    print(matrix_df.to_string())

    return matrix_df


def analyze_errors(predictions):

    errors = predictions.loc[
        ~predictions["correct"]
    ].copy()

    print("\n=== ERROR ANALYSIS ===")

    print(f"Total errors: {len(errors)}")

    print(
        f"Error rate: "
        f"{len(errors) / len(predictions):.2%}"
    )

    error_pairs = (

        errors.groupby(
            ["expected", "predicted"]
        )

        .size()

        .reset_index(name="count")

        .sort_values(
            "count",
            ascending=False
        )
    )

    print("\nMost frequent error pairs:")

    print(
        error_pairs.head(15)
        .to_string(index=False)
    )

    print("\nErrors by group:")

    print(
        errors["group"]
        .value_counts()
        .reindex(GROUPS, fill_value=0)
        .to_string()
    )

    recovered_by_top3 = int(
        errors["top3_correct"].sum()
    )

    print(
        "\nIncorrect Top-1 predictions "
        "with correct label in Top-3: "
        f"{recovered_by_top3}"
    )

    if not errors.empty:

        print(
            "Top-3 recovery among errors: "
            f"{recovered_by_top3 / len(errors):.2%}"
        )

    print("\n=== ALL INCORRECT PREDICTIONS ===")

    for row in errors.itertuples(index=False):

        print(f"\nID: {row.id}")

        print(f"Group: {row.group}")

        print(f"Text: {row.text}")

        print(f"Expected: {row.expected}")

        print(f"Predicted: {row.predicted}")

        print(
            f"Model score: {row.model_score:.4f}"
        )

        print(
            f"Top-3: "
            f"{row.top1_label} "
            f"({row.top1_score:.4f}), "
            f"{row.top2_label} "
            f"({row.top2_score:.4f}), "
            f"{row.top3_label} "
            f"({row.top3_score:.4f})"
        )

        print(
            f"Top-3 correct: {row.top3_correct}"
        )

    return errors, error_pairs


def main():

    print(
        "=== E09.06.2e: FROZEN MODEL "
        "INSTALLATION EVALUATION ==="
    )

    EXPERIMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    dataset = load_dataset()

    validate_dataset(dataset)

    validate_overlap(dataset)

    dataset_hash = calculate_sha256(
        DATASET_PATH
    )

    model_hash = calculate_sha256(
        MODEL_PATH
    )

    print("\n=== EXPERIMENT IDENTIFICATION ===")

    print(f"Dataset: {DATASET_PATH}")

    print(f"Dataset SHA-256: {dataset_hash}")

    print(f"Model: {MODEL_PATH}")

    print(f"Model SHA-256: {model_hash}")

    model = joblib.load(MODEL_PATH)

    print("\n=== FROZEN MODEL INFORMATION ===")

    print("Model: E09.05a")

    print(f"Classes: {len(model.classes_)}")

    print(
        "Vocabulary size: "
        f"{len(model.named_steps['tfidf'].vocabulary_):,}"
    )

    print(
        "Class weight: "
        f"{model.named_steps['classifier'].class_weight}"
    )

    predictions, metrics = evaluate_model(
        model,
        dataset
    )

    print("\n=== OVERALL EVALUATION ===")

    print_metrics(metrics)

    report_df, class_summary = analyze_classes(
        predictions
    )

    group_results = analyze_groups(
        predictions
    )

    intent_group_accuracy = (
        analyze_intent_group_accuracy(
            predictions
        )
    )

    matrix_df = create_confusion_matrix(
        predictions
    )

    errors, error_pairs = analyze_errors(
        predictions
    )

    results = pd.DataFrame([{

        "experiment": EXPERIMENT,

        "model": "E09.05a",

        "dataset_sha256": dataset_hash,

        "model_sha256": model_hash,

        **metrics,
    }])

    outputs = {

        "results":
            (results, False),

        "predictions":
            (predictions, False),

        "classification_report":
            (report_df, True),

        "class_summary":
            (class_summary, False),

        "group_results":
            (group_results, False),

        "intent_group_accuracy":
            (intent_group_accuracy, True),

        "confusion_matrix":
            (matrix_df, True),

        "errors":
            (errors, False),

        "error_pairs":
            (error_pairs, False),
    }

    print("\n=== SAVED EXPERIMENT FILES ===")

    for name, (frame, save_index) in outputs.items():

        path = (
            EXPERIMENTS_DIR
            / f"{EXPERIMENT}_{name}.csv"
        )

        frame.to_csv(
            path,
            index=save_index,
            encoding="utf-8"
        )

        print(path.name)

    print("\n=== EXPERIMENT INTEGRITY ===")

    dataset_unchanged = (
        calculate_sha256(DATASET_PATH)
        == dataset_hash
    )

    model_unchanged = (
        calculate_sha256(MODEL_PATH)
        == model_hash
    )

    print(
        f"Dataset unchanged: "
        f"{dataset_unchanged}"
    )

    print(
        f"Model file unchanged: "
        f"{model_unchanged}"
    )

    if not dataset_unchanged or not model_unchanged:

        raise RuntimeError(
            "Experiment integrity check failed."
        )

    print("Exact training text overlap: 0")

    print("Model training performed: NO")

    print("Hyperparameter tuning performed: NO")

    print("\n=== FINAL EXPERIMENT SUMMARY ===")

    print_metrics(metrics)

    print("\n=== FROZEN MODEL EVALUATION COMPLETE ===")


if __name__ == "__main__":
    main()
