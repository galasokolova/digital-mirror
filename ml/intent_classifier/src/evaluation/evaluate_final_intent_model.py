
from pathlib import Path
from time import perf_counter
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


MODELS_DIR = ML_ROOT / "models"
EXPERIMENTS_DIR = ML_ROOT / "experiments"
PROCESSED_DIR = ML_ROOT / "data" / "processed"

SWITCHBOARD_DIR = PROCESSED_DIR / "intent_v1"

DOMAIN_PATH = (
    PROCESSED_DIR
    / "digital_mirror_v1"
    / "train_eligible.csv"
)

MODEL_PATH = (
    MODELS_DIR
    / "E09_05a_domain_adapted.joblib"
)

EXPERIMENT = "E09_06_01"

TEXT_COLUMN = "model_text"
TARGET_COLUMN = "intent"


def load_dataset(path):

    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False
    )


def validate_datasets(
    train,
    validation,
    test,
    domain
):

    print("\n=== DATASET VALIDATION ===")

    train_conversations = set(
        train["conversation_no"]
    )

    validation_conversations = set(
        validation["conversation_no"]
    )

    test_conversations = set(
        test["conversation_no"]
    )

    checks = {

        "Expected training size":
            len(train) == 135827,

        "Expected validation size":
            len(validation) == 16913,

        "Expected test size":
            len(test) == 17966,

        "Expected domain size":
            len(domain) == 579,

        "No empty test texts":
            not test[TEXT_COLUMN].eq("").any(),

        "All test classes present":
            set(test[TARGET_COLUMN])
            == set(INTENT_CLASSES),

        "Train-test conversation isolation":
            not (
                train_conversations
                & test_conversations
            ),

        "Validation-test conversation isolation":
            not (
                validation_conversations
                & test_conversations
            ),

        "No domain-test text overlap":
            not domain[TEXT_COLUMN].isin(
                test[TEXT_COLUMN]
            ).any(),

        "No domain-validation text overlap":
            not domain[TEXT_COLUMN].isin(
                validation[TEXT_COLUMN]
            ).any(),
    }

    for name, passed in checks.items():

        status = "PASS" if passed else "FAIL"

        print(f"{status} | {name}")

    if not all(checks.values()):

        raise ValueError(
            "Dataset validation failed."
        )


def calculate_metrics(
    expected,
    predicted,
    top3_correct
):

    return {

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
            top3_correct.mean()
        ),
    }


def evaluate_dataset(
    model,
    dataset,
    dataset_name,
    training_texts
):

    print(f"\n{'=' * 70}")
    print(f"DATASET: {dataset_name.upper()}")
    print(f"{'=' * 70}")

    X = dataset[TEXT_COLUMN]
    y = dataset[TARGET_COLUMN]

    expected = y.to_numpy()

    classes = model.named_steps[
        "classifier"
    ].classes_

    # Inference only: no model training

    start = perf_counter()

    probabilities = model.predict_proba(X)

    inference_seconds = perf_counter() - start

    predicted = classes[
        probabilities.argmax(axis=1)
    ]

    # Extract top-3 predictions

    top3_indices = np.argsort(
        -probabilities,
        axis=1
    )[:, :3]

    top3_labels = classes[top3_indices]

    top3_correct = (
        top3_labels == expected[:, None]
    ).any(axis=1)

    correct = predicted == expected

    metrics = calculate_metrics(
        expected,
        predicted,
        top3_correct
    )

    print("\n=== EVALUATION METRICS ===")

    print(f"Records: {len(dataset):,}")

    print(
        f"Correct predictions: "
        f"{correct.sum():,}"
    )

    print(
        f"Incorrect predictions: "
        f"{(~correct).sum():,}"
    )

    for name, value in metrics.items():

        print(f"{name}: {value:.4f}")

    print("\n=== INFERENCE PERFORMANCE ===")

    print(
        f"Total inference time: "
        f"{inference_seconds:.4f} seconds"
    )

    milliseconds_per_record = (
        inference_seconds
        / len(dataset)
        * 1000
    )

    print(
        f"Average batch inference time per record: "
        f"{milliseconds_per_record:.6f} ms"
    )

    # Classification report

    print("\n=== CLASSIFICATION REPORT ===")

    print(
        classification_report(
            expected,
            predicted,
            labels=INTENT_CLASSES,
            digits=4,
            zero_division=0
        )
    )

    report = classification_report(
        expected,
        predicted,
        labels=INTENT_CLASSES,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        EXPERIMENTS_DIR
        / f"{EXPERIMENT}_{dataset_name}_report.csv"
    )

    # Confusion matrix

    matrix = confusion_matrix(
        expected,
        predicted,
        labels=INTENT_CLASSES
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=INTENT_CLASSES,
        columns=INTENT_CLASSES
    )

    print("\n=== CONFUSION MATRIX ===")

    print(matrix_df.to_string())

    matrix_df.to_csv(
        EXPERIMENTS_DIR
        / f"{EXPERIMENT}_{dataset_name}_confusion_matrix.csv"
    )

    # Seen / unseen evaluation

    seen_mask = X.isin(
        training_texts
    ).to_numpy()

    print("\n=== SEEN / UNSEEN ANALYSIS ===")

    subset_results = []

    for name, mask in [

        ("seen", seen_mask),
        ("unseen", ~seen_mask),

    ]:

        count = int(mask.sum())

        print(f"\n{name.upper()}")
        print(f"Records: {count:,}")

        if count == 0:
            continue

        subset_metrics = calculate_metrics(
            expected[mask],
            predicted[mask],
            top3_correct[mask]
        )

        print(
            f"Correct predictions: "
            f"{correct[mask].sum():,}"
        )

        print(
            f"Classes present: "
            f"{len(set(expected[mask]))}"
            f" / {len(INTENT_CLASSES)}"
        )

        for metric, value in subset_metrics.items():

            print(f"{metric}: {value:.4f}")

        subset_results.append({

            "dataset": dataset_name,
            "subset": name,
            "records": count,
            "correct": int(correct[mask].sum()),
            **subset_metrics,
        })

    # Save individual predictions

    details = pd.DataFrame({

        "model_text": X,
        "expected": expected,
        "predicted": predicted,
        "correct": correct,
        "seen_in_training": seen_mask,

        "model_score":
            probabilities.max(axis=1),

        "top3_correct":
            top3_correct,

        "top3_labels": [
            ", ".join(labels)
            for labels in top3_labels
        ],
    })

    details.to_csv(
        EXPERIMENTS_DIR
        / f"{EXPERIMENT}_{dataset_name}_predictions.csv",
        index=False
    )

    # Analyze the most frequent classification errors

    mistakes = details.loc[
        ~details["correct"]
    ]

    error_pairs = (
        mistakes.groupby(
            ["expected", "predicted"]
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            "count",
            ascending=False
        )
    )

    print("\n=== TOP 15 CLASSIFICATION ERRORS ===")

    print(
        error_pairs.head(15)
        .to_string(index=False)
    )

    error_pairs.to_csv(
        EXPERIMENTS_DIR
        / f"{EXPERIMENT}_{dataset_name}_errors.csv",
        index=False
    )

    summary = {

        "dataset": dataset_name,
        "records": len(dataset),
        "correct": int(correct.sum()),
        "incorrect": int((~correct).sum()),
        **metrics,

        "inference_seconds":
            inference_seconds,

        "milliseconds_per_record":
            milliseconds_per_record,
    }

    return summary, subset_results


def main():

    print(
        "=== E09.06.1: SWITCHBOARD TEST EVALUATION ==="
    )

    EXPERIMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load frozen datasets

    train = load_dataset(
        SWITCHBOARD_DIR / "train.csv"
    )

    validation = load_dataset(
        SWITCHBOARD_DIR / "validation.csv"
    )

    test = load_dataset(
        SWITCHBOARD_DIR / "test.csv"
    )

    domain = load_dataset(
        DOMAIN_PATH
    )

    print("\n=== DATASET OVERVIEW ===")

    print(f"Switchboard train: {len(train):,}")
    print(f"Digital Mirror train: {len(domain):,}")
    print(f"Switchboard validation: {len(validation):,}")
    print(f"Switchboard test: {len(test):,}")

    validate_datasets(
        train,
        validation,
        test,
        domain
    )

    # Load the selected frozen model

    print("\n=== MODEL INFORMATION ===")

    print("Selected model: E09.05a")
    print(f"Model path: {MODEL_PATH}")

    model = joblib.load(
        MODEL_PATH
    )

    classifier = model.named_steps[
        "classifier"
    ]

    assert (
        set(classifier.classes_)
        == set(INTENT_CLASSES)
    )

    print(
        f"Classes: "
        f"{len(classifier.classes_)}"
    )

    print(
        f"Vocabulary size: "
        f"{len(model.named_steps['tfidf'].vocabulary_)}"
    )

    print(
        f"Class weight: "
        f"{classifier.class_weight}"
    )

    print("Training performed: NO")

    training_texts = (
        set(train[TEXT_COLUMN])
        | set(domain[TEXT_COLUMN])
    )

    print("\n=== TEST CLASS DISTRIBUTION ===")

    print(
        test[TARGET_COLUMN]
        .value_counts()
        .reindex(INTENT_CLASSES)
        .to_string()
    )

    # Evaluate validation and test datasets

    summaries = []
    all_subsets = []

    for dataset_name, dataset in [

        ("validation", validation),
        ("test", test),

    ]:

        summary, subsets = evaluate_dataset(
            model=model,
            dataset=dataset,
            dataset_name=dataset_name,
            training_texts=training_texts
        )

        summaries.append(summary)

        all_subsets.extend(subsets)

    # Save evaluation summaries

    results = pd.DataFrame(summaries)

    results.to_csv(
        EXPERIMENTS_DIR
        / f"{EXPERIMENT}_results.csv",
        index=False
    )

    subset_df = pd.DataFrame(
        all_subsets
    )

    subset_df.to_csv(
        EXPERIMENTS_DIR
        / f"{EXPERIMENT}_seen_unseen.csv",
        index=False
    )

    # Compare validation and test metrics

    metric_columns = [

        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "top3_accuracy",
    ]

    comparison = (
        results
        .set_index("dataset")[metric_columns]
        .transpose()
    )

    comparison["difference"] = (
        comparison["test"]
        - comparison["validation"]
    )

    comparison["difference_pp"] = (
        comparison["difference"] * 100
    )

    print("\n=== VALIDATION VS TEST ===")

    print(
        comparison.round(4)
        .to_string()
    )

    comparison.to_csv(
        EXPERIMENTS_DIR
        / f"{EXPERIMENT}_validation_test_comparison.csv"
    )

    print("\n=== SEEN / UNSEEN COMPARISON ===")

    print(
        subset_df.round(4)
        .to_string(index=False)
    )

    print("\n=== FINAL TEST EVALUATION COMPLETE ===")


if __name__ == "__main__":
    main()
