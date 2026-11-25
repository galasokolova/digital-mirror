
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


ML_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ML_ROOT / "src"))

from preprocessing.intent_text_normalizer import (
    remove_punctuation,
)

from taxonomy.intent_mapping import INTENT_CLASSES


MODELS_DIR = ML_ROOT / "models"
EXPERIMENTS_DIR = ML_ROOT / "experiments"
PROCESSED_DIR = ML_ROOT / "data" / "processed"

BENCHMARK_PATH = (
    EXPERIMENTS_DIR
    / "E09_03_installation_predictions.csv"
)

MODELS = {
    "E09_01": "E09_01_word_logreg.joblib",
    "E09_02": "E09_02_word_logreg_balanced.joblib",
    "E09_05a": "E09_05a_domain_adapted.joblib",
    "E09_05b": "E09_05b_domain_adapted.joblib",
    "E09_05c": "E09_05c_domain_adapted.joblib",
}


def load_benchmark():

    benchmark = pd.read_csv(
        BENCHMARK_PATH,
        dtype=str,
        keep_default_na=False
    )

    benchmark = (
        benchmark[["text", "expected"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    benchmark["model_text"] = (
        benchmark["text"]
        .apply(remove_punctuation)
    )

    assert len(benchmark) == 60

    assert benchmark["model_text"].is_unique

    assert (
        set(benchmark["expected"])
        == set(INTENT_CLASSES)
    )

    assert (
        benchmark["expected"]
        .value_counts()
        .eq(5)
        .all()
    )

    return benchmark


def load_training_texts():

    switchboard = pd.read_csv(
        PROCESSED_DIR / "intent_v1" / "train.csv",
        usecols=["model_text"],
        dtype=str,
        keep_default_na=False
    )

    domain = pd.read_csv(
        PROCESSED_DIR
        / "digital_mirror_v1"
        / "train_eligible.csv",
        usecols=["model_text"],
        dtype=str,
        keep_default_na=False
    )

    return (
        set(switchboard["model_text"]),
        set(domain["model_text"])
    )


def evaluate_model(
    experiment,
    filename,
    benchmark,
    training_texts,
):

    print(f"\n{'=' * 70}")
    print(f"MODEL: {experiment}")
    print(f"{'=' * 70}")

    model = joblib.load(
        MODELS_DIR / filename
    )

    X = benchmark["model_text"]
    y = benchmark["expected"]

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)

    classes = model.named_steps[
        "classifier"
    ].classes_

    top3_indices = np.argsort(
        -probabilities,
        axis=1
    )[:, :3]

    top3_labels = classes[top3_indices]

    top3_correct = (
        top3_labels == y.to_numpy()[:, None]
    ).any(axis=1)

    correct = predictions == y.to_numpy()

    seen_mask = X.isin(
        training_texts
    ).to_numpy()

    accuracy = accuracy_score(
        y,
        predictions
    )

    macro_f1 = f1_score(
        y,
        predictions,
        labels=INTENT_CLASSES,
        average="macro",
        zero_division=0
    )

    print("\n=== BENCHMARK METRICS ===")

    print(f"Correct: {correct.sum()} / {len(y)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Top-3 Accuracy: {top3_correct.mean():.4f}")

    print("\n=== CLASSIFICATION REPORT ===")

    report = classification_report(
        y,
        predictions,
        labels=INTENT_CLASSES,
        digits=4,
        zero_division=0
    )

    print(report)

    report_dict = classification_report(
        y,
        predictions,
        labels=INTENT_CLASSES,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report_dict).transpose().to_csv(
        EXPERIMENTS_DIR
        / f"E09_05_01_{experiment}_report.csv"
    )

    matrix = confusion_matrix(
        y,
        predictions,
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
        / f"E09_05_01_{experiment}_confusion_matrix.csv"
    )

    print("\n=== SEEN / UNSEEN ANALYSIS ===")

    subset_results = {}

    for name, mask in [
        ("seen", seen_mask),
        ("unseen", ~seen_mask),
    ]:

        count = int(mask.sum())

        subset_accuracy = (
            float(correct[mask].mean())
            if count > 0
            else float("nan")
        )

        print(f"\n{name.upper()}")
        print(f"Records: {count}")
        print(f"Correct: {correct[mask].sum()}")
        print(f"Accuracy: {subset_accuracy:.4f}")

        subset_results[f"{name}_records"] = count
        subset_results[f"{name}_accuracy"] = (
            subset_accuracy
        )

    # Store individual predictions

    details = benchmark.copy()

    details["experiment"] = experiment
    details["predicted"] = predictions
    details["correct"] = correct
    details["seen_in_training"] = seen_mask

    details["model_score"] = (
        probabilities.max(axis=1)
    )

    details["top3_correct"] = top3_correct

    details["top3_labels"] = [
        ", ".join(labels)
        for labels in top3_labels
    ]

    print("\n=== MISCLASSIFIED UTTERANCES ===")

    mistakes = details.loc[
        ~details["correct"],
        [
            "text",
            "expected",
            "predicted",
            "model_score",
            "seen_in_training",
        ]
    ]

    print(
        mistakes.round(4).to_string(index=False)
    )

    summary = {
        "experiment": experiment,
        "records": len(y),
        "correct": int(correct.sum()),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "top3_accuracy": float(
            top3_correct.mean()
        ),
        **subset_results,
    }

    return summary, details


def main():

    print(
        "=== E09.05.1: INSTALLATION BENCHMARK COMPARISON ==="
    )

    benchmark = load_benchmark()

    switchboard_texts, domain_texts = (
        load_training_texts()
    )

    assert not benchmark["model_text"].isin(
        domain_texts
    ).any()

    print(f"\nBenchmark records: {len(benchmark)}")
    print(f"Models: {len(MODELS)}")

    print("\nClass distribution:")

    print(
        benchmark["expected"]
        .value_counts()
        .to_string()
    )

    summaries = []
    all_predictions = []

    for experiment, filename in MODELS.items():

        if experiment.startswith("E09_05"):

            training_texts = (
                switchboard_texts | domain_texts
            )

        else:

            training_texts = switchboard_texts

        summary, details = evaluate_model(
            experiment,
            filename,
            benchmark,
            training_texts
        )

        summaries.append(summary)
        all_predictions.append(details)

    results = pd.DataFrame(summaries)

    results = results.sort_values(
        ["macro_f1", "accuracy"],
        ascending=False
    )

    print("\n=== FINAL MODEL COMPARISON ===")

    print(
        results.round(4).to_string(index=False)
    )

    results.to_csv(
        EXPERIMENTS_DIR
        / "E09_05_01_installation_comparison.csv",
        index=False
    )

    predictions = pd.concat(
        all_predictions,
        ignore_index=True
    )

    predictions.to_csv(
        EXPERIMENTS_DIR
        / "E09_05_01_installation_predictions.csv",
        index=False
    )

    # Compare correct predictions per class

    class_comparison = predictions.pivot_table(
        index="expected",
        columns="experiment",
        values="correct",
        aggfunc="sum"
    ).reindex(INTENT_CLASSES)

    class_comparison = class_comparison.astype(int)

    print("\n=== CORRECT PREDICTIONS PER CLASS ===")
    print("Maximum per class: 5")

    print(class_comparison.to_string())

    class_comparison.to_csv(
        EXPERIMENTS_DIR
        / "E09_05_01_class_comparison.csv"
    )

    print("\n=== BENCHMARK COMPARISON COMPLETE ===")


if __name__ == "__main__":
    main()
