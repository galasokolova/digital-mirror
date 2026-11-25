
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


ML_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ML_ROOT / "data" / "processed" / "intent_v1"
MODELS_DIR = ML_ROOT / "models"
EXPERIMENTS_DIR = ML_ROOT / "experiments"

RANDOM_STATE = 42

TEXT_COLUMN = "model_text"
TARGET_COLUMN = "intent"


def load_data():

    train = pd.read_csv(
        DATA_DIR / "train.csv",
        keep_default_na=False
    )

    validation = pd.read_csv(
        DATA_DIR / "validation.csv",
        keep_default_na=False
    )

    assert train[TEXT_COLUMN].str.strip().ne("").all()
    assert validation[TEXT_COLUMN].str.strip().ne("").all()

    assert not (
        set(train["conversation_no"])
        & set(validation["conversation_no"])
    )

    assert set(train[TARGET_COLUMN]) == set(
        validation[TARGET_COLUMN]
    )

    return train, validation


def evaluate_model(
    experiment,
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    seen_mask
):

    print("\n" + "=" * 70)
    print(experiment)
    print("=" * 70)

    # Training
    print("\nTraining started...")

    start = perf_counter()

    model.fit(X_train, y_train)

    training_time = perf_counter() - start

    print("Training completed.")

    # Prediction
    start = perf_counter()

    predictions = model.predict(X_val)

    inference_time = perf_counter() - start

    labels = sorted(y_train.unique())

    # Evaluation
    accuracy = accuracy_score(
        y_val,
        predictions
    )

    macro_f1 = f1_score(
        y_val,
        predictions,
        labels=labels,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_val,
        predictions,
        labels=labels,
        average="weighted",
        zero_division=0
    )

    print("\n=== METRICS ===")

    print(f"Accuracy:       {accuracy:.4f}")
    print(f"Macro F1:       {macro_f1:.4f}")
    print(f"Weighted F1:    {weighted_f1:.4f}")
    print(f"Training time:  {training_time:.2f}s")
    print(f"Inference time: {inference_time:.4f}s")

    print("\n=== CLASSIFICATION REPORT ===")

    print(
        classification_report(
            y_val,
            predictions,
            labels=labels,
            digits=4,
            zero_division=0
        )
    )

    print("\n=== CONFUSION MATRIX ===")

    matrix = pd.DataFrame(
        confusion_matrix(
            y_val,
            predictions,
            labels=labels
        ),
        index=labels,
        columns=labels
    )

    matrix.index.name = "Actual"
    matrix.columns.name = "Predicted"

    print(matrix.to_string())

    # Seen and unseen text evaluation
    print("\n=== SEEN / UNSEEN TEXT EVALUATION ===")

    for name, mask in [
        ("Seen", seen_mask),
        ("Unseen", ~seen_mask)
    ]:

        if not mask.any():
            continue

        subset_true = y_val[mask]
        subset_pred = predictions[mask.to_numpy()]

        subset_accuracy = accuracy_score(
            subset_true,
            subset_pred
        )

        subset_f1 = f1_score(
            subset_true,
            subset_pred,
            labels=labels,
            average="macro",
            zero_division=0
        )

        print(f"\n{name} texts")

        print(f"Rows: {mask.sum():,}")
        print(f"Accuracy: {subset_accuracy:.4f}")
        print(f"Macro F1 (all classes): {subset_f1:.4f}")

        print(
            f"Classes present: "
            f"{subset_true.nunique()}/{len(labels)}"
        )

    # Save confusion matrix
    matrix.to_csv(
        EXPERIMENTS_DIR
        / f"{experiment}_confusion_matrix.csv"
    )

    return {
        "experiment": experiment,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "training_seconds": training_time,
        "inference_seconds": inference_time,
    }


def main():

    print("=== E09: INTENT CLASSIFIER BASELINES ===")

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    EXPERIMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train, validation = load_data()

    X_train = train[TEXT_COLUMN]
    y_train = train[TARGET_COLUMN]

    X_val = validation[TEXT_COLUMN]
    y_val = validation[TARGET_COLUMN]

    seen_mask = X_val.isin(set(X_train))

    print(f"Train rows: {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Classes: {y_train.nunique()}")
    print(f"Text column: {TEXT_COLUMN}")
    print(f"Random state: {RANDOM_STATE}")

    print(
        f"Seen validation rows: "
        f"{seen_mask.sum():,}"
    )

    print(
        f"Unseen validation rows: "
        f"{(~seen_mask).sum():,}"
    )

    experiments = {

        # E09.00: Majority baseline
        "E09_00_majority": DummyClassifier(
            strategy="most_frequent"
        ),

        # E09.01: Word TF-IDF + Logistic Regression
        "E09_01_word_logreg": Pipeline([

            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=100_000,
                    sublinear_tf=True,
                    token_pattern=r"(?u)\b\w[\w']*\b"
                )
            ),

            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    solver="lbfgs",
                    class_weight=None,
                    random_state=RANDOM_STATE
                )
            ),

        ]),
    }

    results = []

    for experiment, model in experiments.items():

        result = evaluate_model(
            experiment,
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            seen_mask
        )

        results.append(result)

        if isinstance(model, Pipeline):

            model_path = (
                MODELS_DIR
                / f"{experiment}.joblib"
            )

            joblib.dump(model, model_path)

            print(f"\nModel saved: {model_path}")

            vocabulary_size = len(
                model.named_steps["tfidf"].vocabulary_
            )

            print(
                f"Vocabulary size: "
                f"{vocabulary_size:,}"
            )

            print(
                "Training iterations:",
                model.named_steps["classifier"].n_iter_
            )

    summary = pd.DataFrame(results)

    print("\n=== EXPERIMENT COMPARISON ===")

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    summary_path = (
        EXPERIMENTS_DIR
        / "E09_baseline_results.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    print(f"\nResults saved: {summary_path}")

    print("\n=== TRAINING COMPLETE ===")


if __name__ == "__main__":
    main()
