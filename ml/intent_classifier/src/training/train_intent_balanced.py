
from time import perf_counter

import joblib
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

from train_intent_baseline import (
    load_data,
    evaluate_model,
    MODELS_DIR,
    EXPERIMENTS_DIR,
    RANDOM_STATE,
    TEXT_COLUMN,
    TARGET_COLUMN,
)


EXPERIMENT = "E09_02_word_logreg_balanced"


def main():

    print("=== E09.02: BALANCED LOGISTIC REGRESSION ===")

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
    print(f"Random state: {RANDOM_STATE}")
    print(f"Text column: {TEXT_COLUMN}")

    # Calculate and display class weights
    classes = np.unique(y_train.to_numpy())

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train.to_numpy()
    )

    weight_table = pd.DataFrame({
        "intent": classes,
        "weight": weights
    })

    print("\n=== CLASS WEIGHTS ===")

    print(
        weight_table.sort_values(
            "weight",
            ascending=False
        ).to_string(index=False)
    )

    # Same configuration as E09.01,
    # except for class_weight
    model = Pipeline([

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
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        ),

    ])

    result = evaluate_model(
        EXPERIMENT,
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        seen_mask
    )

    # Save trained model
    model_path = (
        MODELS_DIR
        / f"{EXPERIMENT}.joblib"
    )

    joblib.dump(model, model_path)

    print(f"\nModel saved: {model_path}")

    print(
        "Vocabulary size:",
        len(model.named_steps["tfidf"].vocabulary_)
    )

    print(
        "Training iterations:",
        model.named_steps["classifier"].n_iter_
    )

    # Compare with previous experiments
    baseline_path = (
        EXPERIMENTS_DIR
        / "E09_baseline_results.csv"
    )

    current_result = pd.DataFrame([result])

    if baseline_path.exists():

        baseline = pd.read_csv(baseline_path)

        comparison = pd.concat(
            [baseline, current_result],
            ignore_index=True
        )

    else:

        comparison = current_result

    print("\n=== EXPERIMENT COMPARISON ===")

    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    results_path = (
        EXPERIMENTS_DIR
        / "E09_02_results.csv"
    )

    comparison.to_csv(
        results_path,
        index=False
    )

    print(f"\nResults saved: {results_path}")

    print("\n=== TRAINING COMPLETE ===")


if __name__ == "__main__":
    main()
