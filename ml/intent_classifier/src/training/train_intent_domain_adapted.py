
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from train_intent_baseline import (
    load_data,
    MODELS_DIR,
    EXPERIMENTS_DIR,
    RANDOM_STATE,
    TEXT_COLUMN,
    TARGET_COLUMN,
)


DOMAIN_PATH = (
    MODELS_DIR.parent
    / "data"
    / "processed"
    / "digital_mirror_v1"
    / "train_eligible.csv"
)

EXPERIMENTS = {
    "E09_05a": 1.0,
    "E09_05b": 10.0,
    "E09_05c": 50.0,
}


def create_model():

    return Pipeline([

        (
            "tfidf",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=2,
                max_features=100_000,
                sublinear_tf=True,
                token_pattern=r"(?u)\b\w[\w']*\b",
            )
        ),

        (
            "classifier",
            LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
    ])


def train_experiment(
    experiment,
    domain_weight,
    X_train,
    y_train,
    X_val,
    y_val,
    domain_mask,
):

    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT: {experiment}")
    print(f"{'=' * 70}")

    # Assign weights to individual training records

    sample_weights = np.ones(
        len(X_train),
        dtype=float
    )

    sample_weights[domain_mask] = domain_weight

    print("\n=== SAMPLE WEIGHTS ===")

    print("Switchboard weight: 1.0")
    print(f"Digital Mirror weight: {domain_weight}")

    print(
        f"Switchboard total weight: "
        f"{sample_weights[~domain_mask].sum():.2f}"
    )

    print(
        f"Digital Mirror total weight: "
        f"{sample_weights[domain_mask].sum():.2f}"
    )

    domain_share = (
        sample_weights[domain_mask].sum()
        / sample_weights.sum()
    )

    print(
        f"Domain weight share before class weighting: "
        f"{domain_share * 100:.2f}%"
    )

    # Create and train the model

    model = create_model()

    print("\nTraining model...")

    start = perf_counter()

    model.fit(
        X_train,
        y_train,
        classifier__sample_weight=sample_weights
    )

    training_seconds = perf_counter() - start

    print(
        f"Training completed in "
        f"{training_seconds:.4f} seconds."
    )

    # Evaluate the model

    print("\nEvaluating model...")

    start = perf_counter()

    predictions = model.predict(X_val)

    inference_seconds = perf_counter() - start

    labels = sorted(y_train.unique())

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

    print("\n=== VALIDATION METRICS ===")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print(
        f"Inference time: "
        f"{inference_seconds:.4f} seconds"
    )

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

    # Save the classification report

    report = classification_report(
        y_val,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        EXPERIMENTS_DIR
        / f"{experiment}_classification_report.csv"
    )

    # Confusion matrix

    matrix = confusion_matrix(
        y_val,
        predictions,
        labels=labels
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels
    )

    print("\n=== CONFUSION MATRIX ===")

    print(matrix_df.to_string())

    matrix_df.to_csv(
        EXPERIMENTS_DIR
        / f"{experiment}_confusion_matrix.csv"
    )

    # Seen and unseen validation texts

    training_texts = set(X_train)

    seen_mask = X_val.isin(
        training_texts
    ).to_numpy()

    print("\n=== SEEN / UNSEEN ANALYSIS ===")

    subset_results = {}

    for name, mask in [
        ("seen", seen_mask),
        ("unseen", ~seen_mask),
    ]:

        count = int(mask.sum())

        print(f"\n{name.upper()}")
        print(f"Records: {count}")

        if count == 0:
            continue

        subset_accuracy = accuracy_score(
            y_val.iloc[mask],
            predictions[mask]
        )

        subset_macro_f1 = f1_score(
            y_val.iloc[mask],
            predictions[mask],
            labels=labels,
            average="macro",
            zero_division=0
        )

        print(
            f"Accuracy: "
            f"{subset_accuracy:.4f}"
        )

        print(
            f"Macro F1: "
            f"{subset_macro_f1:.4f}"
        )

        subset_results[f"{name}_records"] = count

        subset_results[
            f"{name}_accuracy"
        ] = subset_accuracy

        subset_results[
            f"{name}_macro_f1"
        ] = subset_macro_f1

    # Model information

    vocabulary_size = len(
        model.named_steps["tfidf"].vocabulary_
    )

    iterations = int(
        model.named_steps["classifier"]
        .n_iter_.max()
    )

    print("\n=== MODEL INFORMATION ===")

    print(f"Vocabulary size: {vocabulary_size}")
    print(f"Training iterations: {iterations}")

    # Save the trained model

    model_path = (
        MODELS_DIR
        / f"{experiment}_domain_adapted.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    print(f"\nModel saved to: {model_path}")

    return {
        "experiment": experiment,
        "domain_weight": domain_weight,
        "training_records": len(X_train),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "vocabulary_size": vocabulary_size,
        "iterations": iterations,
        **subset_results,
    }


def main():

    print(
        "=== E09.05: DOMAIN-ADAPTED MODEL TRAINING ==="
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    EXPERIMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load the original Switchboard datasets

    train, validation = load_data()

    # Load the eligible domain-specific records

    domain = pd.read_csv(
        DOMAIN_PATH,
        dtype=str,
        keep_default_na=False
    )

    print("\n=== DATASET OVERVIEW ===")

    print(f"Switchboard train: {len(train):,}")
    print(f"Switchboard validation: {len(validation):,}")
    print(f"Digital Mirror train: {len(domain):,}")

    # Validate the domain-specific dataset

    assert len(domain) == 579

    assert domain["id"].is_unique

    assert domain[TEXT_COLUMN].is_unique

    assert not domain[TEXT_COLUMN].eq("").any()

    assert (
        set(domain[TARGET_COLUMN])
        == set(train[TARGET_COLUMN])
    )

    assert not domain[TEXT_COLUMN].isin(
        validation[TEXT_COLUMN]
    ).any()

    # Combine the training datasets

    switchboard = train[
        [TEXT_COLUMN, TARGET_COLUMN]
    ].copy()

    switchboard["dataset_origin"] = "switchboard"

    domain_train = domain[
        [TEXT_COLUMN, TARGET_COLUMN]
    ].copy()

    domain_train["dataset_origin"] = "digital_mirror"

    combined = pd.concat(
        [switchboard, domain_train],
        ignore_index=True
    )

    assert len(combined) == (
        len(train) + len(domain)
    )

    print("\n=== COMBINED TRAINING DATASET ===")

    print(f"Records: {len(combined):,}")

    print("\nDataset origins:")

    print(
        combined["dataset_origin"]
        .value_counts()
        .to_string()
    )

    # Compare class distributions

    distribution = pd.DataFrame({
        "switchboard": train[TARGET_COLUMN].value_counts(),
        "digital_mirror": domain[TARGET_COLUMN].value_counts(),
        "combined": combined[TARGET_COLUMN].value_counts(),
    }).fillna(0).astype(int)

    print("\n=== CLASS DISTRIBUTION ===")

    print(distribution.to_string())

    # Prepare model inputs

    X_train = combined[TEXT_COLUMN]
    y_train = combined[TARGET_COLUMN]

    X_val = validation[TEXT_COLUMN]
    y_val = validation[TARGET_COLUMN]

    domain_mask = (
        combined["dataset_origin"]
        .eq("digital_mirror")
        .to_numpy()
    )

    assert domain_mask.sum() == len(domain)

    # Run all experiments

    results = []

    for experiment, weight in EXPERIMENTS.items():

        result = train_experiment(
            experiment=experiment,
            domain_weight=weight,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            domain_mask=domain_mask,
        )

        results.append(result)

        # Save intermediate results after each model

        pd.DataFrame(results).to_csv(
            EXPERIMENTS_DIR
            / "E09_05_domain_adaptation_results.csv",
            index=False
        )

    # Final experiment comparison

    results_df = pd.DataFrame(results)

    print("\n=== DOMAIN ADAPTATION COMPARISON ===")

    print(
        results_df.round(4).to_string(
            index=False
        )
    )

    print("\n=== ALL EXPERIMENTS COMPLETE ===")


if __name__ == "__main__":
    main()
