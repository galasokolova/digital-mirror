from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data" / "processed" / "fever"

LABELS = ["REFUTES", "SUPPORTS"]
RANDOM_STATE = 42


# ============================================================
# DATA LOADING
# ============================================================

def load_data(split):
    path = DATA_DIR / f"{split}_verified.csv"

    df = pd.read_csv(path)

    # Use one complete evidence set per claim.
    df = (
        df.sort_values(["claim_id", "evidence_set_id"])
        .drop_duplicates(subset=["claim_id"])
        .reset_index(drop=True)
    )

    return df


# ============================================================
# INPUT PREPARATION
# ============================================================

def make_input(df, use_evidence):
    if use_evidence:
        return (
            "CLAIM: " + df["claim"]
            + " EVIDENCE: " + df["evidence"]
        )

    return df["claim"]


# ============================================================
# MODEL
# ============================================================

def create_model():
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
                max_features=100000,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ])


# ============================================================
# SHUFFLED EVIDENCE EXPERIMENT
# ============================================================

def test_shuffled_evidence(model, dev, original_predictions):
    print("\n=== E02: SHUFFLED EVIDENCE TEST ===")

    shuffled_dev = dev.copy()

    # Shuffle evidence while keeping claims and labels unchanged.
    shuffled_dev["evidence"] = (
        dev["evidence"]
        .sample(frac=1, random_state=RANDOM_STATE)
        .to_numpy()
    )

    x_shuffled = make_input(
        shuffled_dev,
        use_evidence=True,
    )

    y_dev = dev["label"]

    shuffled_predictions = model.predict(x_shuffled)

    original_f1 = f1_score(
        y_dev,
        original_predictions,
        average="macro",
    )

    shuffled_f1 = f1_score(
        y_dev,
        shuffled_predictions,
        average="macro",
    )

    shuffled_accuracy = accuracy_score(
        y_dev,
        shuffled_predictions,
    )

    changed_predictions = (
        original_predictions != shuffled_predictions
    ).mean()

    print(f"Original Macro F1: {original_f1:.4f}")
    print(f"Shuffled Macro F1: {shuffled_f1:.4f}")
    print(f"Shuffled Accuracy: {shuffled_accuracy:.4f}")

    print(
        f"Predictions changed: {changed_predictions:.2%}"
    )

    print(
        f"Macro F1 difference: "
        f"{shuffled_f1 - original_f1:+.4f}"
    )

    return shuffled_accuracy, shuffled_f1


# ============================================================
# TRAINING AND EVALUATION
# ============================================================

def train_and_evaluate(name, train, dev, use_evidence):
    print(f"\n{'=' * 55}")
    print(name)
    print("=" * 55)

    x_train = make_input(train, use_evidence)
    y_train = train["label"]

    x_dev = make_input(dev, use_evidence)
    y_dev = dev["label"]

    model = create_model()

    print("Training model...", flush=True)
    model.fit(x_train, y_train)

    print("Evaluating model...", flush=True)
    predictions = model.predict(x_dev)

    accuracy = accuracy_score(
        y_dev,
        predictions,
    )

    macro_f1 = f1_score(
        y_dev,
        predictions,
        average="macro",
    )

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_dev,
            predictions,
            labels=LABELS,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print("Labels:", LABELS)

    print(
        confusion_matrix(
            y_dev,
            predictions,
            labels=LABELS,
        )
    )

    shuffled_results = None

    # E02 applies only to the evidence-based model.
    if use_evidence:
        shuffled_results = test_shuffled_evidence(
            model,
            dev,
            predictions,
        )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "shuffled": shuffled_results,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading FEVER datasets...", flush=True)

    train = load_data("train")
    dev = load_data("dev")

    print("Train claims:", len(train))
    print("Dev claims:", len(dev))

    # Check for claim-ID overlap.
    overlap = (
        set(train["claim_id"])
        & set(dev["claim_id"])
    )

    print("Overlapping claim IDs:", len(overlap))

    if overlap:
        raise ValueError(
            "Train/dev claim-ID overlap detected"
        )

    results = {}

    experiments = [
        ("A: CLAIM ONLY", False),
        ("B: CLAIM + EVIDENCE", True),
    ]

    for name, use_evidence in experiments:
        results[name] = train_and_evaluate(
            name,
            train,
            dev,
            use_evidence,
        )

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print("\n" + "=" * 55)
    print("FINAL COMPARISON")
    print("=" * 55)

    for name, result in results.items():
        print(
            f"{name}: "
            f"Accuracy={result['accuracy']:.4f}, "
            f"Macro F1={result['macro_f1']:.4f}"
        )

    a_f1 = results["A: CLAIM ONLY"]["macro_f1"]
    b_f1 = results["B: CLAIM + EVIDENCE"]["macro_f1"]

    print(
        f"\nEvidence effect (B - A): {b_f1 - a_f1:+.4f}"
    )

    shuffled = results["B: CLAIM + EVIDENCE"]["shuffled"]

    if shuffled is not None:
        shuffled_accuracy, shuffled_f1 = shuffled

        print("\n=== E02 SUMMARY ===")
        print(f"Correct Evidence F1:  {b_f1:.4f}")
        print(f"Shuffled Evidence F1: {shuffled_f1:.4f}")

        print(
            f"Evidence shuffle effect: "
            f"{shuffled_f1 - b_f1:+.4f}"
        )

    print("\nExperiments completed.")


if __name__ == "__main__":
    main()
