from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

BASE = Path(__file__).resolve().parents[3]
DATA = BASE / "data" / "processed" / "fever"
OUTPUT = BASE / "data" / "experiments" / "fever"

SEED = 42


def load_data(split):
    df = pd.read_csv(DATA / f"{split}_context.csv")

    return (
        df.sort_values(["claim_id", "evidence_set_id"])
        .drop_duplicates("claim_id")
        .reset_index(drop=True)
    )


def make_features(df, claim_vectorizer, evidence_vectorizer):
    claim_features = claim_vectorizer.transform(df["claim"])
    evidence_features = evidence_vectorizer.transform(
        df["context_evidence"]
    )

    return hstack(
        [claim_features, evidence_features],
        format="csr"
    )


def evaluate(name, model, X, y, original_predictions=None):
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    accuracy = accuracy_score(y, predictions)
    macro_f1 = f1_score(y, predictions, average="macro")

    print(f"\n=== {name} ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    if original_predictions is not None:
        changed = np.mean(predictions != original_predictions)
        print(f"Changed predictions: {changed:.2%}")

    return predictions, probabilities


def main():
    train = load_data("train")
    dev = load_data("dev")

    assert set(train.claim_id).isdisjoint(set(dev.claim_id))

    print("Training model D...", flush=True)

    claim_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
        dtype=np.float32,
    )

    evidence_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
        dtype=np.float32,
    )

    claim_vectorizer.fit(train["claim"])
    evidence_vectorizer.fit(train["context_evidence"])

    X_train = make_features(
        train, claim_vectorizer, evidence_vectorizer
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=SEED,
    )

    model.fit(X_train, train["label"])

    # Original evidence
    X_original = make_features(
        dev, claim_vectorizer, evidence_vectorizer
    )

    original_pred, original_proba = evaluate(
        "Original Evidence",
        model,
        X_original,
        dev["label"],
    )

    rng = np.random.default_rng(SEED)

    # Random evidence from a different claim
    random_indices = rng.permutation(len(dev))

    # Avoid pairing a claim with its own evidence
    same = random_indices == np.arange(len(dev))
    if same.any():
        random_indices = np.roll(random_indices, 1)

    random_dev = dev.copy()
    random_dev["context_evidence"] = (
        dev["context_evidence"]
        .iloc[random_indices]
        .to_numpy()
    )

    X_random = make_features(
        random_dev, claim_vectorizer, evidence_vectorizer
    )

    random_pred, random_proba = evaluate(
        "Random Evidence",
        model,
        X_random,
        dev["label"],
        original_pred,
    )

    # Evidence from a claim with the opposite label
    opposite_dev = dev.copy()

    refutes_indices = np.flatnonzero(
        dev["label"].to_numpy() == "REFUTES"
    )
    supports_indices = np.flatnonzero(
        dev["label"].to_numpy() == "SUPPORTS"
    )

    opposite_indices = np.empty(len(dev), dtype=int)

    labels = dev["label"].to_numpy()

    for i, label in enumerate(labels):
        pool = (
            supports_indices
            if label == "REFUTES"
            else refutes_indices
        )
        opposite_indices[i] = rng.choice(pool)

    opposite_dev["context_evidence"] = (
        dev["context_evidence"]
        .iloc[opposite_indices]
        .to_numpy()
    )

    X_opposite = make_features(
        opposite_dev, claim_vectorizer, evidence_vectorizer
    )

    opposite_pred, opposite_proba = evaluate(
        "Opposite Label Evidence",
        model,
        X_opposite,
        dev["label"],
        original_pred,
    )

    # Compare probability assigned to SUPPORTS
    supports_col = list(model.classes_).index("SUPPORTS")

    original_supports = original_proba[:, supports_col]
    random_supports = random_proba[:, supports_col]
    opposite_supports = opposite_proba[:, supports_col]

    print("\n=== PROBABILITY SENSITIVITY ===")

    print(
        "Mean |P_original - P_random|:",
        round(np.mean(np.abs(
            original_supports - random_supports
        )), 4)
    )

    print(
        "Mean |P_original - P_opposite|:",
        round(np.mean(np.abs(
            original_supports - opposite_supports
        )), 4)
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame({
        "claim_id": dev["claim_id"],
        "claim": dev["claim"],
        "label": dev["label"],
        "original_evidence": dev["context_evidence"],
        "random_evidence": random_dev["context_evidence"],
        "opposite_evidence": opposite_dev["context_evidence"],
        "original_prediction": original_pred,
        "random_prediction": random_pred,
        "opposite_prediction": opposite_pred,
        "original_p_supports": original_supports,
        "random_p_supports": random_supports,
        "opposite_p_supports": opposite_supports,
    })

    path = OUTPUT / "e05c_evidence_sensitivity.csv"
    results.to_csv(path, index=False)

    print("\nSaved:", path)


if __name__ == "__main__":
    main()
