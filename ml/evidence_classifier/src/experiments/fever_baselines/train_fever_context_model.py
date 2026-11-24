import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from joblib import dump

BASE = Path(__file__).resolve().parents[3]
DATA = BASE / "data" / "processed" / "fever"
OUTPUT = BASE / "data" / "experiments" / "fever"
MODELS = BASE / "models"

LABELS = ["REFUTES", "SUPPORTS"]
NEGATIONS = {"not", "no", "never", "without", "neither"}


def tokens(text):
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def numbers(text):
    return set(re.findall(r"\b\d+\b", str(text)))


def relationship_features(df):
    features = []

    for row in df.itertuples(index=False):
        claim = tokens(row.claim)
        evidence = tokens(row.evidence)
        title = tokens(row.wiki_titles)

        shared = len(claim & evidence)
        union = len(claim | evidence)

        claim_numbers = numbers(row.claim)
        evidence_numbers = numbers(row.evidence)

        claim_neg = bool(claim & NEGATIONS)
        evidence_neg = bool(evidence & NEGATIONS)

        features.append([
            shared / max(len(claim), 1),
            shared / max(union, 1),
            len(claim & title) / max(len(title), 1),
            float(claim_neg != evidence_neg),
            float(claim_numbers == evidence_numbers),
            float(
                bool(claim_numbers)
                and bool(evidence_numbers)
                and claim_numbers != evidence_numbers
            ),
        ])

    return np.asarray(features, dtype=np.float32)


def load_split(name):
    df = pd.read_csv(DATA / f"{name}_context.csv")

    # One evidence set per claim, matching earlier baselines.
    df = (
        df.sort_values(["claim_id", "evidence_set_id"])
        .drop_duplicates("claim_id")
        .reset_index(drop=True)
    )

    if df[["claim", "evidence", "wiki_titles", "label"]].isna().any().any():
        raise ValueError(f"Missing values in {name}")

    return df


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    train = load_split("train")
    dev = load_split("dev")

    if not set(train.claim_id).isdisjoint(set(dev.claim_id)):
        raise ValueError("Train/Dev claim ID overlap!")

    print("=== E05a: OWN CONTEXT-AWARE MODEL ===")
    print("Train claims:", len(train))
    print("Dev claims:", len(dev), flush=True)

    # Learn vocabulary exclusively from training data.
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
        dtype=np.float32,
    )

    train_text = (
        train["claim"].astype(str)
        + " "
        + train["context_evidence"].astype(str)
    )

    dev_text = (
        dev["claim"].astype(str)
        + " "
        + dev["context_evidence"].astype(str)
    )

    print("Fitting TF-IDF...", flush=True)
    X_train_text = vectorizer.fit_transform(train_text)
    X_dev_text = vectorizer.transform(dev_text)

    print("Calculating relationship features...", flush=True)
    train_rel = relationship_features(train)
    dev_rel = relationship_features(dev)

    scaler = StandardScaler()
    X_train_rel = scaler.fit_transform(train_rel)
    X_dev_rel = scaler.transform(dev_rel)

    X_train = hstack(
        [X_train_text, csr_matrix(X_train_rel)],
        format="csr",
    )
    X_dev = hstack(
        [X_dev_text, csr_matrix(X_dev_rel)],
        format="csr",
    )

    print("Training Logistic Regression...", flush=True)
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )

    model.fit(X_train, train["label"])

    predictions = model.predict(X_dev)

    accuracy = accuracy_score(dev["label"], predictions)
    macro_f1 = f1_score(
        dev["label"],
        predictions,
        labels=LABELS,
        average="macro",
    )

    print("\n=== E05a RESULTS ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nClassification report:")
    print(classification_report(
        dev["label"], predictions, labels=LABELS
    ))

    print("Confusion matrix:")
    print(confusion_matrix(
        dev["label"], predictions, labels=LABELS
    ))

    dev["prediction"] = predictions
    dev["correct"] = dev["label"] == predictions

    results_file = OUTPUT / "e05a_context_model_results.csv"
    dev.to_csv(results_file, index=False)

    model_file = MODELS / "fever_context_logreg.joblib"
    dump({
        "model": model,
        "vectorizer": vectorizer,
        "scaler": scaler,
        "labels": LABELS,
        "feature_version": "e05a",
    }, model_file)

    print("\nSaved results:", results_file)
    print("Saved model:", model_file)


if __name__ == "__main__":
    main()
