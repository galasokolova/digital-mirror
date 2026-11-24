import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import StandardScaler


BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data" / "processed" / "fever"
OUTPUT = BASE / "data" / "experiments" / "fever"
MODELS = BASE / "models"

LABELS = ["REFUTES", "SUPPORTS"]
NEGATIONS = {"not", "no", "never", "without", "neither"}


def tokenize(text):
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def get_numbers(text):
    return set(re.findall(r"\b\d+\b", str(text)))


def relationship_features(df, claim_tfidf, evidence_tfidf):
    features = []

    # Cosine similarity for L2-normalized TF-IDF vectors
    similarities = np.asarray(
        claim_tfidf.multiply(evidence_tfidf).sum(axis=1)
    ).ravel()

    for i, row in enumerate(df.itertuples(index=False)):
        claim = tokenize(row.claim)
        evidence = tokenize(row.evidence)
        title = tokenize(row.wiki_titles)

        overlap = len(claim & evidence)
        union = len(claim | evidence)

        claim_numbers = get_numbers(row.claim)
        evidence_numbers = get_numbers(row.evidence)

        claim_neg = bool(claim & NEGATIONS)
        evidence_neg = bool(evidence & NEGATIONS)

        features.append([
            overlap / max(len(claim), 1),
            overlap / max(union, 1),
            len(claim & title) / max(len(title), 1),
            float(claim_neg != evidence_neg),
            float(bool(claim_numbers & evidence_numbers)),
            float(
                bool(claim_numbers)
                and bool(evidence_numbers)
                and not bool(claim_numbers & evidence_numbers)
            ),
            float(similarities[i]),
        ])

    return np.asarray(features, dtype=np.float32)


def load_split(name):
    df = pd.read_csv(DATA / f"{name}_context.csv")

    df = (
        df.sort_values(["claim_id", "evidence_set_id"])
        .drop_duplicates("claim_id")
        .reset_index(drop=True)
    )

    required = ["claim", "evidence", "wiki_titles", "label"]

    if df[required].isna().any().any():
        raise ValueError(f"Missing values in {name}")

    return df


def make_features(df, claim_vectorizer, evidence_vectorizer,
                  similarity_vectorizer, scaler=None, fit=False):

    X_claim = claim_vectorizer.transform(df["claim"])
    X_evidence = evidence_vectorizer.transform(df["context_evidence"])

    X_sim_claim = similarity_vectorizer.transform(df["claim"])
    X_sim_evidence = similarity_vectorizer.transform(df["evidence"])

    X_rel = relationship_features(
        df, X_sim_claim, X_sim_evidence
    )

    if fit:
        X_rel = scaler.fit_transform(X_rel)
    else:
        X_rel = scaler.transform(X_rel)

    return hstack(
        [X_claim, X_evidence, csr_matrix(X_rel)],
        format="csr",
    )


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    train = load_split("train")
    dev = load_split("dev")

    if not set(train.claim_id).isdisjoint(set(dev.claim_id)):
        raise ValueError("Train/Dev claim ID overlap")

    print("=== E06b: RELATIONSHIP MODEL ===", flush=True)
    print("Train:", len(train))
    print("Dev:", len(dev))

    params = dict(
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
        dtype=np.float32,
    )

    claim_vectorizer = TfidfVectorizer(**params)
    evidence_vectorizer = TfidfVectorizer(**params)
    similarity_vectorizer = TfidfVectorizer(**params)

    print("Fitting vectorizers...", flush=True)

    claim_vectorizer.fit(train["claim"])
    evidence_vectorizer.fit(train["context_evidence"])

    # Shared vocabulary for meaningful cosine similarity
    similarity_vectorizer.fit(
        pd.concat(
            [train["claim"], train["evidence"]],
            ignore_index=True,
        )
    )

    scaler = StandardScaler()

    print("Building training features...", flush=True)
    X_train = make_features(
        train,
        claim_vectorizer,
        evidence_vectorizer,
        similarity_vectorizer,
        scaler,
        fit=True,
    )

    print("Building dev features...", flush=True)
    X_dev = make_features(
        dev,
        claim_vectorizer,
        evidence_vectorizer,
        similarity_vectorizer,
        scaler,
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )

    print("Training...", flush=True)
    model.fit(X_train, train["label"])

    predictions = model.predict(X_dev)

    print("\n=== E06b RESULTS ===")
    print(
        "Accuracy:",
        round(accuracy_score(dev["label"], predictions), 4)
    )
    print(
        "Macro F1:",
        round(f1_score(
            dev["label"], predictions, average="macro"
        ), 4)
    )

    print("\nClassification report:")
    print(classification_report(
        dev["label"], predictions, labels=LABELS
    ))

    print("Confusion matrix:")
    print(confusion_matrix(
        dev["label"], predictions, labels=LABELS
    ))

    results = dev.copy()
    results["prediction"] = predictions

    results.to_csv(
        OUTPUT / "e06b_relationship_results.csv",
        index=False,
    )

    joblib.dump({
        "model": model,
        "claim_vectorizer": claim_vectorizer,
        "evidence_vectorizer": evidence_vectorizer,
        "similarity_vectorizer": similarity_vectorizer,
        "scaler": scaler,
        "feature_version": "e06b",
    }, MODELS / "fever_relationship_logreg.joblib")

    print("\nModel and results saved.")


if __name__ == "__main__":
    main()
