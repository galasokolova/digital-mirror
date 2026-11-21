from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from ml.src.models.train_fever_relationship_model import (
    load_split,
    relationship_features,
)

BASE = Path(__file__).resolve().parents[3]
OUTPUT = BASE / "data" / "experiments" / "fever"

FEATURE_NAMES = [
    "claim_overlap",
    "jaccard_overlap",
    "title_overlap",
    "negation_mismatch",
    "number_overlap",
    "number_mismatch",
    "cosine_similarity",
]

EXPERIMENTS = {
    "Baseline (TF-IDF only)": [],
    "Full (all features)": list(range(7)),
    "No Cosine": [0, 1, 2, 3, 4, 5],
    "No Numbers": [0, 1, 2, 3, 6],
    "No Negation": [0, 1, 2, 4, 5, 6],
    "No Overlap": [2, 3, 4, 5, 6],
    "No Title Feature": [0, 1, 3, 4, 5, 6],
}


def evaluate(name, indices, X_train_text, X_dev_text,
             train_rel, dev_rel, y_train, y_dev):

    if indices:
        scaler = StandardScaler()

        X_train_rel = scaler.fit_transform(
            train_rel[:, indices]
        )
        X_dev_rel = scaler.transform(
            dev_rel[:, indices]
        )

        X_train = hstack(
            [X_train_text, csr_matrix(X_train_rel)],
            format="csr",
        )
        X_dev = hstack(
            [X_dev_text, csr_matrix(X_dev_rel)],
            format="csr",
        )
    else:
        X_train = X_train_text
        X_dev = X_dev_text

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_dev)

    accuracy = accuracy_score(y_dev, predictions)
    macro_f1 = f1_score(
        y_dev, predictions, average="macro"
    )

    print(
        f"{name:28} "
        f"Accuracy={accuracy:.4f} "
        f"Macro F1={macro_f1:.4f}",
        flush=True,
    )

    return {
        "experiment": name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "features": ", ".join(
            FEATURE_NAMES[i] for i in indices
        ),
    }


def main():
    train = load_split("train")
    dev = load_split("dev")

    if not set(train.claim_id).isdisjoint(set(dev.claim_id)):
        raise ValueError("Train/Dev claim ID overlap")

    params = dict(
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
        dtype=np.float32,
    )

    print("Fitting vectorizers...", flush=True)

    claim_vectorizer = TfidfVectorizer(**params)
    evidence_vectorizer = TfidfVectorizer(**params)
    similarity_vectorizer = TfidfVectorizer(**params)

    X_train_claim = claim_vectorizer.fit_transform(
        train["claim"]
    )
    X_dev_claim = claim_vectorizer.transform(
        dev["claim"]
    )

    X_train_evidence = evidence_vectorizer.fit_transform(
        train["context_evidence"]
    )
    X_dev_evidence = evidence_vectorizer.transform(
        dev["context_evidence"]
    )

    similarity_vectorizer.fit(
        pd.concat(
            [train["claim"], train["evidence"]],
            ignore_index=True,
        )
    )

    X_train_sim_claim = similarity_vectorizer.transform(
        train["claim"]
    )
    X_train_sim_evidence = similarity_vectorizer.transform(
        train["evidence"]
    )

    X_dev_sim_claim = similarity_vectorizer.transform(
        dev["claim"]
    )
    X_dev_sim_evidence = similarity_vectorizer.transform(
        dev["evidence"]
    )

    print("Calculating relationship features...", flush=True)

    train_rel = relationship_features(
        train, X_train_sim_claim, X_train_sim_evidence
    )

    dev_rel = relationship_features(
        dev, X_dev_sim_claim, X_dev_sim_evidence
    )

    X_train_text = hstack(
        [X_train_claim, X_train_evidence],
        format="csr",
    )

    X_dev_text = hstack(
        [X_dev_claim, X_dev_evidence],
        format="csr",
    )

    print("\n=== E06d: FEATURE ABLATION ===", flush=True)

    results = []

    for name, indices in EXPERIMENTS.items():
        results.append(evaluate(
            name,
            indices,
            X_train_text,
            X_dev_text,
            train_rel,
            dev_rel,
            train["label"],
            dev["label"],
        ))

    OUTPUT.mkdir(parents=True, exist_ok=True)

    path = OUTPUT / "e06d_feature_ablation.csv"
    pd.DataFrame(results).to_csv(path, index=False)

    print("\nSaved:", path)


if __name__ == "__main__":
    main()
