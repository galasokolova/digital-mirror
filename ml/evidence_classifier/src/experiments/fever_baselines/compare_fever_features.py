from pathlib import Path

import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

BASE = Path(__file__).resolve().parents[3]
DATA = BASE / "data" / "processed" / "fever"
OUTPUT = BASE / "data" / "experiments" / "fever"

def load_data(split):
    df = pd.read_csv(DATA / f"{split}_context.csv")
    return (
        df.sort_values(["claim_id", "evidence_set_id"])
        .drop_duplicates("claim_id")
        .reset_index(drop=True)
    )

def evaluate(name, X_train, X_dev, y_train, y_dev):
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_dev)

    accuracy = accuracy_score(y_dev, predictions)
    f1 = f1_score(y_dev, predictions, average="macro")

    print(f"{name:30} Accuracy={accuracy:.4f} Macro F1={f1:.4f}")
    return {"model": name, "accuracy": accuracy, "macro_f1": f1}


def main():
    train = load_data("train")
    dev = load_data("dev")

    assert set(train.claim_id).isdisjoint(set(dev.claim_id))

    y_train = train["label"]
    y_dev = dev["label"]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
        dtype=np.float32,
    )

    results = []

    # A: Claim only
    print("Training A...", flush=True)
    X_train = vectorizer.fit_transform(train["claim"])
    X_dev = vectorizer.transform(dev["claim"])
    results.append(evaluate(
        "A: Claim Only", X_train, X_dev, y_train, y_dev
    ))

    # B: Claim + Evidence combined
    print("Training B...", flush=True)
    train_text = train["claim"] + " " + train["evidence"]
    dev_text = dev["claim"] + " " + dev["evidence"]

    X_train = vectorizer.fit_transform(train_text)
    X_dev = vectorizer.transform(dev_text)
    results.append(evaluate(
        "B: Combined", X_train, X_dev, y_train, y_dev
    ))

    # C: Claim + Evidence + Title combined
    print("Training C...", flush=True)
    train_text = train["claim"] + " " + train["context_evidence"]
    dev_text = dev["claim"] + " " + dev["context_evidence"]

    X_train = vectorizer.fit_transform(train_text)
    X_dev = vectorizer.transform(dev_text)
    results.append(evaluate(
        "C: Combined + Title", X_train, X_dev, y_train, y_dev
    ))

    # D: Separate claim/evidence feature spaces
    print("Training D...", flush=True)
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

    X_train_claim = claim_vectorizer.fit_transform(train["claim"])
    X_dev_claim = claim_vectorizer.transform(dev["claim"])

    X_train_evidence = evidence_vectorizer.fit_transform(
        train["context_evidence"]
    )
    X_dev_evidence = evidence_vectorizer.transform(
        dev["context_evidence"]
    )

    X_train = hstack(
        [X_train_claim, X_train_evidence], format="csr"
    )
    X_dev = hstack(
        [X_dev_claim, X_dev_evidence], format="csr"
    )

    results.append(evaluate(
        "D: Separate + Title", X_train, X_dev, y_train, y_dev
    ))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "e05b_feature_comparison.csv"
    pd.DataFrame(results).to_csv(path, index=False)
    print("\nSaved:", path)


if __name__ == "__main__":
    main()
