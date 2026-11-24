from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = Path(__file__).resolve().parents[3]

TRAIN_FILE = (
    BASE_DIR / "data" / "processed"
    / "fever" / "train_verified.csv"
)

RESULTS_FILE = (
    BASE_DIR / "data" / "experiments"
    / "fever" / "e03_context_results.csv"
)

OUTPUT_FILE = (
    BASE_DIR / "data" / "experiments"
    / "fever" / "e04_model_comparison.csv"
)

LABELS = ["REFUTES", "SUPPORTS"]


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
                random_state=42,
            ),
        ),
    ])


def claim_only(df):
    return df["claim"].astype(str)


def claim_evidence(df):
    return (
        "CLAIM: " + df["claim"].astype(str)
        + " EVIDENCE: " + df["evidence"].astype(str)
    )


def evaluate(name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)

    f1 = f1_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    print(
        f"{name:30s} "
        f"Accuracy={accuracy:.4f} "
        f"Macro F1={f1:.4f}"
    )


def main():
    print("Loading datasets...", flush=True)

    train = pd.read_csv(TRAIN_FILE)

    train = (
        train.sort_values(["claim_id", "evidence_set_id"])
        .drop_duplicates(subset=["claim_id"])
    )

    dev = pd.read_csv(RESULTS_FILE)

    assert set(train["claim_id"]).isdisjoint(
        set(dev["claim_id"])
    ), "Train/dev claim ID overlap!"

    print("Train claims:", len(train))
    print("Evaluation claims:", len(dev))

    print("\nTraining TF-IDF Claim Only...", flush=True)

    model_a = create_model()
    model_a.fit(claim_only(train), train["label"])

    dev["tfidf_claim_prediction"] = model_a.predict(
        claim_only(dev)
    )

    print("Training TF-IDF Claim + Evidence...", flush=True)

    model_b = create_model()
    model_b.fit(claim_evidence(train), train["label"])

    dev["tfidf_evidence_prediction"] = model_b.predict(
        claim_evidence(dev)
    )

    print("\n=== E04 FAIR COMPARISON ===")

    evaluate(
        "TF-IDF Claim Only",
        dev["label"],
        dev["tfidf_claim_prediction"],
    )

    evaluate(
        "TF-IDF Claim + Evidence",
        dev["label"],
        dev["tfidf_evidence_prediction"],
    )

    evaluate(
        "DeBERTa Original Evidence",
        dev["label"],
        dev["prediction"],
    )

    evaluate(
        "DeBERTa Context Evidence",
        dev["label"],
        dev["context_prediction"],
    )

    dev.to_csv(OUTPUT_FILE, index=False)

    print("\nSaved:", OUTPUT_FILE)
    print("E04 comparison completed.")


if __name__ == "__main__":
    main()
