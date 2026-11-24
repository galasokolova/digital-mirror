from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.src.models.train_fever_relationship_model import (
    make_features,
    relationship_features,
)

MODEL_PATH = Path("ml/evidence_classifier/models/fever_relationship_logreg.joblib")
OUTPUT_PATH = Path("ml/evidence_classifier/data/experiments/e12_negation_variants.csv")

CLAIM = "NATO provided military assistance to Ukraine."
TITLE = "NATO and Ukraine relations"

VARIANTS = [
    ("SUPPORT", "NATO provided military assistance to Ukraine.", "SUPPORTS"),
    ("DID_SUPPORT", "NATO did provide military assistance to Ukraine.", "SUPPORTS"),
    ("DID_NOT", "NATO did not provide military assistance to Ukraine.", "REFUTES"),
    ("NEVER", "NATO never provided military assistance to Ukraine.", "REFUTES"),
    ("NO", "No military assistance was provided to Ukraine by NATO.", "REFUTES"),
]


def main():
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]

    df = pd.DataFrame(
        [
            {
                "variant": variant,
                "claim": CLAIM,
                "evidence": evidence,
                "expected": expected,
                "wiki_titles": TITLE,
            }
            for variant, evidence, expected in VARIANTS
        ]
    )

    df["context_evidence"] = (
        "Wikipedia article: "
        + df["wiki_titles"]
        + ". "
        + df["evidence"]
    )

    X = make_features(
        df,
        saved["claim_vectorizer"],
        saved["evidence_vectorizer"],
        saved["similarity_vectorizer"],
        saved["scaler"],
        fit=False,
    )

    predictions = model.predict(X)

    supports_index = list(model.classes_).index("SUPPORTS")
    probabilities = model.predict_proba(X)[:, supports_index]

    similarity_vectorizer = saved["similarity_vectorizer"]

    claim_vectors = similarity_vectorizer.transform(df["claim"])
    evidence_vectors = similarity_vectorizer.transform(df["evidence"])

    relational = relationship_features(
        df,
        claim_vectors,
        evidence_vectors,
    )

    df["prediction"] = predictions
    df["p_supports"] = probabilities
    df["negation_mismatch"] = relational[:, 3]
    df["claim_overlap"] = relational[:, 0]
    df["tfidf_similarity"] = relational[:, 6]
    df["correct"] = df["prediction"] == df["expected"]

    baseline = float(
        df.loc[df["variant"] == "DID_SUPPORT", "p_supports"].iloc[0]
    )

    df["delta_vs_did_support"] = df["p_supports"] - baseline

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== E12: NEGATION VARIANTS ===")

    columns = [
        "variant",
        "expected",
        "prediction",
        "p_supports",
        "negation_mismatch",
        "claim_overlap",
        "tfidf_similarity",
        "delta_vs_did_support",
    ]

    print(
        df[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n=== SUMMARY ===")
    print("Correct predictions:", int(df["correct"].sum()), "/", len(df))

    print("\n=== CONTROLLED COMPARISON ===")
    for variant in ["DID_SUPPORT", "DID_NOT"]:
        row = df[df["variant"] == variant].iloc[0]
        print(
            f"{variant}: "
            f"P(SUPPORTS)={row['p_supports']:.4f}, "
            f"negation_mismatch={row['negation_mismatch']:.0f}"
        )

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
