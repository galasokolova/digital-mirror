from pathlib import Path

import joblib
import pandas as pd

from ml.src.models.train_fever_relationship_model import (
    make_features,
    relationship_features,
)

MODEL_PATH = Path("ml/evidence_classifier/models/fever_relationship_logreg.joblib")
OUTPUT_PATH = Path(
    "ml/evidence_classifier/data/experiments/e13_negation_generalization.csv"
)

CLAIM = "The committee approved the proposal."
TITLE = "Committee decision"

CASES = [
    (
        "DIRECT_SUPPORT",
        "The committee approved the proposal.",
        "SUPPORTS",
    ),
    (
        "DIRECT_REFUTE",
        "The committee did not approve the proposal.",
        "REFUTES",
    ),
    (
        "NEVER_REFUTE",
        "The committee never approved the proposal.",
        "REFUTES",
    ),
    (
        "NO_REFUTE",
        "No approval was given to the proposal by the committee.",
        "REFUTES",
    ),
    (
        "NEGATION_ELSEWHERE",
        "The committee approved the proposal, not the amendment.",
        "SUPPORTS",
    ),
    (
        "DOUBLE_NEGATION",
        "The committee did not fail to approve the proposal.",
        "SUPPORTS",
    ),
    (
        "PARAPHRASE_SUPPORT",
        "The committee accepted the proposal.",
        "SUPPORTS",
    ),
    (
        "PARAPHRASE_REFUTE",
        "The committee rejected the proposal.",
        "REFUTES",
    ),
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
            for variant, evidence, expected in CASES
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

    supports_index = list(model.classes_).index("SUPPORTS")

    df["prediction"] = model.predict(X)
    df["p_supports"] = model.predict_proba(X)[:, supports_index]

    sim_vec = saved["similarity_vectorizer"]

    relational = relationship_features(
        df,
        sim_vec.transform(df["claim"]),
        sim_vec.transform(df["evidence"]),
    )

    df["negation_mismatch"] = relational[:, 3]
    df["claim_overlap"] = relational[:, 0]
    df["tfidf_similarity"] = relational[:, 6]

    df["correct"] = df["prediction"] == df["expected"]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== E13: NEGATION GENERALIZATION ===")

    columns = [
        "variant",
        "expected",
        "prediction",
        "p_supports",
        "negation_mismatch",
        "claim_overlap",
        "tfidf_similarity",
        "correct",
    ]

    print(
        df[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n=== SUMMARY ===")
    print(
        "Correct predictions:",
        int(df["correct"].sum()),
        "/",
        len(df),
    )

    print("\n=== SUPPORTS WITH NEGATION ===")
    print(
        df[
            (df["expected"] == "SUPPORTS")
            & (df["negation_mismatch"] == 1)
        ][
            ["variant", "prediction", "p_supports"]
        ].to_string(index=False)
    )

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
