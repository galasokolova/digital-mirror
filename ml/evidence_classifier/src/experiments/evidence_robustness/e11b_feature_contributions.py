from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.evidence_classifier.src.models.train_fever_relationship_model import make_features


MODEL_PATH = Path("ml/evidence_classifier/models/fever_relationship_logreg.joblib")
INPUT_PATH = Path(
    "ml/evidence_classifier/data/experiments/e10b_negation_ablation_results.csv"
)
OUTPUT_PATH = Path(
    "ml/evidence_classifier/data/experiments/e11b_feature_contributions.csv"
)

FEATURE_NAMES = [
    "claim_overlap",
    "jaccard_similarity",
    "title_overlap",
    "negation_mismatch",
    "shared_numbers",
    "different_numbers",
    "tfidf_cosine_similarity",
]


def main():
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    df = pd.read_csv(INPUT_PATH)

    # Reconstruct the same context used in E10b.
    df["context_evidence"] = (
        "Wikipedia article: "
        + df["wiki_titles"].astype(str)
        + ". "
        + df["evidence"].astype(str)
    )
    
    # Restore the context used during E10b.
    # This column was excluded when saving E10b results to CSV.
    df["context_evidence"] = (
        "Wikipedia article: "
        + df["wiki_titles"].astype(str)
        + ". "
        + df["evidence"].astype(str)
    )

    X = make_features(
        df,
        saved["claim_vectorizer"],
        saved["evidence_vectorizer"],
        saved["similarity_vectorizer"],
        saved["scaler"],
        fit=False,
    )

    n_claim = len(
        saved["claim_vectorizer"].get_feature_names_out()
    )
    n_evidence = len(
        saved["evidence_vectorizer"].get_feature_names_out()
    )

    start_rel = n_claim + n_evidence

    assert X.shape[1] == start_rel + len(FEATURE_NAMES)

    if len(model.classes_) != 2:
        raise ValueError("Expected binary Logistic Regression")

    # In sklearn binary LogisticRegression, coef_[0]
    # represents the log-odds of classes_[1].
    sign = 1 if model.classes_[1] == "SUPPORTS" else -1

    weights = sign * model.coef_[0]
    intercept = sign * model.intercept_[0]

    contributions = X.multiply(weights).tocsr()

    claim_contrib = np.asarray(
        contributions[:, :n_claim].sum(axis=1)
    ).ravel()

    evidence_contrib = np.asarray(
        contributions[:, n_claim:start_rel].sum(axis=1)
    ).ravel()

    rel_contrib = contributions[
        :, start_rel:
    ].toarray()

    result = df[
        ["pair_id", "variant", "prediction", "p_supports"]
    ].copy()

    result["intercept"] = intercept
    result["claim_tfidf"] = claim_contrib
    result["evidence_tfidf"] = evidence_contrib

    for i, name in enumerate(FEATURE_NAMES):
        result[name] = rel_contrib[:, i]

    result["total_log_odds"] = (
        intercept
        + claim_contrib
        + evidence_contrib
        + rel_contrib.sum(axis=1)
    )

    calculated_probability = (
        1 / (1 + np.exp(-result["total_log_odds"]))
    )

    if not np.allclose(
        calculated_probability,
        result["p_supports"],
        atol=1e-4,
    ):
        raise ValueError(
            "Feature contributions do not match predictions"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(OUTPUT_PATH, index=False)

    print("\n=== E11b: FEATURE CONTRIBUTIONS ===")
    print("Positive contribution -> SUPPORTS")
    print("Negative contribution -> REFUTES")

    columns = [
        "pair_id",
        "variant",
        "p_supports",
        "claim_tfidf",
        "evidence_tfidf",
        "negation_mismatch",
        "claim_overlap",
        "jaccard_similarity",
        "tfidf_cosine_similarity",
        "total_log_odds",
    ]

    print(
        result[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:+.4f}",
        )
    )

    print("\n=== NEGATION CONTRIBUTION ===")

    negated = result[
        result["variant"] == "NEGATED"
    ]

    print(
        "Mean negation contribution:",
        round(
            float(negated["negation_mismatch"].mean()),
            4,
        ),
    )

    print(
        "Mean claim TF-IDF contribution:",
        round(
            float(negated["claim_tfidf"].mean()),
            4,
        ),
    )

    print(
        "Mean evidence TF-IDF contribution:",
        round(
            float(negated["evidence_tfidf"].mean()),
            4,
        ),
    )

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
