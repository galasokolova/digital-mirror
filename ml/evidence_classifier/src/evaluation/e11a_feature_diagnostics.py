from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.src.models.train_fever_relationship_model import (
    relationship_features,
)

MODEL_PATH = Path("ml/evidence_classifier/models/fever_relationship_logreg.joblib")
INPUT_PATH = Path(
    "ml/evidence_classifier/data/experiments/e10b_negation_ablation_results.csv"
)
OUTPUT_PATH = Path(
    "ml/evidence_classifier/data/experiments/e11a_feature_diagnostics.csv"
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
    df = pd.read_csv(INPUT_PATH)

    vectorizer = saved["similarity_vectorizer"]

    claim_vectors = vectorizer.transform(df["claim"])
    evidence_vectors = vectorizer.transform(df["evidence"])

    raw_features = relationship_features(
        df,
        claim_vectors,
        evidence_vectors,
    )

    if raw_features.shape[1] != len(FEATURE_NAMES):
        raise ValueError("Unexpected number of features")

    feature_df = pd.DataFrame(
        raw_features,
        columns=FEATURE_NAMES,
    )

    result = pd.concat(
        [
            df[
                [
                    "pair_id",
                    "variant",
                    "expected",
                    "prediction",
                    "p_supports",
                ]
            ].reset_index(drop=True),
            feature_df,
        ],
        axis=1,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(OUTPUT_PATH, index=False)

    print("\n=== E11a: FEATURE DIAGNOSTICS ===")

    columns = [
        "pair_id",
        "variant",
        "prediction",
        "p_supports",
        "claim_overlap",
        "jaccard_similarity",
        "negation_mismatch",
        "tfidf_cosine_similarity",
    ]

    print(
        result[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n=== NEGATION FEATURE CHECK ===")

    negated = result[
        result["variant"] == "NEGATED"
    ]

    print(
        "Negated cases with mismatch = 1:",
        int((negated["negation_mismatch"] == 1).sum()),
        "/",
        len(negated),
    )

    print("\n=== FEATURE DIFFERENCES ===")

    for pair_id, group in result.groupby("pair_id", sort=False):
        support = group[
            group["variant"] == "SUPPORT"
        ].iloc[0]

        negated = group[
            group["variant"] == "NEGATED"
        ].iloc[0]

        print(f"\n{pair_id}")

        for name in FEATURE_NAMES:
            difference = (
                float(negated[name])
                - float(support[name])
            )

            print(f"  {name}: {difference:+.4f}")

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
