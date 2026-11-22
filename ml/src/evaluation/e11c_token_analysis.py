from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.src.models.train_fever_relationship_model import make_features


MODEL_PATH = Path("ml/models/fever_relationship_logreg.joblib")
INPUT_PATH = Path("ml/data/experiments/e10b_negation_ablation_results.csv")
OUTPUT_PATH = Path("ml/data/experiments/e11c_token_contributions.csv")

TOP_N = 5


def main():
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    df = pd.read_csv(INPUT_PATH)

    required = ["pair_id", "variant", "claim", "evidence", "wiki_titles"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["context_evidence"] = (
        "Wikipedia article: "
        + df["wiki_titles"].astype(str)
        + ". "
        + df["evidence"].astype(str)
    )

    claim_vec = saved["claim_vectorizer"]
    evidence_vec = saved["evidence_vectorizer"]

    X = make_features(
        df,
        claim_vec,
        evidence_vec,
        saved["similarity_vectorizer"],
        saved["scaler"],
        fit=False,
    )

    if list(model.classes_) not in (
        ["REFUTES", "SUPPORTS"],
        ["SUPPORTS", "REFUTES"],
    ):
        raise ValueError(f"Unexpected classes: {model.classes_}")

    sign = 1 if model.classes_[1] == "SUPPORTS" else -1
    weights = sign * model.coef_[0]

    claim_names = claim_vec.get_feature_names_out()
    evidence_names = evidence_vec.get_feature_names_out()

    n_claim = len(claim_names)
    n_evidence = len(evidence_names)

    evidence_start = n_claim
    evidence_end = n_claim + n_evidence

    evidence_weights = weights[evidence_start:evidence_end]
    evidence_matrix = X[:, evidence_start:evidence_end].tocsr()

    records = []
    contributions_by_case = {}

    for i, row in df.iterrows():
        vector = evidence_matrix.getrow(i)
        contributions = np.zeros(n_evidence)

        contributions[vector.indices] = (
            vector.data * evidence_weights[vector.indices]
        )

        key = (row["pair_id"], row["variant"])
        contributions_by_case[key] = contributions

        active = np.flatnonzero(contributions)

        positive = sorted(
            (j for j in active if contributions[j] > 0),
            key=lambda j: contributions[j],
            reverse=True,
        )[:TOP_N]

        negative = sorted(
            (j for j in active if contributions[j] < 0),
            key=lambda j: contributions[j],
        )[:TOP_N]

        print(f"\n=== {row['pair_id']} | {row['variant']} ===")
        print(f"Evidence: {row['evidence']}")

        print("\nTop SUPPORTS features:")
        for j in positive:
            print(f"  {evidence_names[j]:35s} {contributions[j]:+.4f}")

        print("Top REFUTES features:")
        for j in negative:
            print(f"  {evidence_names[j]:35s} {contributions[j]:+.4f}")

        for j in active:
            records.append({
                "pair_id": row["pair_id"],
                "variant": row["variant"],
                "feature": evidence_names[j],
                "tfidf_value": float(vector[0, j]),
                "weight": float(evidence_weights[j]),
                "contribution": float(contributions[j]),
            })

    print("\n\n=== LARGEST CHANGES AFTER NEGATION ===")

    for pair_id in df["pair_id"].unique():
        support_key = (pair_id, "SUPPORT")
        negated_key = (pair_id, "NEGATED")

        if support_key not in contributions_by_case:
            continue
        if negated_key not in contributions_by_case:
            continue

        support = contributions_by_case[support_key]
        negated = contributions_by_case[negated_key]

        delta = negated - support

        top_changes = np.argsort(np.abs(delta))[-TOP_N:][::-1]

        print(f"\n{pair_id}")
        print(f"Total Evidence TF-IDF change: {delta.sum():+.4f}")

        for j in top_changes:
            if abs(delta[j]) > 1e-10:
                print(
                    f"  {evidence_names[j]:35s} "
                    f"{delta[j]:+.4f}"
                )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
