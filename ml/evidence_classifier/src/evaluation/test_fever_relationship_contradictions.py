from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.src.models.train_fever_relationship_model import make_features
from ml.src.experiments.fever_baselines.test_controlled_contradictions import TEST_CASES


BASE = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    BASE / "models" / "fever_relationship_logreg.joblib"
)

OUTPUT = (
    BASE / "data" / "experiments" / "fever"
)


def main():
    saved = joblib.load(MODEL_PATH)

    model = saved["model"]
    claim_vectorizer = saved["claim_vectorizer"]
    evidence_vectorizer = saved["evidence_vectorizer"]
    similarity_vectorizer = saved["similarity_vectorizer"]
    scaler = saved["scaler"]

    rows = []

    for case in TEST_CASES:
        for expected, evidence in [
            ("SUPPORTS", case["supports"]),
            ("REFUTES", case["refutes"]),
        ]:
            rows.append({
                "claim": case["claim"],
                "wiki_titles": case["wiki_titles"],
                "evidence": evidence,
                "context_evidence": (
                    f"Wikipedia article: "
                    f"{case['wiki_titles']}. {evidence}"
                ),
                "expected": expected,
            })

    df = pd.DataFrame(rows)

    X = make_features(
        df,
        claim_vectorizer,
        evidence_vectorizer,
        similarity_vectorizer,
        scaler,
        fit=False,
    )

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    supports_index = list(model.classes_).index("SUPPORTS")

    df["prediction"] = predictions
    df["p_supports"] = probabilities[:, supports_index]
    df["correct"] = df["prediction"] == df["expected"]

    changed = 0
    correct_pairs = 0

    for i in range(0, len(df), 2):
        support = df.iloc[i]
        refute = df.iloc[i + 1]

        pair_changed = (
            support["prediction"] != refute["prediction"]
        )

        pair_correct = (
            support["correct"] and refute["correct"]
        )

        changed += int(pair_changed)
        correct_pairs += int(pair_correct)

        print("\nCLAIM:", support["claim"])

        print("SUPPORTING EVIDENCE:", support["evidence"])
        print(
            f"  Prediction: {support['prediction']}"
            f" | P(SUPPORTS): {support['p_supports']:.4f}"
        )

        print("REFUTING EVIDENCE:", refute["evidence"])
        print(
            f"  Prediction: {refute['prediction']}"
            f" | P(SUPPORTS): {refute['p_supports']:.4f}"
        )

        difference = (
            support["p_supports"] - refute["p_supports"]
        )

        print(f"  Probability difference: {difference:+.4f}")
        print("  Both correct:", bool(pair_correct))

    print("\n=== E06c SUMMARY ===")
    print(f"Correct predictions: {df['correct'].sum()}/{len(df)}")
    print(f"Changed class between evidence pairs: {changed}/{len(TEST_CASES)}")
    print(f"Both predictions correct: {correct_pairs}/{len(TEST_CASES)}")

    OUTPUT.mkdir(parents=True, exist_ok=True)

    path = OUTPUT / "e06c_controlled_contradictions.csv"
    df.to_csv(path, index=False)

    print("\nSaved:", path)


if __name__ == "__main__":
    main()
