from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.src.models.train_fever_relationship_model import make_features


BASE = Path(__file__).resolve().parents[3]

MODEL_PATH = BASE / "models/fever_relationship_logreg.joblib"
INPUT_PATH = BASE / "data/experiments/fever/e07c_audited_pairs.csv"
OUTPUT_PATH = BASE / "data/experiments/fever/e08_negation_results.csv"


def main():
    saved = joblib.load(MODEL_PATH)

    model = saved["model"]

    source = pd.read_csv(INPUT_PATH)

    mask = source["added_tokens"].fillna("").str.contains(
        r"\b(?:not|never)\b",
        case=False,
        regex=True,
    )

    pairs = source.loc[mask].copy()

    rows = []

    for pair_id, (_, pair) in enumerate(pairs.iterrows(), 1):
        # Use the same evidence for both claims.
        evidence = pair["supports_evidence"]

        # IMPORTANT:
        # wiki_titles must come from the actual dataset.
        # Do not invent article titles.
        wiki_titles = pair["supports_titles"]

        if pd.isna(wiki_titles):
            raise ValueError(
                f"Missing Wikipedia title for pair {pair_id}"
                )

        for expected, claim in [
            ("SUPPORTS", pair["supports_claim"]),
            ("REFUTES", pair["refutes_claim"]),
        ]:
            rows.append({
                "pair_id": pair_id,
                "claim": claim,
                "evidence": evidence,
                "wiki_titles": str(wiki_titles),
                "context_evidence": (
                    f"Wikipedia article: {wiki_titles}. {evidence}"
                ),
                "expected": expected,
            })

    df = pd.DataFrame(rows)

    X = make_features(
        df,
        saved["claim_vectorizer"],
        saved["evidence_vectorizer"],
        saved["similarity_vectorizer"],
        saved["scaler"],
        fit=False,
    )

    df["prediction"] = model.predict(X)

    probabilities = model.predict_proba(X)
    supports_index = list(model.classes_).index("SUPPORTS")

    df["p_supports"] = probabilities[:, supports_index]
    df["correct"] = df["prediction"] == df["expected"]

    changed = 0
    both_correct = 0

    for pair_id, group in df.groupby("pair_id"):
        a, b = group.iloc[0], group.iloc[1]

        changed += int(a["prediction"] != b["prediction"])
        both_correct += int(a["correct"] and b["correct"])

        print(f"\nPAIR {pair_id}")
        print("SUPPORTS CLAIM:", a["claim"])
        print(
            f"  Predicted: {a['prediction']}"
            f" | P(SUPPORTS): {a['p_supports']:.4f}"
        )

        print("REFUTES CLAIM:", b["claim"])
        print(
            f"  Predicted: {b['prediction']}"
            f" | P(SUPPORTS): {b['p_supports']:.4f}"
        )

        print(
            "Probability shift:",
            round(float(a["p_supports"] - b["p_supports"]), 4),
        )

        print("Both correct:", bool(a["correct"] and b["correct"]))

    total = df["pair_id"].nunique()

    print("\n=== E08 NEGATION TEST ===")
    print(f"Individual accuracy: {df['correct'].mean():.4f}")
    print(f"Changed prediction: {changed}/{total}")
    print(f"Both correct: {both_correct}/{total}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
