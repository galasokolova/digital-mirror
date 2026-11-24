from pathlib import Path

import joblib
import pandas as pd

from ml.src.models.train_fever_relationship_model import make_features


MODEL_PATH = Path("ml/evidence_classifier/models/fever_relationship_logreg.joblib")
OUTPUT_PATH = Path("ml/evidence_classifier/data/experiments/e10a_amnesty_results.csv")

claim = (
    "Amnesty International admits that Ukraine fights "
    "like terrorists and uses civilians as human shields"
)

test_cases = [
    {
        "test": "ORIGINAL_MIXED",
        "expected": "REFUTES",
        "evidence": (
            "Amnesty International reported that Ukrainian "
            "forces operated in populated residential areas "
            "and endangered civilians. The report did not "
            "characterize Ukrainian military tactics as "
            "terrorism or explicitly accuse Ukraine of "
            "deliberately using civilians as human shields."
        ),
    },
    {
        "test": "DIRECT_REFUTATION",
        "expected": "REFUTES",
        "evidence": (
            "Amnesty International did not describe Ukrainian "
            "forces as terrorists and did not accuse Ukraine "
            "of using civilians as human shields."
        ),
    },
    {
        "test": "SYNTHETIC_SUPPORT",
        "expected": "SUPPORTS",
        "evidence": (
            "Amnesty International explicitly stated that "
            "Ukrainian forces fight like terrorists and "
            "deliberately use civilians as human shields."
        ),
    },
]

rows = []

for case in test_cases:
    evidence = case["evidence"]

    rows.append({
        "test": case["test"],
        "claim": claim,
        "evidence": evidence,
        "wiki_titles": "Amnesty International Ukraine",
        "context_evidence": (
            "Wikipedia article: Amnesty International Ukraine. "
            + evidence
        ),
        "expected": case["expected"],
    })

df = pd.DataFrame(rows)

saved = joblib.load(MODEL_PATH)
model = saved["model"]

X = make_features(
    df,
    saved["claim_vectorizer"],
    saved["evidence_vectorizer"],
    saved["similarity_vectorizer"],
    saved["scaler"],
    fit=False,
)

predictions = model.predict(X)
probabilities = model.predict_proba(X)

supports_idx = list(model.classes_).index("SUPPORTS")
refutes_idx = list(model.classes_).index("REFUTES")

df["prediction"] = predictions
df["p_supports"] = probabilities[:, supports_idx]
df["p_refutes"] = probabilities[:, refutes_idx]
df["correct"] = df["prediction"] == df["expected"]

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

df.drop(columns=["context_evidence"]).to_csv(
    OUTPUT_PATH,
    index=False,
)

print("\n=== E10a: AMNESTY EVIDENCE SENSITIVITY ===")

for _, row in df.iterrows():
    print("\n" + "=" * 65)
    print("TEST:", row["test"])
    print("EVIDENCE:", row["evidence"])
    print("EXPECTED:", row["expected"])
    print("PREDICTION:", row["prediction"])
    print(f"P(SUPPORTS): {row['p_supports']:.4f}")
    print(f"P(REFUTES):  {row['p_refutes']:.4f}")
    print("CORRECT:", row["correct"])

print("\n=== SUMMARY ===")

print("Correct:", int(df["correct"].sum()), "/", len(df))

print(
    "Direct refutation P(SUPPORTS):",
    round(float(df.loc[1, "p_supports"]), 4),
)

print(
    "Synthetic support P(SUPPORTS):",
    round(float(df.loc[2, "p_supports"]), 4),
)

print(
    "Support minus refutation:",
    round(
        float(df.loc[2, "p_supports"] - df.loc[1, "p_supports"]),
        4,
    ),
)

print(
    "Predictions changed:",
    df.loc[1, "prediction"] != df.loc[2, "prediction"],
)

print("Saved:", OUTPUT_PATH)
