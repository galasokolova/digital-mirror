from pathlib import Path

import joblib
import pandas as pd

from ml.evidence_classifier.src.models.train_fever_relationship_model import make_features


MODEL_PATH = Path("ml/evidence_classifier/models/fever_relationship_logreg.joblib")
OUTPUT_PATH = Path(
    "ml/evidence_classifier/data/experiments/e10b_negation_ablation_results.csv"
)

# Controlled pairs: the evidence changes only by adding "not".
# These are synthetic linguistic examples, NOT verified facts.

cases = [
    (
        "NATO troops are fighting in Ukraine.",
        "NATO troops are fighting in Ukraine.",
        "NATO troops are not fighting in Ukraine.",
        "NATO",
    ),
    (
        "The court declared the conflict international.",
        "The court declared the conflict international.",
        "The court did not declare the conflict international.",
        "Court",
    ),
    (
        "Amnesty International accused Ukraine of using human shields.",
        "Amnesty International accused Ukraine of using human shields.",
        "Amnesty International did not accuse Ukraine of using human shields.",
        "Amnesty International",
    ),
    (
        "Russian forces attacked civilian buildings.",
        "Russian forces attacked civilian buildings.",
        "Russian forces did not attack civilian buildings.",
        "Russian forces",
    ),
    (
        "Ukraine requested nuclear strikes against Russia.",
        "Ukraine requested nuclear strikes against Russia.",
        "Ukraine did not request nuclear strikes against Russia.",
        "Ukraine",
    ),
    (
        "NATO provided military assistance to Ukraine.",
        "NATO provided military assistance to Ukraine.",
        "NATO did not provide military assistance to Ukraine.",
        "NATO",
    ),
    (
        "The report described the attacks as war crimes.",
        "The report described the attacks as war crimes.",
        "The report did not describe the attacks as war crimes.",
        "Report",
    ),
    (
        "The government confirmed the missile attack.",
        "The government confirmed the missile attack.",
        "The government did not confirm the missile attack.",
        "Government",
    ),
]

rows = []

for i, (claim, support, refute, title) in enumerate(cases, 1):
    for variant, evidence, expected in [
        ("SUPPORT", support, "SUPPORTS"),
        ("NEGATED", refute, "REFUTES"),
    ]:
        rows.append({
            "pair_id": f"E10b-{i:03d}",
            "variant": variant,
            "claim": claim,
            "evidence": evidence,
            "wiki_titles": title,
            "context_evidence": (
                f"Wikipedia article: {title}. {evidence}"
            ),
            "expected": expected,
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

df["prediction"] = model.predict(X)

probabilities = model.predict_proba(X)
classes = list(model.classes_)

df["p_supports"] = probabilities[:, classes.index("SUPPORTS")]
df["p_refutes"] = probabilities[:, classes.index("REFUTES")]
df["correct"] = df["prediction"] == df["expected"]

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

df.drop(columns=["context_evidence"]).to_csv(
    OUTPUT_PATH,
    index=False,
)

print("\n=== E10b: NEGATION ABLATION TEST ===")

flips = 0
both_correct = 0
deltas = []

for pair_id, group in df.groupby("pair_id", sort=False):
    support = group[group["variant"] == "SUPPORT"].iloc[0]
    negated = group[group["variant"] == "NEGATED"].iloc[0]

    flipped = support["prediction"] != negated["prediction"]
    correct_pair = bool(support["correct"] and negated["correct"])

    delta = (
        float(support["p_supports"])
        - float(negated["p_supports"])
    )

    flips += int(flipped)
    both_correct += int(correct_pair)
    deltas.append(delta)

    print("\n" + "=" * 65)
    print("PAIR:", pair_id)
    print("CLAIM:", support["claim"])
    print("SUPPORT EVIDENCE:", support["evidence"])
    print("NEGATED EVIDENCE:", negated["evidence"])
    print(
        "SUPPORT:",
        support["prediction"],
        f"P(SUPPORTS)={support['p_supports']:.4f}",
    )
    print(
        "NEGATED:",
        negated["prediction"],
        f"P(SUPPORTS)={negated['p_supports']:.4f}",
    )
    print("DELTA:", round(delta, 4))
    print("PREDICTION FLIPPED:", flipped)
    print("BOTH CORRECT:", correct_pair)

print("\n=== SUMMARY ===")
print("Pairs:", len(cases))
print("Correct predictions:", int(df["correct"].sum()), "/", len(df))
print("Prediction flips:", flips, "/", len(cases))
print("Both correct:", both_correct, "/", len(cases))
print("Mean probability delta:", round(sum(deltas) / len(deltas), 4))
print("Saved:", OUTPUT_PATH)
