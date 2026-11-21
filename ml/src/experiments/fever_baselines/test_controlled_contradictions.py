from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

# Reuse exactly the same feature extraction
# as during E05a training.
from ml.src.experiments.fever_baselines.train_fever_context_model import relationship_features

BASE = Path(__file__).resolve().parents[3]
MODEL_PATH = BASE / "models" / "fever_context_logreg.joblib"

TEST_CASES = [
    {
        "claim": "Marie Curie was born in Poland.",
        "wiki_titles": "Marie Curie",
        "supports": "Marie Curie was born in Poland.",
        "refutes": "Marie Curie was born in France.",
    },
    {
        "claim": "The Eiffel Tower is located in Paris.",
        "wiki_titles": "Eiffel Tower",
        "supports": "The Eiffel Tower is located in Paris.",
        "refutes": "The Eiffel Tower is located in London.",
    },
    {
        "claim": "Albert Einstein was born in Germany.",
        "wiki_titles": "Albert Einstein",
        "supports": "Albert Einstein was born in Germany.",
        "refutes": "Albert Einstein was born in Canada.",
    },
    {
        "claim": "The film Titanic was released in 1997.",
        "wiki_titles": "Titanic (1997 film)",
        "supports": "The film Titanic was released in 1997.",
        "refutes": "The film Titanic was released in 2005.",
    },
    {
        "claim": "Water freezes at 0 degrees Celsius under standard atmospheric pressure.",
        "wiki_titles": "Freezing point",
        "supports": "Water freezes at 0 degrees Celsius under standard atmospheric pressure.",
        "refutes": "Water freezes at 50 degrees Celsius under standard atmospheric pressure.",
    },
]


def predict(model, vectorizer, scaler, claim, title, evidence):
    df = pd.DataFrame([{
        "claim": claim,
        "wiki_titles": title,
        "evidence": evidence,
    }])

    text = pd.Series([
        f"{claim} Wikipedia article: {title}. {evidence}"
    ])

    X_text = vectorizer.transform(text)
    X_rel = scaler.transform(relationship_features(df))

    X = hstack(
        [X_text, csr_matrix(X_rel)],
        format="csr",
    )

    probabilities = model.predict_proba(X)[0]
    classes = list(model.classes_)

    return {
        label: float(probabilities[i])
        for i, label in enumerate(classes)
    }


def main():
    saved = joblib.load(MODEL_PATH)

    model = saved["model"]
    vectorizer = saved["vectorizer"]
    scaler = saved["scaler"]

    correct = 0
    total = 0
    changed = 0

    for case in TEST_CASES:
        supports = predict(
            model, vectorizer, scaler,
            case["claim"], case["wiki_titles"],
            case["supports"],
        )

        refutes = predict(
            model, vectorizer, scaler,
            case["claim"], case["wiki_titles"],
            case["refutes"],
        )

        supports_prediction = max(supports, key=supports.get)
        refutes_prediction = max(refutes, key=refutes.get)

        correct += int(supports_prediction == "SUPPORTS")
        correct += int(refutes_prediction == "REFUTES")
        total += 2

        changed += int(supports_prediction != refutes_prediction)

        print("\nCLAIM:", case["claim"])
        print("SUPPORTING EVIDENCE:", case["supports"])
        print(
            f"  Prediction: {supports_prediction}"
            f" | P(SUPPORTS): {supports['SUPPORTS']:.4f}"
        )

        print("REFUTING EVIDENCE:", case["refutes"])
        print(
            f"  Prediction: {refutes_prediction}"
            f" | P(SUPPORTS): {refutes['SUPPORTS']:.4f}"
        )

        print(
            "  Probability difference:",
            f"{supports['SUPPORTS'] - refutes['SUPPORTS']:+.4f}",
        )

    print("\n=== E06a SUMMARY ===")
    print(f"Correct predictions: {correct}/{total}")
    print(f"Changed class between evidence pairs: {changed}/{len(TEST_CASES)}")


if __name__ == "__main__":
    main()
