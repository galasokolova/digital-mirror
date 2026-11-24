
from pathlib import Path
import sys

import joblib
import pandas as pd


# Project paths
ML_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ML_ROOT / "models" / "fever_relationship_logreg.joblib"

# Allow importing the original feature extraction function
sys.path.insert(0, str(ML_ROOT / "src"))

from models.train_fever_relationship_model import make_features


class FeverPredictor:

    def __init__(self, model_path=MODEL_PATH):

        model_path = Path(model_path)

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Train E06b first or restore the saved model."
            )

        bundle = joblib.load(model_path)

        if bundle.get("feature_version") != "e06b":
            raise ValueError("Expected E06b model bundle.")

        self.model = bundle["model"]
        self.claim_vectorizer = bundle["claim_vectorizer"]
        self.evidence_vectorizer = bundle["evidence_vectorizer"]
        self.similarity_vectorizer = bundle["similarity_vectorizer"]
        self.scaler = bundle["scaler"]

        self.classes = list(self.model.classes_)

        if set(self.classes) != {"SUPPORTS", "REFUTES"}:
            raise ValueError(
                f"Unexpected model classes: {self.classes}"
            )

        print("E06b model loaded successfully.")

    def predict(self, claim, evidence, wiki_title=""):

        if not isinstance(claim, str) or not claim.strip():
            raise ValueError("Claim cannot be empty.")

        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError("Evidence cannot be empty.")

        if not isinstance(wiki_title, str):
            raise ValueError("wiki_title must be a string.")

        claim = claim.strip()
        evidence = evidence.strip()
        wiki_title = wiki_title.strip()

        # Reproduce the training data structure
        context_evidence = (
            "Wikipedia article: "
            + wiki_title
            + ". "
            + evidence
        )

        df = pd.DataFrame([{
            "claim": claim,
            "evidence": evidence,
            "wiki_titles": wiki_title,
            "context_evidence": context_evidence
        }])

        # Use exactly the same feature extraction pipeline
        X = make_features(
            df,
            self.claim_vectorizer,
            self.evidence_vectorizer,
            self.similarity_vectorizer,
            self.scaler
        )

        probabilities = self.model.predict_proba(X)[0]

        support_index = self.classes.index("SUPPORTS")
        refute_index = self.classes.index("REFUTES")

        p_supports = float(probabilities[support_index])
        p_refutes = float(probabilities[refute_index])

        prediction = self.model.predict(X)[0]

        return {
            "claim": claim,
            "evidence": evidence,
            "wiki_title": wiki_title,
            "prediction": str(prediction),
            "p_supports": round(p_supports, 4),
            "p_refutes": round(p_refutes, 4),
            "confidence": round(
                max(p_supports, p_refutes), 4
            ),
            "model_version": "e06b"
        }


if __name__ == "__main__":

    predictor = FeverPredictor()

    result = predictor.predict(
        claim="The committee approved the proposal.",
        evidence="The committee rejected the proposal.",
        wiki_title="Committee"
    )

    print("\n=== E06b PREDICTION ===")

    for key, value in result.items():
        print(f"{key}: {value}")
