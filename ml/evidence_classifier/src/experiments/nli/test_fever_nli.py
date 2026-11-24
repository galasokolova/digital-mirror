
from pathlib import Path

import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_FILE = (
    BASE_DIR / "data" / "processed"
    / "fever" / "dev_verified.csv"
)

MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

LABEL_MAPPING = {
    "entailment": "SUPPORTS",
    "contradiction": "REFUTES",
    "neutral": "NOT ENOUGH INFO",
}


def predict(model, tokenizer, evidence, claim):
    # Order matters: evidence = premise, claim = hypothesis.
    inputs = tokenizer(
        evidence,
        claim,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.inference_mode():
        logits = model(**inputs).logits[0]
        probabilities = torch.softmax(logits, dim=-1)

    predicted_id = int(probabilities.argmax())
    nli_label = model.config.id2label[predicted_id].lower()

    return {
        "nli_label": nli_label,
        "fever_label": LABEL_MAPPING[nli_label],
        "probabilities": {
            model.config.id2label[i].lower(): float(prob)
            for i, prob in enumerate(probabilities)
        },
    }


def show_result(model, tokenizer, evidence, claim, expected):
    result = predict(model, tokenizer, evidence, claim)

    print("\n" + "=" * 65)
    print("EVIDENCE:", evidence[:350])
    print("CLAIM:", claim[:350])
    print("EXPECTED:", expected)
    print("PREDICTED:", result["fever_label"])
    print("NLI LABEL:", result["nli_label"])

    print("\nProbabilities:")
    for label, probability in result["probabilities"].items():
        print(f"  {label}: {probability:.4f}")


def main():
    print("Loading NLI model...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    print("Model loaded.")
    print("Labels:", model.config.id2label)

    examples = [
        (
            "The Eiffel Tower is located in Paris, France.",
            "The Eiffel Tower is in France.",
            "SUPPORTS",
        ),
        (
            "Marie Curie was born in Warsaw, Poland.",
            "Marie Curie was born in France.",
            "REFUTES",
        ),
        (
            "The Eiffel Tower is located in Paris, France.",
            "The Eiffel Tower was painted in 2025.",
            "NOT ENOUGH INFO",
        ),
    ]

    print("\n=== CONTROLLED EXAMPLES ===")

    for evidence, claim, expected in examples:
        show_result(
            model, tokenizer, evidence, claim, expected
        )

    print("\n=== FEVER DEV EXAMPLES ===")

    df = pd.read_csv(DATA_FILE)

    # One evidence set per claim.
    df = (
        df.sort_values(["claim_id", "evidence_set_id"])
        .drop_duplicates(subset=["claim_id"])
    )

    # Fixed sample: 2 SUPPORTS and 1 REFUTES.
    supports = df[df["label"] == "SUPPORTS"].head(2)
    refutes = df[df["label"] == "REFUTES"].head(1)

    sample = pd.concat([supports, refutes])

    for row in sample.itertuples():
        show_result(
            model,
            tokenizer,
            row.evidence,
            row.claim,
            row.label,
        )

    print("\nE03 initial test completed.")


if __name__ == "__main__":
    main()
