from pathlib import Path

import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

INPUT_PATH = Path(
    "ml/data/experiments/e13_negation_generalization.csv"
)
OUTPUT_PATH = Path(
    "ml/data/experiments/e14_semantic_contradiction.csv"
)

LABEL_MAP = {
    "entailment": "SUPPORTS",
    "contradiction": "REFUTES",
    "neutral": "NEUTRAL",
}


def main():
    df = pd.read_csv(INPUT_PATH)

    print("Loading DeBERTa from local cache...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )

    model.eval()

    label_names = {
        i: name.lower()
        for i, name in model.config.id2label.items()
    }

    print("Model labels:", label_names)

    if set(label_names.values()) != set(LABEL_MAP):
        raise ValueError(
            f"Unexpected model labels: {label_names}"
        )

    results = []

    for row in df.itertuples(index=False):
        inputs = tokenizer(
            row.evidence,  # premise
            row.claim,     # hypothesis
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.inference_mode():
            logits = model(**inputs).logits
            probabilities = torch.softmax(
                logits, dim=-1
            )[0]

        scores = {
            label_names[i]: probabilities[i].item()
            for i in range(len(probabilities))
        }

        predicted_nli = max(scores, key=scores.get)
        predicted = LABEL_MAP[predicted_nli]

        results.append({
            "variant": row.variant,
            "expected": row.expected,
            "e06b_prediction": row.prediction,
            "deberta_prediction": predicted,
            "p_entailment": scores["entailment"],
            "p_contradiction": scores["contradiction"],
            "p_neutral": scores["neutral"],
        })

    result = pd.DataFrame(results)

    result["e06b_correct"] = (
        result["e06b_prediction"] == result["expected"]
    )
    result["deberta_correct"] = (
        result["deberta_prediction"] == result["expected"]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    result.to_csv(OUTPUT_PATH, index=False)

    print("\n=== E14: SEMANTIC CONTRADICTION TEST ===")

    print(
        result[
            [
                "variant",
                "expected",
                "e06b_prediction",
                "deberta_prediction",
                "p_entailment",
                "p_contradiction",
                "p_neutral",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n=== SUMMARY ===")
    print(
        f"E06b: {result['e06b_correct'].sum()}/{len(result)}"
    )
    print(
        f"DeBERTa: {result['deberta_correct'].sum()}/{len(result)}"
    )

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
