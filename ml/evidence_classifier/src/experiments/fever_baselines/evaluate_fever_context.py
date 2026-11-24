from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

BASE_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = (
    BASE_DIR / "data" / "experiments" / "fever"
)

INPUT_FILE = DATA_DIR / "e03_context_sample.csv"
OUTPUT_FILE = DATA_DIR / "e03_context_results.csv"

MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
BATCH_SIZE = 8

LABEL_MAPPING = {
    "contradiction": "REFUTES",
    "entailment": "SUPPORTS",
    "neutral": "NOT ENOUGH INFO",
}

LABELS = ["REFUTES", "SUPPORTS"]


def evaluate_predictions(y_true, predictions):
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "macro_f1": f1_score(
            y_true,
            predictions,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
    }


def main():
    df = pd.read_csv(INPUT_FILE)

    required = [
        "claim",
        "label",
        "prediction",
        "evidence",
        "context_evidence",
    ]

    if df[required].isna().any().any():
        raise ValueError("Missing values in required columns")

    print("=== E03c: CONTEXT EXPERIMENT ===")
    print("Claims:", len(df), flush=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
    ).to(device)

    model.eval()

    predictions = []
    confidences = []

    for start in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[start:start + BATCH_SIZE]

        inputs = tokenizer(
            batch["context_evidence"].tolist(),
            batch["claim"].tolist(),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)

        ids = probabilities.argmax(dim=-1).tolist()

        for i, predicted_id in enumerate(ids):
            label = model.config.id2label[
                predicted_id
            ].lower()

            predictions.append(LABEL_MAPPING[label])
            confidences.append(
                probabilities[i, predicted_id].item()
            )

        print(
            f"Processed: {min(start + BATCH_SIZE, len(df))}"
            f"/{len(df)}",
            flush=True,
        )

    df["context_prediction"] = predictions
    df["context_confidence"] = confidences

    df["original_correct"] = (
        df["prediction"] == df["label"]
    )

    df["context_correct"] = (
        df["context_prediction"] == df["label"]
    )

    original = evaluate_predictions(
        df["label"], df["prediction"]
    )

    context = evaluate_predictions(
        df["label"], df["context_prediction"]
    )

    print("\n=== RESULTS ===")

    print(
        f"Original Accuracy: {original['accuracy']:.4f}"
    )
    print(
        f"Context Accuracy:  {context['accuracy']:.4f}"
    )

    print(
        f"Original Macro F1: {original['macro_f1']:.4f}"
    )
    print(
        f"Context Macro F1:  {context['macro_f1']:.4f}"
    )

    print(
        "Macro F1 difference:",
        f"{context['macro_f1'] - original['macro_f1']:+.4f}",
    )

    fixed = df[
        ~df["original_correct"] & df["context_correct"]
    ]

    broken = df[
        df["original_correct"] & ~df["context_correct"]
    ]

    print("\n=== PREDICTION CHANGES ===")
    print("Originally wrong, now correct:", len(fixed))
    print("Originally correct, now wrong:", len(broken))

    print(
        "Total changed predictions:",
        int(
            (df["prediction"] != df["context_prediction"])
            .sum()
        ),
    )

    print("\n=== ERRORS BY CLASS ===")

    for label in LABELS:
        subset = df[df["label"] == label]

        old_errors = (~subset["original_correct"]).sum()
        new_errors = (~subset["context_correct"]).sum()

        print(
            f"{label}: "
            f"original={old_errors}, context={new_errors}"
        )

    print("\n=== FIXED EXAMPLES ===")

    for row in fixed.head(8).itertuples():
        print("\n" + "=" * 65)
        print("CLAIM:", row.claim)
        print("WIKIPEDIA:", row.wiki_titles)
        print("EVIDENCE:", row.evidence[:400])
        print("EXPECTED:", row.label)
        print("BEFORE:", row.prediction)
        print("AFTER:", row.context_prediction)

    print("\n=== NEW ERRORS ===")

    for row in broken.head(5).itertuples():
        print("\n" + "=" * 65)
        print("CLAIM:", row.claim)
        print("WIKIPEDIA:", row.wiki_titles)
        print("EXPECTED:", row.label)
        print("BEFORE:", row.prediction)
        print("AFTER:", row.context_prediction)

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nSaved:", OUTPUT_FILE)
    print("E03c completed.")


if __name__ == "__main__":
    main()
