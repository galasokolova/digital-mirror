from pathlib import Path
from collections import Counter

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data" / "processed" / "fever"
OUTPUT_DIR = BASE_DIR / "data" / "experiments" / "fever"

MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

SAMPLE_PER_CLASS = 150
BATCH_SIZE = 8
RANDOM_STATE = 42

LABEL_MAPPING = {
    "entailment": "SUPPORTS",
    "contradiction": "REFUTES",
    "neutral": "NOT ENOUGH INFO",
}


def load_sample():
    df = pd.read_csv(DATA_DIR / "dev_verified.csv")

    df = (
        df.sort_values(["claim_id", "evidence_set_id"])
        .drop_duplicates(subset=["claim_id"])
    )

    samples = []

    for label in ["SUPPORTS", "REFUTES"]:
        subset = df[df["label"] == label]

        samples.append(
            subset.sample(
                n=min(SAMPLE_PER_CLASS, len(subset)),
                random_state=RANDOM_STATE,
            )
        )

    return pd.concat(samples).sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_sample()
    print(f"Evaluation claims: {len(df)}", flush=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
    )

    model.to(device)
    model.eval()

    predictions = []
    confidences = []

    for start in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[start:start + BATCH_SIZE]

        inputs = tokenizer(
            batch["evidence"].tolist(),
            batch["claim"].tolist(),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)

        predicted_ids = probabilities.argmax(dim=-1)

        for i, predicted_id in enumerate(predicted_ids.tolist()):
            nli_label = model.config.id2label[
                predicted_id
            ].lower()

            predictions.append(LABEL_MAPPING[nli_label])
            confidences.append(
                probabilities[i, predicted_id].item()
            )

        print(
            f"Processed: {min(start + BATCH_SIZE, len(df))}"
            f"/{len(df)}",
            flush=True,
        )

    df["prediction"] = predictions
    df["confidence"] = confidences
    df["correct"] = df["label"] == df["prediction"]

    labels = ["REFUTES", "SUPPORTS", "NOT ENOUGH INFO"]

    print("\n=== E03 RESULTS ===")
    print(f"Accuracy: {accuracy_score(df['label'], predictions):.4f}")

    print(
        "Macro F1 (SUPPORTS/REFUTES):",
        f"{f1_score(df['label'], predictions, labels=labels[:2], average='macro', zero_division=0):.4f}",
    )

    print("\nPrediction distribution:")
    print(Counter(predictions))

    print("\nClassification report:")
    print(
        classification_report(
            df["label"],
            predictions,
            labels=labels,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print("Labels:", labels)
    print(
        confusion_matrix(
            df["label"],
            predictions,
            labels=labels,
        )
    )

    print("\nErrors:", (~df["correct"]).sum())

    output_path = OUTPUT_DIR / "e03_nli_sample.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path}")
    print("E03 evaluation completed.")


if __name__ == "__main__":
    main()
