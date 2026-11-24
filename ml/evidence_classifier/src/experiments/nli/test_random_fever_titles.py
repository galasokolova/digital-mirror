from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data" / "experiments" / "fever"

INPUT_FILE = DATA_DIR / "e04_model_comparison.csv"
OUTPUT_FILE = DATA_DIR / "e04_random_titles.csv"

MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
BATCH_SIZE = 8
RANDOM_STATE = 42

LABELS = ["REFUTES", "SUPPORTS"]

LABEL_MAPPING = {
    "contradiction": "REFUTES",
    "entailment": "SUPPORTS",
    "neutral": "NOT ENOUGH INFO",
}


def make_random_titles(df):
    titles = df["wiki_titles"].astype(str).to_numpy()
    rng = np.random.default_rng(RANDOM_STATE)

    # Shuffle titles until every row has a different title.
    # This avoids accidentally retaining the correct title.
    for _ in range(10000):
        shuffled = rng.permutation(titles)

        if np.all(shuffled != titles):
            return shuffled

    raise RuntimeError(
        "Could not generate different titles for every row"
    )


def evaluate(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)

    f1 = f1_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    return accuracy, f1


def main():
    df = pd.read_csv(INPUT_FILE)

    if df["wiki_titles"].isna().any():
        raise ValueError("Missing Wikipedia titles")

    random_titles = make_random_titles(df)

    df["random_title"] = random_titles

    df["random_context_evidence"] = [
        f"Wikipedia article: {title}. {evidence}"
        for title, evidence in zip(
            df["random_title"],
            df["evidence"],
        )
    ]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=== E04b: RANDOM TITLES TEST ===")
    print("Claims:", len(df))
    print("Device:", device, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
    ).to(device)

    model.eval()

    predictions = []

    for start in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[start:start + BATCH_SIZE]

        inputs = tokenizer(
            batch["random_context_evidence"].tolist(),
            batch["claim"].tolist(),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            logits = model(**inputs).logits

        predicted_ids = logits.argmax(dim=-1).tolist()

        for predicted_id in predicted_ids:
            label = model.config.id2label[
                predicted_id
            ].lower()

            predictions.append(LABEL_MAPPING[label])

        print(
            f"Processed: {min(start + BATCH_SIZE, len(df))}"
            f"/{len(df)}",
            flush=True,
        )

    df["random_prediction"] = predictions

    print("\n=== RESULTS ===")

    experiments = {
        "Original Evidence": df["prediction"],
        "Correct Wikipedia Title": df["context_prediction"],
        "Random Wikipedia Title": df["random_prediction"],
    }

    for name, predictions in experiments.items():
        accuracy, f1 = evaluate(df["label"], predictions)

        errors = int(
            (df["label"] != predictions).sum()
        )

        print(
            f"{name:25s} "
            f"Accuracy={accuracy:.4f} "
            f"Macro F1={f1:.4f} "
            f"Errors={errors}"
        )

    print("\n=== CONTEXT VS RANDOM ===")

    correct_context = (
        df["context_prediction"] == df["label"]
    )

    correct_random = (
        df["random_prediction"] == df["label"]
    )

    print(
        "Correct context wins:",
        int((correct_context & ~correct_random).sum()),
    )

    print(
        "Random context wins:",
        int((~correct_context & correct_random).sum()),
    )

    print(
        "Predictions changed:",
        int(
            (
                df["context_prediction"]
                != df["random_prediction"]
            ).sum()
        ),
    )

    print("\n=== EXAMPLES ===")

    for row in df.head(5).itertuples():
        print("\n" + "=" * 60)
        print("CLAIM:", row.claim)
        print("CORRECT TITLE:", row.wiki_titles)
        print("RANDOM TITLE:", row.random_title)
        print("EXPECTED:", row.label)
        print("CORRECT CONTEXT:", row.context_prediction)
        print("RANDOM CONTEXT:", row.random_prediction)

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nSaved:", OUTPUT_FILE)
    print("E04b completed.")


if __name__ == "__main__":
    main()
