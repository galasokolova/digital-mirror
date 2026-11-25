
from pathlib import Path
import sys

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)


ML_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ML_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from preprocessing.intent_text_normalizer import (
    remove_punctuation
)


MODELS_DIR = ML_ROOT / "models"
EXPERIMENTS_DIR = ML_ROOT / "experiments"

TRAIN_PATH = (
    ML_ROOT / "data" / "processed"
    / "intent_v1" / "train.csv"
)


MODEL_FILES = {
    "E09_01": "E09_01_word_logreg.joblib",
    "E09_02": "E09_02_word_logreg_balanced.joblib",
}


TEST_CASES = {

    "GREETING": [
        "Hello!",
        "Hi there!",
        "Good morning.",
        "Good evening.",
        "Hey, how are you?",
    ],

    "GOODBYE": [
        "Goodbye!",
        "See you later.",
        "Bye for now.",
        "Have a nice day.",
        "I have to go now.",
    ],

    "GRATITUDE": [
        "Thank you!",
        "Thanks a lot.",
        "Thank you for explaining.",
        "I appreciate your help.",
        "Many thanks.",
    ],

    "AGREEMENT": [
        "I agree with you.",
        "You are absolutely right.",
        "I think so too.",
        "I completely agree.",
        "That is exactly what I think.",
    ],

    "DISAGREEMENT": [
        "I disagree with you.",
        "I don't agree.",
        "I think you are wrong.",
        "I have a different opinion.",
        "I don't think that's right.",
    ],

    "UNCERTAINTY": [
        "I'm not sure.",
        "I don't know.",
        "I can't decide.",
        "I'm not certain.",
        "I have some doubts.",
    ],

    "ACKNOWLEDGMENT": [
        "I see.",
        "I understand.",
        "Got it.",
        "I understand what you mean.",
        "Okay, I see.",
    ],

    "AFFIRMATIVE_ANSWER": [
        "Yes, please.",
        "Yes, I would.",
        "Yes, I do.",
        "Yes, I can.",
        "Yes, I am.",
    ],

    "NEGATIVE_ANSWER": [
        "No, thank you.",
        "No, I don't.",
        "No, I can't.",
        "No, I wouldn't.",
        "No, I am not.",
    ],

    "QUESTION": [
        "Where is the evidence?",
        "Is this information true?",
        "How do you know that?",
        "Who published this news?",
        "Can you show me the source?",
    ],

    "CLARIFICATION": [
        "What do you mean?",
        "Could you repeat that?",
        "I didn't understand that.",
        "Could you explain that again?",
        "Sorry, what did you say?",
    ],

    "STATEMENT": [
        "I thought it was true.",
        "This information is new to me.",
        "I read this news yesterday.",
        "The article was published last week.",
        "I have seen this video before.",
    ],
}


def load_test_cases():

    records = []

    for intent, phrases in TEST_CASES.items():

        for phrase in phrases:

            records.append({
                "text": phrase,
                "model_text": remove_punctuation(phrase),
                "expected_intent": intent,
            })

    return pd.DataFrame(records)


def evaluate_model(
    model_name,
    model,
    test_data
):

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    X = test_data["model_text"]
    y = test_data["expected_intent"]

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    classes = model.classes_

    results = []

    for i, row in test_data.iterrows():

        predicted = predictions[i]
        expected = row["expected_intent"]

        correct = predicted == expected

        top_indices = probabilities[i].argsort()[::-1][:3]

        print(f"\nText: {row['text']}")
        print(f"Expected:  {expected}")
        print(f"Predicted: {predicted}")

        print(
            "Result:",
            "CORRECT" if correct else "INCORRECT"
        )

        print(
            "Seen in training:",
            row["seen_in_training"]
        )

        print("Top 3:")

        for index in top_indices:

            print(
                f"  {classes[index]:22s}"
                f" {probabilities[i][index]:.4f}"
            )

        expected_index = list(classes).index(expected)

        results.append({
            "model": model_name,
            "text": row["text"],
            "expected": expected,
            "predicted": predicted,
            "correct": correct,
            "seen_in_training": row["seen_in_training"],
            "top_1_score": probabilities[i].max(),
            "expected_class_score":
                probabilities[i][expected_index],
            "expected_in_top_3":
                expected in classes[top_indices],
        })

    results_df = pd.DataFrame(results)

    accuracy = accuracy_score(y, predictions)

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
        zero_division=0
    )

    top_3_accuracy = (
        results_df["expected_in_top_3"].mean()
    )

    print("\n=== INSTALLATION TEST RESULTS ===")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Top-3 accuracy: {top_3_accuracy:.4f}")

    print(
        f"Correct predictions: "
        f"{results_df['correct'].sum()}"
        f"/{len(results_df)}"
    )

    print("\n=== CLASSIFICATION REPORT ===")

    print(
        classification_report(
            y,
            predictions,
            labels=sorted(TEST_CASES),
            digits=4,
            zero_division=0
        )
    )

    print("\n=== SEEN / UNSEEN ===")

    for seen in [True, False]:

        subset = results_df[
            results_df["seen_in_training"] == seen
        ]

        if subset.empty:
            continue

        print(
            f"Seen={seen}: "
            f"{subset['correct'].sum()}/{len(subset)} "
            f"({subset['correct'].mean():.4f})"
        )

    summary = {
        "model": model_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "top_3_accuracy": top_3_accuracy,
        "correct": int(results_df["correct"].sum()),
        "total": len(results_df),
    }

    return results_df, summary


def main():

    print("=== E09.03: INSTALLATION INTENT TEST ===")

    EXPERIMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    test_data = load_test_cases()

    train = pd.read_csv(
        TRAIN_PATH,
        usecols=["model_text"],
        keep_default_na=False
    )

    test_data["seen_in_training"] = (
        test_data["model_text"].isin(
            set(train["model_text"])
        )
    )

    print(f"Test cases: {len(test_data)}")
    print(f"Intent classes: {len(TEST_CASES)}")

    print(
        "Seen texts:",
        test_data["seen_in_training"].sum()
    )

    print(
        "Unseen texts:",
        (~test_data["seen_in_training"]).sum()
    )

    all_results = []
    summaries = []

    for model_name, filename in MODEL_FILES.items():

        model = joblib.load(
            MODELS_DIR / filename
        )

        results, summary = evaluate_model(
            model_name,
            model,
            test_data
        )

        all_results.append(results)
        summaries.append(summary)

    detailed_results = pd.concat(
        all_results,
        ignore_index=True
    )

    comparison = pd.DataFrame(summaries)

    detailed_results.to_csv(
        EXPERIMENTS_DIR
        / "E09_03_installation_predictions.csv",
        index=False
    )

    comparison.to_csv(
        EXPERIMENTS_DIR
        / "E09_03_installation_comparison.csv",
        index=False
    )

    print("\n=== FINAL MODEL COMPARISON ===")

    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    print("\n=== TEST COMPLETE ===")


if __name__ == "__main__":
    main()
