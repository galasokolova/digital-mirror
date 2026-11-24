import pandas as pd

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data/processed/liar"

train = pd.read_csv(DATA_DIR / "train.csv")
valid = pd.read_csv(DATA_DIR / "valid.csv")

X_train = train["text"].fillna("")
y_train = train["target"]

X_valid = valid["text"].fillna("")
y_valid = valid["target"]

# Baseline 0: always predict the majority class
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)

dummy_pred = dummy.predict(X_valid)

print("=== DUMMY BASELINE ===")
print("Accuracy:", round(
    accuracy_score(y_valid, dummy_pred), 4
))
print("Macro F1:", round(
    f1_score(y_valid, dummy_pred, average="macro"), 4
))

# Baseline 1: text-based classifier
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        ),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
    ),
])

model.fit(X_train, y_train)

predictions = model.predict(X_valid)

print("\n=== TF-IDF + LOGISTIC REGRESSION ===")

print("Accuracy:", round(
    accuracy_score(y_valid, predictions), 4
))

print("Macro F1:", round(
    f1_score(y_valid, predictions, average="macro"), 4
))

print("\nClassification report:")
print(classification_report(
    y_valid,
    predictions,
    labels=[0, 1],
    target_names=["FALSE", "TRUE"],
    zero_division=0,
))

print("Confusion matrix:")
print(confusion_matrix(
    y_valid,
    predictions,
    labels=[0, 1],
))

print("\n=== EXAMPLE PREDICTIONS ===")

examples = [
    "Russia launched a full-scale invasion of Ukraine in February 2022.",
    "The United Nations General Assembly adopted a resolution on Ukraine in March 2022.",
    "The European Union has never imposed sanctions on Russia.",
    "Ukraine became a member of NATO in 2015.",
]

probabilities = model.predict_proba(examples)
classes = list(model.named_steps["classifier"].classes_)
true_index = classes.index(1)

for text, probs in zip(examples, probabilities):
    p_true = probs[true_index]

    print(f"\nClaim: {text}")
    print(f"P(TRUE): {p_true:.4f}")
    print(
        "Prediction:",
        "TRUE" if p_true >= 0.5 else "FALSE"
    )
