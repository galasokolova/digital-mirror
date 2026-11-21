import pandas as pd

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data/processed/liar"

train = pd.read_csv(DATA_DIR / "train.csv")

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

model.fit(
    train["text"].fillna(""),
    train["target"],
)

pairs = [
    (
        "Ukraine became a member of NATO in 2015.",
        "Ukraine did not become a member of NATO in 2015.",
    ),
    (
        "The European Union imposed sanctions on Russia.",
        "The European Union never imposed sanctions on Russia.",
    ),
    (
        "Russia invaded Ukraine in February 2022.",
        "Russia did not invade Ukraine in February 2022.",
    ),
    (
        "Crimea was annexed by Russia in 2014.",
        "Crimea was not annexed by Russia in 2014.",
    ),
]

true_index = list(
    model.named_steps["classifier"].classes_
).index(1)

print("=== CONTRADICTION SENSITIVITY TEST ===")

for original, opposite in pairs:
    p1 = model.predict_proba([original])[0][true_index]
    p2 = model.predict_proba([opposite])[0][true_index]

    print("\nOriginal:", original)
    print(f"P(TRUE): {p1:.4f}")

    print("Opposite:", opposite)
    print(f"P(TRUE): {p2:.4f}")

    print(f"Difference: {abs(p1 - p2):.4f}")
