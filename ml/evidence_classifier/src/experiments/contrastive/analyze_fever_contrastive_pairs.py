from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from ml.evidence_classifier.src.models.train_fever_relationship_model import load_split


BASE = Path(__file__).resolve().parents[3]
OUTPUT = BASE / "data" / "experiments" / "fever"

SIMILARITY_THRESHOLD = 0.80
MAX_PAIRS = 1000
NEIGHBORS = 10


def main():
    train = load_split("train")

    supports = train[
        train["label"] == "SUPPORTS"
    ].reset_index(drop=True)

    refutes = train[
        train["label"] == "REFUTES"
    ].reset_index(drop=True)

    print("SUPPORTS:", len(supports))
    print("REFUTES:", len(refutes))

    print("Building TF-IDF...", flush=True)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
    )

    X = vectorizer.fit_transform(train["claim"])

    X_supports = vectorizer.transform(supports["claim"])
    X_refutes = vectorizer.transform(refutes["claim"])

    print("Finding similar opposite-label claims...", flush=True)

    neighbors = NearestNeighbors(
        n_neighbors=NEIGHBORS,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )

    neighbors.fit(X_refutes)

    distances, indices = neighbors.kneighbors(X_supports)

    pairs = []

    for i in range(len(supports)):
        for distance, j in zip(distances[i], indices[i]):
            similarity = 1 - distance

            if similarity < SIMILARITY_THRESHOLD:
                continue

            support = supports.iloc[i]
            refute = refutes.iloc[j]

            if support["claim"] == refute["claim"]:
                continue

            pairs.append({
                "supports_claim_id": support["claim_id"],
                "supports_claim": support["claim"],
                "supports_evidence": support["evidence"],
                "refutes_claim_id": refute["claim_id"],
                "refutes_claim": refute["claim"],
                "refutes_evidence": refute["evidence"],
                "similarity": similarity,
            })

    result = pd.DataFrame(pairs)

    if result.empty:
        print("No candidate pairs found.")
        return

    result = result.sort_values(
        "similarity", ascending=False
    ).head(MAX_PAIRS)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    path = OUTPUT / "e07a_contrastive_candidates.csv"
    result.to_csv(path, index=False)

    print("\n=== E07a RESULTS ===")
    print("Candidate pairs:", len(result))
    print("Mean similarity:", round(result["similarity"].mean(), 4))

    print("\nTop 10 pairs:")

    for _, row in result.head(10).iterrows():
        print("\nSimilarity:", round(row["similarity"], 4))
        print("SUPPORTS:", row["supports_claim"])
        print("REFUTES :", row["refutes_claim"])

    print("\nSaved:", path)


if __name__ == "__main__":
    main()
