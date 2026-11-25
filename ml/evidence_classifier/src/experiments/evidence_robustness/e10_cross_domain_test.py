from pathlib import Path

import joblib
import pandas as pd

from ml.evidence_classifier.src.models.train_fever_relationship_model import make_features


MODEL_PATH = Path("ml/evidence_classifier/models/fever_relationship_logreg.joblib")
CANDIDATES_PATH = Path(
    "ml/evidence_classifier/data/experiments/e10_euvsdisinfo_candidates.csv"
)
OUTPUT_PATH = Path(
    "ml/evidence_classifier/data/experiments/e10_euvsdisinfo_results.csv"
)


# Short evidence statements based on external sources.
# These are curated summaries, not verbatim quotations.
EVIDENCE = {
    "E10-002": {
        "evidence": (
            "NATO Secretary General Jens Stoltenberg stated "
            "in September 2022 that NATO and its allies were "
            "not parties to the conflict and had no NATO troops "
            "fighting in Ukraine. NATO allies provided military "
            "support to Ukrainian forces."
        ),
        "wiki_titles": "NATO Ukraine Russia",
        "source_url": (
            "https://www.nato.int/en/news-and-events/events/"
            "transcripts/2022/09/27/opening-remarks"
        ),
        "evidence_quality": "PARTIAL",
    },
    "E10-003": {
        "evidence": (
            "Amnesty International reported that Ukrainian "
            "forces operated in populated residential areas "
            "and endangered civilians. The report did not "
            "characterize Ukrainian military tactics as "
            "terrorism or explicitly accuse Ukraine of "
            "deliberately using civilians as human shields."
        ),
        "wiki_titles": "Amnesty International Ukraine",
        "source_url": (
            "https://www.amnesty.org/en/latest/news/2022/08/"
            "ukraine-ukrainian-fighting-tactics-endanger-civilians/"
        ),
        "evidence_quality": "PARTIAL",
    },
    "E10-012": {
        "evidence": (
            "Amnesty International documented repeated "
            "indiscriminate bombardments of residential "
            "neighbourhoods in Kharkiv by Russian forces "
            "in 2022. These attacks killed and injured "
            "civilians and damaged civilian buildings."
        ),
        "wiki_titles": "Kharkiv Russian invasion of Ukraine",
        "source_url": (
            "https://www.amnesty.org/en/latest/research/2022/06/"
            "anyone-can-die-at-any-time-kharkiv/"
        ),
        "evidence_quality": "SUFFICIENT",
    },
}


def main():
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]

    candidates = pd.read_csv(CANDIDATES_PATH)

    rows = []

    for case_id, item in EVIDENCE.items():
        matching = candidates.loc[
            candidates["case_id"] == case_id
        ]

        if len(matching) != 1:
            raise ValueError(
                f"Expected one claim for {case_id}, "
                f"found {len(matching)}"
            )

        claim = str(matching.iloc[0]["claim"])
        evidence = item["evidence"]

        rows.append({
            "case_id": case_id,
            "claim": claim,
            "evidence": evidence,
            "wiki_titles": item["wiki_titles"],
            # Compatibility with the FEVER-trained pipeline.
            # This prefix is synthetic, not a real Wikipedia source.
            "context_evidence": (
                "Wikipedia article: "
                + item["wiki_titles"]
                + ". "
                + evidence
            ),
            "expected": "REFUTES",
            "evidence_quality": item["evidence_quality"],
            "source_url": item["source_url"],
        })

    df = pd.DataFrame(rows)

    X = make_features(
        df,
        saved["claim_vectorizer"],
        saved["evidence_vectorizer"],
        saved["similarity_vectorizer"],
        saved["scaler"],
        fit=False,
    )

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    classes = list(model.classes_)

    supports_idx = classes.index("SUPPORTS")
    refutes_idx = classes.index("REFUTES")

    df["prediction"] = predictions
    df["p_supports"] = probabilities[:, supports_idx]
    df["p_refutes"] = probabilities[:, refutes_idx]
    df["correct"] = df["prediction"] == df["expected"]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.drop(columns=["context_evidence"]).to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print("\n=== E10: CROSS-DOMAIN TEST ===")

    for _, row in df.iterrows():
        print("\n" + "=" * 65)
        print("ID:", row["case_id"])
        print("CLAIM:", row["claim"])
        print("EVIDENCE:", row["evidence"])
        print("QUALITY:", row["evidence_quality"])
        print("EXPECTED:", row["expected"])
        print("PREDICTION:", row["prediction"])
        print(f"P(SUPPORTS): {row['p_supports']:.4f}")
        print(f"P(REFUTES):  {row['p_refutes']:.4f}")
        print("CORRECT:", row["correct"])

    print("\n=== SUMMARY ===")
    print("Cases:", len(df))
    print("Correct:", int(df["correct"].sum()))
    print("Predicted REFUTES:", int(
        (df["prediction"] == "REFUTES").sum()
    ))
    print("Predicted SUPPORTS:", int(
        (df["prediction"] == "SUPPORTS").sum()
    ))
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
