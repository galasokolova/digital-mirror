from pathlib import Path
from difflib import SequenceMatcher
import re
import unicodedata

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[3]

DATA = BASE / "data" / "processed" / "fever"
OUTPUT = BASE / "data" / "experiments" / "fever"

INPUT = OUTPUT / "e07a_contrastive_candidates.csv"


def normalize_claim(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = text.casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def tokenize(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def get_numbers(text):
    return set(re.findall(r"\b\d+\b", str(text)))


def get_titles(text):
    return {
        title.strip().casefold()
        for title in str(text).split("|")
        if title.strip()
    }


def analyze_pair(row):
    claim_a = row["supports_claim"]
    claim_b = row["refutes_claim"]

    tokens_a = tokenize(claim_a)
    tokens_b = tokenize(claim_b)

    numbers_a = get_numbers(claim_a)
    numbers_b = get_numbers(claim_b)

    matcher = SequenceMatcher(
        None,
        tokens_a,
        tokens_b,
        autojunk=False,
    )

    removed = []
    added = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            removed.extend(tokens_a[i1:i2])
            added.extend(tokens_b[j1:j2])

    titles_a = get_titles(row["supports_titles"])
    titles_b = get_titles(row["refutes_titles"])

    return pd.Series({
        "shared_wikipedia_title": bool(titles_a & titles_b),
        "number_changed": numbers_a != numbers_b,
        "removed_tokens": " ".join(removed),
        "added_tokens": " ".join(added),
        "token_change_count": len(removed) + len(added),
        "tfidf_similarity_near_one": bool(
            np.isclose(row["similarity"], 1.0, atol=1e-9)
        ),
    })


def classify_pair(row):
    if row["normalized_supports"] == row["normalized_refutes"]:
        return "IDENTICAL_TEXT_CONFLICT"

    if row["token_change_count"] <= 2:
        if row["shared_wikipedia_title"]:
            return "HIGH_PRIORITY_REVIEW"

        return "SHORT_TEXT_DIFFERENCE"

    return "COMPLEX_DIFFERENCE"


def main():
    print("Loading E07a candidates...", flush=True)

    pairs = pd.read_csv(INPUT)

    original_count = len(pairs)

    # First remove duplicate claim-ID pairs.
    pairs = pairs.drop_duplicates(
        subset=["supports_claim_id", "refutes_claim_id"]
    ).copy()

    print("Original candidates:", original_count)
    print("Unique ID pairs:", len(pairs))

    print("Loading original FEVER context...", flush=True)

    train = pd.read_csv(DATA / "train_context.csv")

    train = (
        train.sort_values(["claim_id", "evidence_set_id"])
        .drop_duplicates("claim_id")
    )

    metadata = train[
        ["claim_id", "wiki_titles"]
    ].copy()

    supports_meta = metadata.rename(columns={
        "claim_id": "supports_claim_id",
        "wiki_titles": "supports_titles",
    })

    refutes_meta = metadata.rename(columns={
        "claim_id": "refutes_claim_id",
        "wiki_titles": "refutes_titles",
    })

    pairs = pairs.merge(
        supports_meta,
        on="supports_claim_id",
        how="left",
        validate="many_to_one",
    )

    pairs = pairs.merge(
        refutes_meta,
        on="refutes_claim_id",
        how="left",
        validate="many_to_one",
    )

    if pairs[["supports_titles", "refutes_titles"]].isna().any().any():
        raise ValueError("Missing Wikipedia title metadata")

    print("Analyzing claim differences...", flush=True)

    analysis = pairs.apply(analyze_pair, axis=1)

    result = pd.concat(
        [
            pairs.reset_index(drop=True),
            analysis.reset_index(drop=True),
        ],
        axis=1,
    )

    # Normalize claims for duplicate and conflict detection.
    result["normalized_supports"] = (
        result["supports_claim"].apply(normalize_claim)
    )

    result["normalized_refutes"] = (
        result["refutes_claim"].apply(normalize_claim)
    )

    # Save all analyzed pairs before text deduplication.
    result.to_csv(
        OUTPUT / "e07b_validated_candidates.csv",
        index=False,
    )

    # Detect text-level duplicates.
    before_text_dedup = len(result)

    result = result.drop_duplicates(
        subset=["normalized_supports", "normalized_refutes"]
    ).copy()

    removed_duplicates = before_text_dedup - len(result)

    # Assign audit categories.
    result["audit_category"] = result.apply(
        classify_pair,
        axis=1,
    )

    # Sort review candidates.
    result = result.sort_values(
        [
            "shared_wikipedia_title",
            "token_change_count",
            "similarity",
        ],
        ascending=[False, True, False],
    )

    conflicts = result[
        result["audit_category"] == "IDENTICAL_TEXT_CONFLICT"
    ].copy()

    review = result[
        result["audit_category"] == "HIGH_PRIORITY_REVIEW"
    ].head(100).copy()

    review["verified_contradiction"] = ""
    review["evidence_valid_for_both_claims"] = ""
    review["review_notes"] = ""

    OUTPUT.mkdir(parents=True, exist_ok=True)

    audit_path = OUTPUT / "e07c_audited_pairs.csv"
    conflicts_path = OUTPUT / "e07c_identical_conflicts.csv"
    review_path = OUTPUT / "e07c_manual_review.csv"

    result.to_csv(audit_path, index=False)
    conflicts.to_csv(conflicts_path, index=False)
    review.to_csv(review_path, index=False)

    print("\n=== E07b + E07c RESULTS ===")

    print("Original candidate pairs:", original_count)
    print("Unique normalized text pairs:", len(result))
    print("Repeated text pairs removed:", removed_duplicates)

    print(
        "Shared Wikipedia title:",
        int(result["shared_wikipedia_title"].sum()),
    )

    print(
        "Number changes:",
        int(result["number_changed"].sum()),
    )

    print(
        "TF-IDF similarity near 1.0:",
        int(result["tfidf_similarity_near_one"].sum()),
    )

    print("\nAudit categories:")
    print(result["audit_category"].value_counts().to_string())

    print("\nIdentical-text conflicts:", len(conflicts))

    for _, row in conflicts.head(10).iterrows():
        print("\nSUPPORTS:", row["supports_claim"])
        print("REFUTES :", row["refutes_claim"])
        print("SUPPORTS ID:", row["supports_claim_id"])
        print("REFUTES ID :", row["refutes_claim_id"])

    print("\nSaved:")
    print(audit_path)
    print(conflicts_path)
    print(review_path)


if __name__ == "__main__":
    main()
