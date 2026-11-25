
from pathlib import Path
import hashlib
import sys

import joblib
import numpy as np
import pandas as pd


ML_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ML_ROOT / "src"))

from taxonomy.intent_mapping import INTENT_CLASSES

from preprocessing.intent_text_normalizer import (
    remove_punctuation,
)


EXPERIMENT = "E09_07_01"

MODELS_DIR = ML_ROOT / "models"

EXPERIMENTS_DIR = ML_ROOT / "experiments"

PROCESSED_DIR = ML_ROOT / "data" / "processed"

MODEL_PATH = (
    MODELS_DIR
    / "E09_05a_domain_adapted.joblib"
)

DATASET_PATH = (
    ML_ROOT
    / "data"
    / "evaluation"
    / "digital_mirror_final"
    / "test_processed.csv"
)

PREDICTIONS_PATH = (
    EXPERIMENTS_DIR
    / "E09_06_02e_predictions.csv"
)

SWITCHBOARD_TRAIN_PATH = (
    PROCESSED_DIR
    / "intent_v1"
    / "train.csv"
)

DOMAIN_TRAIN_PATH = (
    PROCESSED_DIR
    / "digital_mirror_v1"
    / "train_eligible.csv"
)

EXPECTED_MODEL_HASH = (
    "650b1a638d02e0c435cbb4d221dab74aa"
    "756e37e7f145b605b4772cb37276373"
)

EXPECTED_DATASET_HASH = (
    "95edcdb2de241103150890f99800b2ecf"
    "0ad3b6f43961c417fc87169fd86a390"
)

TOP_FEATURES = 20

PROBLEM_CLASSES = [
    "AGREEMENT",
    "ACKNOWLEDGMENT",
    "DISAGREEMENT",
    "STATEMENT",
    "QUESTION",
    "UNCERTAINTY",
]

DIAGNOSTIC_TEXTS = [
    "Howdy",
    "Cheerio",
    "I agree",
    "I disagree",
    "I completely agree",
    "I completely disagree",
    "I share your assessment.",
    "I share your conclusion about the evidence.",
    "I endorse your conclusion because the evidence supports it.",
    "Understood so far",
    "Duly noted",
    "Information received",
    "Heard you",
    "I've followed your explanation so far.",
    "I understand",
    "I understand your explanation.",
    "I don't understand your explanation.",
    "I am uncertain.",
    "I am not uncertain.",
]


def calculate_sha256(path):

    sha256 = hashlib.sha256()

    with path.open("rb") as file:

        for chunk in iter(
            lambda: file.read(65536),
            b""
        ):

            sha256.update(chunk)

    return sha256.hexdigest()


def load_dataset(path):

    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False
    )


def verify_frozen_files():

    print("\n=== FROZEN FILE VERIFICATION ===")

    model_hash = calculate_sha256(MODEL_PATH)

    dataset_hash = calculate_sha256(DATASET_PATH)

    checks = {

        "Frozen model hash":
            model_hash == EXPECTED_MODEL_HASH,

        "Frozen dataset hash":
            dataset_hash == EXPECTED_DATASET_HASH,
    }

    for name, passed in checks.items():

        status = "PASS" if passed else "FAIL"

        print(f"{status} | {name}")

    if not all(checks.values()):

        raise ValueError(
            "Frozen model or dataset has changed."
        )


def validate_predictions(dataset, predictions):

    print("\n=== PREDICTION VALIDATION ===")

    checks = {

        "Expected dataset size":
            len(dataset) == 240,

        "Expected prediction count":
            len(predictions) == 240,

        "Unique dataset IDs":
            dataset["id"].is_unique,

        "Unique prediction IDs":
            predictions["id"].is_unique,

        "Matching record IDs":
            dataset["id"].equals(
                predictions["id"]
            ),

        "Matching model texts":
            dataset["model_text"].equals(
                predictions["model_text"]
            ),

        "Matching expected labels":
            dataset["intent"].equals(
                predictions["expected"]
            ),
    }

    for name, passed in checks.items():

        status = "PASS" if passed else "FAIL"

        print(f"{status} | {name}")

    if not all(checks.values()):

        raise ValueError(
            "Prediction validation failed."
        )


def get_feature_information(text, vectorizer):

    analyzer = vectorizer.build_analyzer()

    vocabulary = vectorizer.vocabulary_

    features = analyzer(text)

    known_features = [
        feature
        for feature in features
        if feature in vocabulary
    ]

    unknown_features = [
        feature
        for feature in features
        if feature not in vocabulary
    ]

    total = len(features)

    coverage = (
        len(known_features) / total
        if total > 0
        else 0.0
    )

    return {

        "total_features": total,

        "known_features":
            len(known_features),

        "unknown_features":
            len(unknown_features),

        "feature_coverage": coverage,

        "known_feature_list":
            " | ".join(known_features),

        "unknown_feature_list":
            " | ".join(unknown_features),
    }


def get_word_information(text, vectorizer):

    tokenizer = vectorizer.build_tokenizer()

    preprocessor = vectorizer.build_preprocessor()

    words = tokenizer(
        preprocessor(text)
    )

    known_words = [
        word
        for word in words
        if word in vectorizer.vocabulary_
    ]

    unknown_words = [
        word
        for word in words
        if word not in vectorizer.vocabulary_
    ]

    return {

        "total_words": len(words),

        "known_words": len(known_words),

        "unknown_words": len(unknown_words),

        "word_coverage": (
            len(known_words) / len(words)
            if words
            else 0.0
        ),

        "unknown_word_list":
            " | ".join(unknown_words),
    }


def analyze_feature_coverage(
    dataset,
    predictions,
    vectorizer
):

    print("\n=== D01: FEATURE COVERAGE ===")

    vectors = vectorizer.transform(
        dataset["model_text"]
    )

    nonzero_counts = np.asarray(
        vectors.getnnz(axis=1)
    ).ravel()

    rows = []

    for i, row in enumerate(
        dataset.itertuples(index=False)
    ):

        feature_info = get_feature_information(
            row.model_text,
            vectorizer
        )

        word_info = get_word_information(
            row.model_text,
            vectorizer
        )

        predicted = predictions.iloc[i]["predicted"]

        rows.append({

            "id": row.id,

            "text": row.text,

            "model_text": row.model_text,

            "intent": row.intent,

            "group": row.group,

            "predicted": predicted,

            "correct":
                row.intent == predicted,

            "nonzero_features":
                int(nonzero_counts[i]),

            "zero_vector":
                nonzero_counts[i] == 0,

            **feature_info,

            **word_info,
        })

    coverage = pd.DataFrame(rows)

    print(f"Records: {len(coverage)}")

    print(
        "Mean feature coverage: "
        f"{coverage['feature_coverage'].mean():.2%}"
    )

    print(
        "Mean word coverage: "
        f"{coverage['word_coverage'].mean():.2%}"
    )

    print(
        "\nFeature coverage statistics:"
    )

    print(
        coverage[
            [
                "total_features",
                "known_features",
                "unknown_features",
                "nonzero_features",
                "feature_coverage",
                "word_coverage",
            ]
        ].describe().round(4).to_string()
    )

    print("\n=== D02: ZERO FEATURE VECTORS ===")

    zero_vectors = coverage.loc[
        coverage["zero_vector"]
    ].copy()

    print(
        f"Zero vectors: {len(zero_vectors)}"
    )

    print(
        f"Zero vector percentage: "
        f"{len(zero_vectors) / len(coverage):.2%}"
    )

    if not zero_vectors.empty:

        print(
            zero_vectors[
                [
                    "id",
                    "text",
                    "intent",
                    "predicted",
                    "unknown_word_list",
                ]
            ].to_string(index=False)
        )

    return coverage, zero_vectors


def analyze_coverage_vs_accuracy(coverage):

    print("\n=== D03: COVERAGE VS ACCURACY ===")

    summary_rows = []

    groups = {

        "all":
            coverage,

        "correct":
            coverage.loc[coverage["correct"]],

        "incorrect":
            coverage.loc[~coverage["correct"]],

        "zero_vector":
            coverage.loc[coverage["zero_vector"]],

        "nonzero_vector":
            coverage.loc[~coverage["zero_vector"]],
    }

    for name, subset in groups.items():

        if subset.empty:

            continue

        summary_rows.append({

            "subset": name,

            "records": len(subset),

            "accuracy":
                subset["correct"].mean(),

            "mean_feature_coverage":
                subset["feature_coverage"].mean(),

            "mean_word_coverage":
                subset["word_coverage"].mean(),

            "mean_nonzero_features":
                subset["nonzero_features"].mean(),
        })

    summary = pd.DataFrame(summary_rows)

    print(
        summary.round(4)
        .to_string(index=False)
    )

    print("\nCoverage by intent:")

    by_intent = (

        coverage.groupby("intent")

        .agg(

            records=("id", "count"),

            accuracy=("correct", "mean"),

            mean_feature_coverage=(
                "feature_coverage",
                "mean"
            ),

            mean_word_coverage=(
                "word_coverage",
                "mean"
            ),

            zero_vectors=(
                "zero_vector",
                "sum"
            ),
        )

        .reindex(INTENT_CLASSES)
    )

    print(
        by_intent.round(4).to_string()
    )

    print("\nCoverage by group:")

    by_group = (

        coverage.groupby("group")

        .agg(

            records=("id", "count"),

            accuracy=("correct", "mean"),

            mean_feature_coverage=(
                "feature_coverage",
                "mean"
            ),

            mean_word_coverage=(
                "word_coverage",
                "mean"
            ),

            zero_vectors=(
                "zero_vector",
                "sum"
            ),
        )
    )

    print(
        by_group.round(4).to_string()
    )

    return summary, by_intent, by_group


def analyze_class_weights(
    vectorizer,
    classifier
):

    print("\n=== D04: CLASS FEATURE WEIGHTS ===")

    feature_names = (
        vectorizer.get_feature_names_out()
    )

    rows = []

    for class_index, intent in enumerate(
        classifier.classes_
    ):

        coefficients = (
            classifier.coef_[class_index]
        )

        positive_indices = np.argsort(
            coefficients
        )[::-1][:TOP_FEATURES]

        negative_indices = np.argsort(
            coefficients
        )[:TOP_FEATURES]

        for direction, indices in [

            ("positive", positive_indices),

            ("negative", negative_indices),
        ]:

            for rank, index in enumerate(
                indices,
                start=1
            ):

                rows.append({

                    "intent": intent,

                    "direction": direction,

                    "rank": rank,

                    "feature":
                        feature_names[index],

                    "coefficient":
                        coefficients[index],
                })

    weights = pd.DataFrame(rows)

    for intent in INTENT_CLASSES:

        print(f"\n--- {intent} ---")

        subset = weights.loc[
            weights["intent"].eq(intent)
        ]

        for direction in [
            "positive",
            "negative"
        ]:

            print(f"\n{direction.upper()} FEATURES:")

            print(
                subset.loc[
                    subset["direction"].eq(direction),
                    [
                        "feature",
                        "coefficient",
                    ]
                ].round(4).to_string(index=False)
            )

    return weights


def explain_prediction(
    text,
    vectorizer,
    classifier,
    expected=None,
    record_id=""
):

    model_text = remove_punctuation(text)

    vector = vectorizer.transform(
        [model_text]
    ).tocsr()

    feature_names = (
        vectorizer.get_feature_names_out()
    )

    probabilities = (
        classifier.predict_proba(vector)[0]
    )

    logits = (
        classifier.decision_function(vector)[0]
    )

    classes = classifier.classes_

    ranked_indices = np.argsort(
        -probabilities,
        kind="stable"
    )

    predicted_index = ranked_indices[0]

    predicted = classes[predicted_index]

    active_indices = vector.indices

    active_values = vector.data

    selected_indices = list(
        ranked_indices[:3]
    )

    if expected is not None:

        expected_index = int(
            np.where(classes == expected)[0][0]
        )

        if expected_index not in selected_indices:

            selected_indices.append(
                expected_index
            )

    print("\n" + "=" * 70)

    print(f"ID: {record_id}")

    print(f"Text: {text}")

    print(f"Model text: {model_text}")

    print(f"Expected: {expected}")

    print(f"Predicted: {predicted}")

    print(
        f"Nonzero features: {vector.nnz}"
    )

    print("\nTop-3 predictions:")

    for index in ranked_indices[:3]:

        print(
            f"{classes[index]}: "
            f"{probabilities[index]:.4f}"
        )

    rows = []

    for class_index in selected_indices:

        intent = classes[class_index]

        intercept = (
            classifier.intercept_[class_index]
        )

        contributions = (

            active_values

            * classifier.coef_[
                class_index,
                active_indices
            ]
        )

        reconstructed_logit = (
            intercept + contributions.sum()
        )

        if not np.isclose(
            reconstructed_logit,
            logits[class_index]
        ):

            raise ValueError(
                "Logit reconstruction failed."
            )

        print(f"\nClass: {intent}")

        print(
            f"Probability: "
            f"{probabilities[class_index]:.4f}"
        )

        print(
            f"Intercept: {intercept:.4f}"
        )

        print(
            f"Logit: {logits[class_index]:.4f}"
        )

        feature_rows = []

        for index, value, contribution in zip(

            active_indices,

            active_values,

            contributions
        ):

            feature_rows.append({

                "id": record_id,

                "text": text,

                "expected": expected,

                "predicted": predicted,

                "analyzed_class": intent,

                "feature": feature_names[index],

                "tfidf_value": value,

                "coefficient":
                    classifier.coef_[
                        class_index,
                        index
                    ],

                "contribution": contribution,

                "intercept": intercept,

                "logit": logits[class_index],

                "probability":
                    probabilities[class_index],
            })

        feature_rows.sort(

            key=lambda row: abs(
                row["contribution"]
            ),

            reverse=True
        )

        if feature_rows:

            print(
                pd.DataFrame(feature_rows)[
                    [
                        "feature",
                        "tfidf_value",
                        "coefficient",
                        "contribution",
                    ]
                ].head(10)
                .round(4)
                .to_string(index=False)
            )

        else:

            print(
                "No active features. "
                "Prediction depends on intercepts."
            )

        rows.extend(feature_rows)

    return rows


def analyze_prediction_explanations(
    dataset,
    predictions,
    vectorizer,
    classifier
):

    print("\n=== D05: PREDICTION EXPLANATIONS ===")

    rows = []

    errors = predictions.loc[
        predictions["expected"]
        != predictions["predicted"]
    ]

    for intent in PROBLEM_CLASSES:

        subset = errors.loc[
            errors["expected"].eq(intent)
        ]

        # Select up to three errors per class.
        # Selection is deterministic.
        subset = subset.assign(

            numeric_score=pd.to_numeric(
                subset["model_score"]
            )

        ).sort_values(

            ["numeric_score", "id"],

            ascending=[False, True]

        ).head(3)

        for row in subset.itertuples(
            index=False
        ):

            rows.extend(

                explain_prediction(

                    row.text,

                    vectorizer,

                    classifier,

                    expected=row.expected,

                    record_id=row.id
                )
            )

    print("\n=== ADDITIONAL DIAGNOSTIC TEXTS ===")

    for i, text in enumerate(
        DIAGNOSTIC_TEXTS,
        start=1
    ):

        rows.extend(

            explain_prediction(

                text,

                vectorizer,

                classifier,

                record_id=f"DIAG-{i:03d}"
            )
        )

    return pd.DataFrame(rows)


def analyze_training_distribution(vectorizer):

    print("\n=== D06: TRAINING DISTRIBUTION ===")

    switchboard = load_dataset(
        SWITCHBOARD_TRAIN_PATH
    )

    domain = load_dataset(
        DOMAIN_TRAIN_PATH
    )

    rows = []

    for name, dataset in [

        ("switchboard", switchboard),

        ("digital_mirror", domain),
    ]:

        print(f"\n--- {name.upper()} ---")

        print(f"Records: {len(dataset):,}")

        for intent in INTENT_CLASSES:

            subset = dataset.loc[
                dataset["intent"].eq(intent)
            ]

            word_counts = (
                subset["model_text"]
                .str.split()
                .str.len()
            )

            vectors = vectorizer.transform(
                subset["model_text"]
            )

            zero_count = int(

                np.sum(
                    vectors.getnnz(axis=1) == 0
                )
            )

            rows.append({

                "dataset": name,

                "intent": intent,

                "records": len(subset),

                "unique_texts":
                    subset["model_text"].nunique(),

                "mean_words":
                    word_counts.mean(),

                "short_utterances":
                    int(word_counts.le(3).sum()),

                "short_percentage": (
                    word_counts.le(3).mean()
                    if len(subset)
                    else 0.0
                ),

                "zero_vectors": zero_count,
            })

            if intent in PROBLEM_CLASSES:

                print(f"\n{intent}")

                print(
                    subset["model_text"]
                    .value_counts()
                    .head(10)
                    .to_string()
                )

    distribution = pd.DataFrame(rows)

    print("\n=== TRAINING CLASS STATISTICS ===")

    print(
        distribution.round(4)
        .to_string(index=False)
    )

    return distribution


def analyze_intercepts(
    vectorizer,
    classifier
):

    print("\n=== D07: INTERCEPT ANALYSIS ===")

    zero_vector = vectorizer.transform([""])

    if zero_vector.nnz != 0:

        raise ValueError(
            "Expected a zero feature vector."
        )

    probabilities = (
        classifier.predict_proba(
            zero_vector
        )[0]
    )

    intercepts = pd.DataFrame({

        "intent": classifier.classes_,

        "intercept": classifier.intercept_,

        "zero_vector_probability":
            probabilities,
    })

    intercepts = intercepts.sort_values(

        "zero_vector_probability",

        ascending=False
    )

    print(
        intercepts.round(4)
        .to_string(index=False)
    )

    print(
        "\nDefault zero-vector prediction: "
        f"{intercepts.iloc[0]['intent']}"
    )

    print("\nZero-vector diagnostic examples:")

    for text in ["Howdy", "Cheerio"]:

        normalized = remove_punctuation(text)

        vector = vectorizer.transform(
            [normalized]
        )

        print(
            f"{text}: "
            f"nonzero_features={vector.nnz}"
        )

    return intercepts


def main():

    print(
        "=== E09.07.1: FEATURE COVERAGE "
        "AND MODEL DIAGNOSTICS ==="
    )

    EXPERIMENTS_DIR.mkdir(

        parents=True,

        exist_ok=True
    )

    verify_frozen_files()

    dataset = load_dataset(DATASET_PATH)

    predictions = load_dataset(
        PREDICTIONS_PATH
    )

    validate_predictions(
        dataset,
        predictions
    )

    model = joblib.load(MODEL_PATH)

    vectorizer = model.named_steps["tfidf"]

    classifier = model.named_steps["classifier"]

    print("\n=== MODEL INFORMATION ===")

    print(
        f"Vocabulary size: "
        f"{len(vectorizer.vocabulary_):,}"
    )

    print(f"Analyzer: {vectorizer.analyzer}")

    print(f"N-gram range: {vectorizer.ngram_range}")

    print(f"Min document frequency: {vectorizer.min_df}")

    print(f"Max features: {vectorizer.max_features}")

    print(
        f"Coefficient matrix shape: "
        f"{classifier.coef_.shape}"
    )

    print(
        f"Intercept shape: "
        f"{classifier.intercept_.shape}"
    )

    if set(classifier.classes_) != set(INTENT_CLASSES):

        raise ValueError(
            "Model classes do not match taxonomy."
        )

    # Verify previously saved predictions.
    current_predictions = model.predict(
        dataset["model_text"]
    )

    if not np.array_equal(

        current_predictions,

        predictions["predicted"].to_numpy()
    ):

        raise ValueError(
            "Saved predictions do not match "
            "the frozen model."
        )

    print("Saved predictions verified: YES")

    coverage, zero_vectors = (

        analyze_feature_coverage(

            dataset,

            predictions,

            vectorizer
        )
    )

    summary, by_intent, by_group = (

        analyze_coverage_vs_accuracy(
            coverage
        )
    )

    weights = analyze_class_weights(

        vectorizer,

        classifier
    )

    explanations = (

        analyze_prediction_explanations(

            dataset,

            predictions,

            vectorizer,

            classifier
        )
    )

    distribution = analyze_training_distribution(
        vectorizer
    )

    intercepts = analyze_intercepts(

        vectorizer,

        classifier
    )

    outputs = {

        "feature_coverage":
            (coverage, False),

        "zero_vectors":
            (zero_vectors, False),

        "coverage_summary":
            (summary, False),

        "coverage_by_intent":
            (by_intent, True),

        "coverage_by_group":
            (by_group, True),

        "class_feature_weights":
            (weights, False),

        "prediction_explanations":
            (explanations, False),

        "training_distribution":
            (distribution, False),

        "intercepts":
            (intercepts, False),
    }

    print("\n=== SAVED DIAGNOSTIC FILES ===")

    for name, (frame, save_index) in outputs.items():

        path = (

            EXPERIMENTS_DIR

            / f"{EXPERIMENT}_{name}.csv"
        )

        frame.to_csv(

            path,

            index=save_index,

            encoding="utf-8"
        )

        print(path.name)

    verify_frozen_files()

    print("\n=== EXPERIMENT INTEGRITY ===")

    print("Model unchanged: YES")

    print("Dataset unchanged: YES")

    print("Model training performed: NO")

    print("Hyperparameter tuning performed: NO")

    print("\n=== MODEL DIAGNOSTICS COMPLETE ===")


if __name__ == "__main__":
    main()
