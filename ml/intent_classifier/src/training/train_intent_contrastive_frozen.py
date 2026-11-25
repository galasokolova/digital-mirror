
import hashlib
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline


ML_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ML_ROOT / "data"
MODELS_DIR = ML_ROOT / "models"
EXPERIMENTS_DIR = ML_ROOT / "experiments"

EXPERIMENT = "E09_07_06_05_01"

TEXT_COLUMN = "model_text"
TARGET_COLUMN = "intent"

EXPECTED_FEATURES = 177043
EXPECTED_TRAINING_RECORDS = 137605

INTENT_CLASSES = [
    "GREETING",
    "GOODBYE",
    "GRATITUDE",
    "AGREEMENT",
    "DISAGREEMENT",
    "UNCERTAINTY",
    "ACKNOWLEDGMENT",
    "AFFIRMATIVE_ANSWER",
    "NEGATIVE_ANSWER",
    "QUESTION",
    "CLARIFICATION",
    "STATEMENT",
]

SWITCHBOARD_DIR = (
    DATA_DIR / "processed" / "intent_v1"
)

DOMAIN_PATH = (
    DATA_DIR
    / "processed"
    / "digital_mirror_v1"
    / "train_eligible.csv"
)

CONTRASTIVE_PATH = (
    DATA_DIR
    / "processed"
    / "digital_mirror_contrastive_v1"
    / "train_eligible.csv"
)

BENCHMARK_PATH = (
    DATA_DIR
    / "evaluation"
    / "digital_mirror_final"
    / "test_processed.csv"
)

REFERENCE_MODEL_PATH = (
    MODELS_DIR
    / "E09_07_03_combined_logreg.joblib"
)

STRATEGIES = {
    "a": {
        "name": "uniform",
        "alpha": 0.0,
        "baseline": "E09_07_05a_lr.joblib",
    },
    "b": {
        "name": "moderate",
        "alpha": 0.5,
        "baseline": "E09_07_05b_lr.joblib",
    },
    "c": {
        "name": "strong",
        "alpha": 1.0,
        "baseline": "E09_07_05c_lr.joblib",
    },
}


def calculate_sha256(path):

    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def save_table(table, name):

    EXPERIMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    path = (
        EXPERIMENTS_DIR
        / f"{EXPERIMENT}_{name}.csv"
    )

    table.to_csv(
        path,
        index=False,
        encoding="utf-8"
    )

    print(f"Saved: {path.name}")


def load_dataset(path, expected, name):

    dataset = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False
    )

    required = {
        TEXT_COLUMN,
        TARGET_COLUMN,
    }

    if not required.issubset(dataset.columns):

        raise ValueError(
            f"Missing columns: {name}"
        )

    if len(dataset) != expected:

        raise ValueError(
            f"Unexpected size: {name}"
        )

    if not dataset[TEXT_COLUMN].str.strip().ne("").all():

        raise ValueError(
            f"Empty model texts: {name}"
        )

    if set(dataset[TARGET_COLUMN]) != set(INTENT_CLASSES):

        raise ValueError(
            f"Unexpected intent classes: {name}"
        )

    print(f"{name}: {len(dataset)} records")

    return dataset


def load_datasets():

    print("\n=== DATASET LOADING ===")

    switchboard = load_dataset(
        SWITCHBOARD_DIR / "train.csv",
        135827,
        "Switchboard Train"
    )

    domain = load_dataset(
        DOMAIN_PATH,
        579,
        "Digital Mirror Eligible"
    )

    contrastive = load_dataset(
        CONTRASTIVE_PATH,
        1199,
        "Contrastive Eligible"
    )

    validation = load_dataset(
        SWITCHBOARD_DIR / "validation.csv",
        16913,
        "Switchboard Validation"
    )

    benchmark = load_dataset(
        BENCHMARK_PATH,
        240,
        "Installation Benchmark"
    )

    return (
        switchboard,
        domain,
        contrastive,
        validation,
        benchmark,
    )


def prepare_training_data(
    switchboard,
    domain,
    contrastive
):

    columns = [
        TEXT_COLUMN,
        TARGET_COLUMN,
    ]

    combined = pd.concat(
        [
            switchboard[columns].assign(
                training_source="switchboard"
            ),

            domain[columns].assign(
                training_source="digital_mirror"
            ),

            contrastive[columns].assign(
                training_source="contrastive"
            ),
        ],
        ignore_index=True
    )

    print("\n=== COMBINED TRAINING DATA ===")

    print(
        combined["training_source"]
        .value_counts()
        .to_string()
    )

    print(f"\nTotal records: {len(combined)}")

    print("\n=== CLASS DISTRIBUTION ===")

    print(
        combined[TARGET_COLUMN]
        .value_counts()
        .reindex(INTENT_CLASSES)
        .to_string()
    )

    return combined


def validate_datasets(
    switchboard,
    domain,
    contrastive,
    combined,
    validation,
    benchmark
):

    print("\n=== DATA VALIDATION ===")

    new_texts = set(
        contrastive[TEXT_COLUMN]
    )

    training_texts = set(
        combined[TEXT_COLUMN]
    )

    benchmark_counts = (
        benchmark[TARGET_COLUMN]
        .value_counts()
    )

    checks = {
        "Expected training size":
            len(combined)
            == EXPECTED_TRAINING_RECORDS,

        "All training classes retained":
            set(combined[TARGET_COLUMN])
            == set(INTENT_CLASSES),

        "No empty training texts":
            combined[TEXT_COLUMN]
            .str.strip()
            .ne("")
            .all(),

        "Unique contrastive IDs":
            contrastive["id"].is_unique,

        "Unique contrastive model texts":
            contrastive[TEXT_COLUMN].is_unique,

        "No contrastive / Switchboard Train overlap":
            not new_texts.intersection(
                switchboard[TEXT_COLUMN]
            ),

        "No contrastive / Digital Mirror overlap":
            not new_texts.intersection(
                domain[TEXT_COLUMN]
            ),

        "No contrastive / Validation overlap":
            not new_texts.intersection(
                validation[TEXT_COLUMN]
            ),

        "No contrastive / Benchmark overlap":
            not new_texts.intersection(
                benchmark[TEXT_COLUMN]
            ),

        "No Digital Mirror / Validation overlap":
            not set(domain[TEXT_COLUMN]).intersection(
                validation[TEXT_COLUMN]
            ),

        "No training / Benchmark overlap":
            not training_texts.intersection(
                benchmark[TEXT_COLUMN]
            ),

        "Unique benchmark model texts":
            benchmark[TEXT_COLUMN].is_unique,

        "Balanced benchmark classes":
            benchmark_counts.eq(20).all(),

        "No Switchboard conversation overlap":
            not set(
                switchboard["conversation_no"]
            ).intersection(
                validation["conversation_no"]
            ),

        "Expected contrastive dataset hash":
            calculate_sha256(CONTRASTIVE_PATH)
            == (
                "bc2fc93bff2b9ef8106cd09728d57f595"
                "6eba6dad551e2569b5976766541ccd8"
            ),
    }

    for name, passed in checks.items():

        print(
            f"{'PASS' if passed else 'FAIL'}"
            f" | {name}"
        )

    save_table(
        pd.DataFrame([
            {
                "check": name,
                "passed": bool(passed)
            }
            for name, passed in checks.items()
        ]),
        "data_checks"
    )

    print(
        f"\nChecks passed: "
        f"{sum(checks.values())}/{len(checks)}"
    )

    if not all(checks.values()):

        raise ValueError(
            "Data validation failed."
        )


def calculate_frequencies(dataset):

    return (
        dataset.groupby(
            [TEXT_COLUMN, TARGET_COLUMN]
        )[TEXT_COLUMN]
        .transform("size")
        .to_numpy(dtype=float)
    )


def calculate_sample_weights(
    dataset,
    frequencies,
    alpha
):

    raw_weights = np.power(
        frequencies,
        -alpha
    )

    weights = np.zeros(
        len(dataset),
        dtype=float
    )

    labels = dataset[
        TARGET_COLUMN
    ].to_numpy()

    target_class_weight = (
        len(dataset) / len(INTENT_CLASSES)
    )

    for intent in INTENT_CLASSES:

        mask = labels == intent

        class_raw_weights = raw_weights[mask]

        weights[mask] = (
            class_raw_weights
            / class_raw_weights.sum()
            * target_class_weight
        )

    if not np.isfinite(weights).all():

        raise ValueError(
            "Non-finite sample weights."
        )

    if not (weights > 0).all():

        raise ValueError(
            "Non-positive sample weights."
        )

    if not np.isclose(
        weights.sum(),
        len(dataset)
    ):

        raise ValueError(
            "Incorrect total sample weight."
        )

    for intent in INTENT_CLASSES:

        mask = labels == intent

        if not np.isclose(
            weights[mask].sum(),
            target_class_weight
        ):

            raise ValueError(
                f"Incorrect class weight: {intent}"
            )

    return weights


def analyze_weights(
    dataset,
    weights,
    frequencies,
    strategy
):

    print("\n=== SAMPLE WEIGHT ANALYSIS ===")

    print(f"Strategy: {strategy}")

    print(f"Minimum: {weights.min():.8f}")
    print(f"Maximum: {weights.max():.8f}")
    print(f"Mean: {weights.mean():.8f}")
    print(f"Total: {weights.sum():.4f}")

    weighted = dataset.assign(
        sample_weight=weights,
        frequency=frequencies
    )

    source_stats = (
        weighted.groupby("training_source")
        ["sample_weight"]
        .agg(
            records="size",
            total_weight="sum",
            mean_weight="mean",
            min_weight="min",
            max_weight="max"
        )
        .reset_index()
    )

    source_stats["weight_share"] = (
        source_stats["total_weight"]
        / weights.sum()
    )

    print("\n=== SOURCE WEIGHTS ===")

    print(
        source_stats.to_string(index=False)
    )

    save_table(
        source_stats,
        f"{strategy}_source_weights"
    )

    class_stats = (
        weighted.groupby(TARGET_COLUMN)
        ["sample_weight"]
        .agg(
            records="size",
            total_weight="sum",
            mean_weight="mean",
            min_weight="min",
            max_weight="max"
        )
        .reset_index()
    )

    print("\n=== CLASS WEIGHTS ===")

    print(
        class_stats.to_string(index=False)
    )

    save_table(
        class_stats,
        f"{strategy}_class_weights"
    )

    source_class_stats = (
        weighted.groupby(
            ["training_source", TARGET_COLUMN]
        )["sample_weight"]
        .agg(
            records="size",
            total_weight="sum",
            mean_weight="mean"
        )
        .reset_index()
    )

    save_table(
        source_class_stats,
        f"{strategy}_source_class_weights"
    )

    ess_rows = []

    for intent in INTENT_CLASSES:

        mask = (
            weighted[TARGET_COLUMN] == intent
        )

        class_weights = weights[
            mask.to_numpy()
        ]

        ess = (
            class_weights.sum() ** 2
            / np.square(class_weights).sum()
        )

        ess_rows.append({
            "intent": intent,
            "records": len(class_weights),
            "effective_sample_size": ess,
        })

    save_table(
        pd.DataFrame(ess_rows),
        f"{strategy}_effective_sample_size"
    )


def verify_feature_space(
    reference_features,
    baseline_features
):

    reference_transformers = dict(
        reference_features.transformer_list
    )

    baseline_transformers = dict(
        baseline_features.transformer_list
    )

    if (
        list(reference_transformers)
        != list(baseline_transformers)
    ):

        raise ValueError(
            "Feature transformer mismatch."
        )

    if (
        reference_features.transformer_weights
        != baseline_features.transformer_weights
    ):

        raise ValueError(
            "Transformer weight mismatch."
        )

    for name, reference in reference_transformers.items():

        baseline = baseline_transformers[name]

        if (
            reference.get_params()
            != baseline.get_params()
        ):

            raise ValueError(
                f"Vectorizer parameter mismatch: {name}"
            )

        if (
            reference.vocabulary_
            != baseline.vocabulary_
        ):

            raise ValueError(
                f"Vocabulary mismatch: {name}"
            )

        if not np.array_equal(
            reference.idf_,
            baseline.idf_
        ):

            raise ValueError(
                f"IDF mismatch: {name}"
            )

    if not np.array_equal(
        reference_features.get_feature_names_out(),
        baseline_features.get_feature_names_out()
    ):

        raise ValueError(
            "Feature order mismatch."
        )

    print("Frozen feature space verification: PASS")


def train_model(
    reference_features,
    baseline_classifier,
    training_matrix,
    labels,
    sample_weights,
    model_name
):

    print(f"\n=== TRAINING {model_name} ===")

    classifier = clone(
        baseline_classifier
    )

    if classifier.class_weight is not None:

        raise ValueError(
            "Expected class_weight=None. "
            "Class balancing is included "
            "in sample_weight."
        )

    print(
        f"Classifier parameters: "
        f"{classifier.get_params()}"
    )

    feature_hash_before = joblib.hash(
        reference_features
    )

    started = time.perf_counter()

    with warnings.catch_warnings(
        record=True
    ) as caught:

        warnings.simplefilter(
            "always",
            ConvergenceWarning
        )

        classifier.fit(
            training_matrix,
            labels,
            sample_weight=sample_weights
        )

    training_time = (
        time.perf_counter() - started
    )

    convergence_warnings = [
        warning
        for warning in caught
        if issubclass(
            warning.category,
            ConvergenceWarning
        )
    ]

    print(
        f"Training time: "
        f"{training_time:.4f} seconds"
    )

    print(
        f"Iterations: "
        f"{classifier.n_iter_.tolist()}"
    )

    print(
        f"Convergence warnings: "
        f"{len(convergence_warnings)}"
    )

    if convergence_warnings:

        raise RuntimeError(
            f"Convergence failed: {model_name}"
        )

    if classifier.coef_.shape != (
        len(INTENT_CLASSES),
        EXPECTED_FEATURES
    ):

        raise ValueError(
            "Unexpected coefficient shape."
        )

    if not np.isfinite(
        classifier.coef_
    ).all():

        raise ValueError(
            "Non-finite coefficients."
        )

    if not np.isfinite(
        classifier.intercept_
    ).all():

        raise ValueError(
            "Non-finite intercepts."
        )

    if set(classifier.classes_) != set(
        INTENT_CLASSES
    ):

        raise ValueError(
            "Unexpected classifier classes."
        )

    if joblib.hash(
        reference_features
    ) != feature_hash_before:

        raise RuntimeError(
            "Frozen feature extractor changed."
        )

    model = Pipeline([
        ("tfidf", reference_features),
        ("classifier", classifier),
    ])

    model_path = (
        MODELS_DIR / f"{model_name}.joblib"
    )

    joblib.dump(model, model_path)

    reloaded = joblib.load(model_path)

    verify_feature_space(
        reference_features,
        reloaded.named_steps["tfidf"]
    )

    if not np.array_equal(
        classifier.coef_,
        reloaded.named_steps["classifier"].coef_
    ):

        raise RuntimeError(
            "Saved classifier verification failed."
        )

    if not np.array_equal(
        classifier.intercept_,
        reloaded.named_steps["classifier"].intercept_
    ):

        raise RuntimeError(
            "Saved intercept verification failed."
        )

    print(f"Saved model: {model_path.name}")

    print(
        f"Model SHA256: "
        f"{calculate_sha256(model_path)}"
    )

    print("Frozen features unchanged: YES")
    print("Saved model verification: PASS")

    return reloaded, training_time


def evaluate_model(
    model,
    dataset,
    feature_matrix,
    model_name,
    dataset_name
):

    print(
        f"\n=== EVALUATION: "
        f"{model_name} / {dataset_name} ==="
    )

    classifier = model.named_steps[
        "classifier"
    ]

    expected = dataset[
        TARGET_COLUMN
    ].to_numpy()

    predicted = classifier.predict(
        feature_matrix
    )

    probabilities = classifier.predict_proba(
        feature_matrix
    )

    classes = classifier.classes_

    top3_indices = np.argsort(
        -probabilities,
        axis=1,
        kind="stable"
    )[:, :3]

    top3_labels = classes[top3_indices]

    top3_correct = (
        top3_labels == expected[:, None]
    ).any(axis=1)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            expected,
            predicted,
            labels=INTENT_CLASSES,
            average="macro",
            zero_division=0
        )
    )

    _, _, weighted_f1, _ = (
        precision_recall_fscore_support(
            expected,
            predicted,
            labels=INTENT_CLASSES,
            average="weighted",
            zero_division=0
        )
    )

    correct = expected == predicted

    statement_fp = (
        (predicted == "STATEMENT")
        & (expected != "STATEMENT")
    ).sum()

    metrics = {
        "model": model_name,
        "dataset": dataset_name,
        "records": len(dataset),
        "correct": int(correct.sum()),
        "accuracy": accuracy_score(
            expected,
            predicted
        ),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "weighted_f1": weighted_f1,
        "top3_accuracy": top3_correct.mean(),
        "statement_false_positives":
            int(statement_fp),
    }

    for key, value in metrics.items():

        print(f"{key}: {value}")

    print("\n=== CLASSIFICATION REPORT ===")

    print(
        classification_report(
            expected,
            predicted,
            labels=INTENT_CLASSES,
            zero_division=0,
            digits=4
        )
    )

    report = classification_report(
        expected,
        predicted,
        labels=INTENT_CLASSES,
        zero_division=0,
        output_dict=True
    )

    class_summary = pd.DataFrame([
        {
            "intent": intent,
            "precision": report[intent]["precision"],
            "recall": report[intent]["recall"],
            "f1": report[intent]["f1-score"],
            "support": report[intent]["support"],
            "top3_accuracy": top3_correct[
                expected == intent
            ].mean(),
        }
        for intent in INTENT_CLASSES
    ])

    predictions = pd.DataFrame({
        "model_text": dataset[TEXT_COLUMN],
        "expected": expected,
        "predicted": predicted,
        "correct": correct,
        "top3_correct": top3_correct,
        "top3_labels": [
            "|".join(labels)
            for labels in top3_labels
        ],
        "max_probability": probabilities.max(
            axis=1
        ),
    })

    for column in ["id", "text", "group"]:

        if column in dataset.columns:

            predictions[column] = (
                dataset[column].to_numpy()
            )

    matrix = pd.DataFrame(
        confusion_matrix(
            expected,
            predicted,
            labels=INTENT_CLASSES
        ),
        index=INTENT_CLASSES,
        columns=INTENT_CLASSES
    )

    matrix.index.name = "expected"

    prefix = f"{model_name}_{dataset_name}"

    save_table(
        predictions,
        f"{prefix}_predictions"
    )

    save_table(
        class_summary,
        f"{prefix}_class_summary"
    )

    save_table(
        matrix.reset_index(),
        f"{prefix}_confusion_matrix"
    )

    save_table(
        predictions[~predictions["correct"]],
        f"{prefix}_errors"
    )

    if "group" in predictions.columns:

        group_results = (
            predictions.groupby("group")
            .agg(
                records=("correct", "size"),
                correct=("correct", "sum"),
                accuracy=("correct", "mean"),
                top3_accuracy=(
                    "top3_correct", "mean"
                )
            )
            .reset_index()
        )

        print("\n=== GROUP RESULTS ===")

        print(
            group_results.to_string(index=False)
        )

        save_table(
            group_results,
            f"{prefix}_group_results"
        )

    return metrics, predictions, class_summary


def compare_models(
    baseline_metrics,
    new_metrics,
    baseline_predictions,
    new_predictions,
    baseline_classes,
    new_classes,
    strategy,
    dataset_name
):

    print(
        f"\n=== PAIRED COMPARISON: "
        f"{strategy} / {dataset_name} ==="
    )

    metric_names = [
        "correct",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "top3_accuracy",
        "statement_false_positives",
    ]

    comparison = pd.DataFrame([
        {
            "metric": metric,
            "baseline": baseline_metrics[metric],
            "contrastive": new_metrics[metric],
            "difference": (
                new_metrics[metric]
                - baseline_metrics[metric]
            ),
        }
        for metric in metric_names
    ])

    print(
        comparison.to_string(index=False)
    )

    if not baseline_predictions[
        ["model_text", "expected"]
    ].equals(
        new_predictions[
            ["model_text", "expected"]
        ]
    ):

        raise ValueError(
            "Prediction alignment mismatch."
        )

    paired = baseline_predictions[
        ["model_text", "expected"]
    ].copy()

    paired["baseline_prediction"] = (
        baseline_predictions["predicted"]
    )

    paired["contrastive_prediction"] = (
        new_predictions["predicted"]
    )

    before = baseline_predictions[
        "correct"
    ].to_numpy()

    after = new_predictions[
        "correct"
    ].to_numpy()

    paired["status"] = np.select(
        [
            before & after,
            ~before & after,
            before & ~after,
        ],
        [
            "BOTH_CORRECT",
            "IMPROVED",
            "REGRESSED",
        ],
        default="BOTH_INCORRECT"
    )

    print("\n=== PREDICTION CHANGES ===")

    print(
        paired["status"]
        .value_counts()
        .to_string()
    )

    class_comparison = (
        baseline_classes.merge(
            new_classes,
            on="intent",
            suffixes=("_baseline", "_contrastive")
        )
    )

    for metric in [
        "precision",
        "recall",
        "f1",
        "top3_accuracy",
    ]:

        class_comparison[
            f"{metric}_difference"
        ] = (
            class_comparison[
                f"{metric}_contrastive"
            ]
            - class_comparison[
                f"{metric}_baseline"
            ]
        )

    print("\n=== CLASS RECALL COMPARISON ===")

    print(
        class_comparison[[
            "intent",
            "recall_baseline",
            "recall_contrastive",
            "recall_difference",
        ]].to_string(index=False)
    )

    prefix = f"{strategy}_{dataset_name}"

    save_table(
        comparison,
        f"{prefix}_metric_comparison"
    )

    save_table(
        paired,
        f"{prefix}_prediction_comparison"
    )

    save_table(
        class_comparison,
        f"{prefix}_class_comparison"
    )


def main():

    print(
        "=== E09.07.6.5.1: "
        "FROZEN FEATURE SPACE TRAINING ==="
    )

    protected_paths = [
        SWITCHBOARD_DIR / "train.csv",
        SWITCHBOARD_DIR / "validation.csv",
        SWITCHBOARD_DIR / "test.csv",
        DOMAIN_PATH,
        CONTRASTIVE_PATH,
        BENCHMARK_PATH,
        REFERENCE_MODEL_PATH,
        *[
            MODELS_DIR / config["baseline"]
            for config in STRATEGIES.values()
        ],
    ]

    original_hashes = {
        path: calculate_sha256(path)
        for path in protected_paths
    }

    results = []

    try:

        (
            switchboard,
            domain,
            contrastive,
            validation,
            benchmark,
        ) = load_datasets()

        combined = prepare_training_data(
            switchboard,
            domain,
            contrastive
        )

        validate_datasets(
            switchboard,
            domain,
            contrastive,
            combined,
            validation,
            benchmark
        )

        print("\n=== LOADING FROZEN FEATURES ===")

        reference_model = joblib.load(
            REFERENCE_MODEL_PATH
        )

        reference_features = (
            reference_model.named_steps["tfidf"]
        )

        frozen_hash = joblib.hash(
            reference_features
        )

        feature_count = len(
            reference_features
            .get_feature_names_out()
        )

        print(f"Features: {feature_count}")

        if feature_count != EXPECTED_FEATURES:

            raise ValueError(
                "Unexpected feature count."
            )

        print("\n=== TRANSFORMING TRAINING DATA ===")

        started = time.perf_counter()

        training_matrix = (
            reference_features.transform(
                combined[TEXT_COLUMN]
            )
        )

        print(
            f"Matrix shape: "
            f"{training_matrix.shape}"
        )

        print(
            f"Non-zero elements: "
            f"{training_matrix.nnz}"
        )

        print(
            f"Transformation time: "
            f"{time.perf_counter() - started:.4f}"
        )

        if training_matrix.shape != (
            EXPECTED_TRAINING_RECORDS,
            EXPECTED_FEATURES
        ):

            raise ValueError(
                "Unexpected training matrix shape."
            )

        zero_vectors = (
            training_matrix.getnnz(axis=1) == 0
        )

        print("\n=== ZERO VECTOR ANALYSIS ===")

        for source in [
            "switchboard",
            "digital_mirror",
            "contrastive",
        ]:

            mask = (
                combined["training_source"]
                .eq(source)
                .to_numpy()
            )

            print(
                f"{source}: "
                f"{zero_vectors[mask].sum()}"
            )

        save_table(
            combined.loc[zero_vectors],
            "zero_vector_records"
        )

        validation_matrix = (
            reference_features.transform(
                validation[TEXT_COLUMN]
            )
        )

        benchmark_matrix = (
            reference_features.transform(
                benchmark[TEXT_COLUMN]
            )
        )

        frequencies = calculate_frequencies(
            combined
        )

        contrastive_mask = (
            combined["training_source"]
            .eq("contrastive")
            .to_numpy()
        )

        if not (
            frequencies[contrastive_mask] == 1
        ).all():

            raise ValueError(
                "Unexpected contrastive frequencies."
            )

        labels = combined[
            TARGET_COLUMN
        ].to_numpy()

        evaluation_sets = [
            (
                "validation",
                validation,
                validation_matrix
            ),
            (
                "benchmark",
                benchmark,
                benchmark_matrix
            ),
        ]

        for strategy, config in STRATEGIES.items():

            print("\n" + "=" * 60)

            print(
                f"STRATEGY: {config['name']} | "
                f"ALPHA: {config['alpha']}"
            )

            print("=" * 60)

            baseline_model = joblib.load(
                MODELS_DIR / config["baseline"]
            )

            verify_feature_space(
                reference_features,
                baseline_model.named_steps["tfidf"]
            )

            baseline_classifier = (
                baseline_model.named_steps[
                    "classifier"
                ]
            )

            weights = calculate_sample_weights(
                combined,
                frequencies,
                config["alpha"]
            )

            analyze_weights(
                combined,
                weights,
                frequencies,
                strategy
            )

            model_name = (
                f"{EXPERIMENT}{strategy}_lr"
            )

            new_model, training_time = (
                train_model(
                    reference_features,
                    baseline_classifier,
                    training_matrix,
                    labels,
                    weights,
                    model_name
                )
            )

            for (
                dataset_name,
                dataset,
                matrix
            ) in evaluation_sets:

                baseline_metrics, baseline_predictions, baseline_classes = (
                    evaluate_model(
                        baseline_model,
                        dataset,
                        matrix,
                        f"baseline_{strategy}_lr",
                        dataset_name
                    )
                )

                new_metrics, new_predictions, new_classes = (
                    evaluate_model(
                        new_model,
                        dataset,
                        matrix,
                        model_name,
                        dataset_name
                    )
                )

                baseline_metrics[
                    "alpha"
                ] = config["alpha"]

                baseline_metrics[
                    "training_seconds"
                ] = np.nan

                new_metrics[
                    "alpha"
                ] = config["alpha"]

                new_metrics[
                    "training_seconds"
                ] = training_time

                results.extend([
                    baseline_metrics,
                    new_metrics
                ])

                compare_models(
                    baseline_metrics,
                    new_metrics,
                    baseline_predictions,
                    new_predictions,
                    baseline_classes,
                    new_classes,
                    strategy,
                    dataset_name
                )

                save_table(
                    pd.DataFrame(results),
                    "results"
                )

        if joblib.hash(
            reference_features
        ) != frozen_hash:

            raise RuntimeError(
                "Frozen feature extractor changed."
            )

        print("\n=== FINAL RESULTS ===")

        results_table = pd.DataFrame(results)

        print(
            results_table.to_string(index=False)
        )

        print("\n=== EXPERIMENT SUMMARY ===")

        print("Models trained: 3")
        print("Training records: 137605")
        print("TF-IDF refitted: NO")
        print("Feature count changed: NO")
        print("Frozen feature extractor changed: NO")
        print("Switchboard Test evaluated: NO")
        print("Hyperparameter tuning: NO")

    finally:

        print("\n=== FILE INTEGRITY ===")

        integrity_rows = []

        for path, original_hash in original_hashes.items():

            current_hash = calculate_sha256(
                path
            )

            unchanged = (
                original_hash == current_hash
            )

            print(
                f"{'PASS' if unchanged else 'FAIL'}"
                f" | {path.relative_to(ML_ROOT)}"
            )

            integrity_rows.append({
                "path": str(
                    path.relative_to(ML_ROOT)
                ),
                "original_sha256": original_hash,
                "current_sha256": current_hash,
                "passed": unchanged,
            })

        integrity_table = pd.DataFrame(
            integrity_rows
        )

        save_table(
            integrity_table,
            "file_integrity"
        )

        print(
            f"\nIntegrity checks passed: "
            f"{integrity_table['passed'].sum()}"
            f"/{len(integrity_table)}"
        )

        if not integrity_table["passed"].all():

            raise RuntimeError(
                "Protected files were modified."
            )

    print("\n=== EXPERIMENT FINISHED ===")


if __name__ == "__main__":
    main()
