
from time import perf_counter
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning

from train_intent_combined_logreg import (
    calculate_sha256,
    load_datasets,
    validate_datasets,
    prepare_training_data,
    INTENT_CLASSES,
    TEXT_COLUMN,
    TARGET_COLUMN,
    MODELS_DIR,
    EXPERIMENTS_DIR,
    SWITCHBOARD_DIR,
    DOMAIN_PATH,
    BENCHMARK_PATH,
)

import train_intent_combined_svm as evaluation


EXPERIMENT = "E09_07_05"

LR_MODEL_PATH = (
    MODELS_DIR
    / "E09_07_03_combined_logreg.joblib"
)

SVM_MODEL_PATH = (
    MODELS_DIR
    / "E09_07_04_combined_svm.joblib"
)

STRATEGIES = {
    "a": ("uniform", 0.0),
    "b": ("inverse_sqrt", 0.5),
    "c": ("inverse_frequency", 1.0),
}


def calculate_frequencies(dataset):

    frequencies = (
        dataset.groupby(
            [TEXT_COLUMN, TARGET_COLUMN]
        )[TEXT_COLUMN]
        .transform("size")
        .to_numpy(dtype=float)
    )

    if len(frequencies) != len(dataset):

        raise ValueError(
            "Frequency array size mismatch."
        )

    if np.any(frequencies < 1):

        raise ValueError(
            "Invalid training frequencies."
        )

    return frequencies


def calculate_sample_weights(
    dataset,
    frequencies,
    alpha
):

    labels = dataset[
        TARGET_COLUMN
    ].to_numpy()

    total_records = len(dataset)

    total_classes = len(INTENT_CLASSES)

    # Frequency-based weights

    raw_weights = np.power(
        frequencies,
        -alpha
    )

    normalized_weights = (
        raw_weights.copy()
    )

    final_weights = np.zeros(
        total_records,
        dtype=float
    )

    expected_class_weight = (
        total_records / total_classes
    )

    class_statistics = []

    for intent in INTENT_CLASSES:

        mask = labels == intent

        class_count = int(mask.sum())

        if class_count == 0:

            raise ValueError(
                f"Missing class: {intent}"
            )

        # Normalize within each class

        normalized_weights[mask] = (
            raw_weights[mask]
            / raw_weights[mask].mean()
        )

        # Fixed balanced class weight

        class_weight = (
            total_records
            / (total_classes * class_count)
        )

        final_weights[mask] = (
            normalized_weights[mask]
            * class_weight
        )

        actual_class_weight = (
            final_weights[mask].sum()
        )

        if not np.isclose(
            actual_class_weight,
            expected_class_weight,
            rtol=1e-10,
            atol=1e-8
        ):

            raise ValueError(
                f"Class weight mismatch: {intent}"
            )

        # Effective sample size

        class_weights = final_weights[mask]

        effective_sample_size = (
            class_weights.sum() ** 2
            / np.square(class_weights).sum()
        )

        class_statistics.append({
            "intent": intent,
            "records": class_count,
            "unique_texts": (
                dataset.loc[
                    mask,
                    TEXT_COLUMN
                ].nunique()
            ),
            "class_weight": class_weight,
            "raw_weight_sum": (
                raw_weights[mask].sum()
            ),
            "normalized_weight_sum": (
                normalized_weights[mask].sum()
            ),
            "final_weight_sum": (
                actual_class_weight
            ),
            "min_final_weight": (
                class_weights.min()
            ),
            "max_final_weight": (
                class_weights.max()
            ),
            "effective_sample_size": (
                effective_sample_size
            ),
        })

    if not np.isfinite(final_weights).all():

        raise ValueError(
            "Non-finite sample weights."
        )

    if np.any(final_weights <= 0):

        raise ValueError(
            "Non-positive sample weights."
        )

    if not np.isclose(
        final_weights.sum(),
        total_records,
        rtol=1e-10,
        atol=1e-8
    ):

        raise ValueError(
            "Total sample weight mismatch."
        )

    print(
        f"Total weight: "
        f"{final_weights.sum():.4f}"
    )

    print(
        f"Mean weight: "
        f"{final_weights.mean():.4f}"
    )

    print(
        f"Minimum weight: "
        f"{final_weights.min():.8f}"
    )

    print(
        f"Maximum weight: "
        f"{final_weights.max():.8f}"
    )

    print(
        "Class weight normalization: PASS"
    )

    return (
        final_weights,
        pd.DataFrame(class_statistics)
    )


def analyze_weight_distribution(
    dataset,
    frequencies,
    all_weights
):

    print("\n=== FREQUENCY ANALYSIS ===")

    analysis = dataset[
        [TEXT_COLUMN, TARGET_COLUMN]
    ].copy()

    analysis["frequency"] = frequencies

    for name, weights in all_weights.items():

        analysis[name] = weights

    aggregation = {
        "frequency": ("frequency", "first"),
    }

    for name in all_weights:

        aggregation[
            f"{name}_weight_per_record"
        ] = (name, "first")

        aggregation[
            f"{name}_total_weight"
        ] = (name, "sum")

    distribution = (
        analysis.groupby(
            [TARGET_COLUMN, TEXT_COLUMN]
        )
        .agg(**aggregation)
        .reset_index()
        .sort_values(
            "frequency",
            ascending=False
        )
    )

    print(
        f"Training records: "
        f"{len(dataset):,}"
    )

    print(
        f"Unique text-intent pairs: "
        f"{len(distribution):,}"
    )

    print("\nTop 20 frequent pairs:")

    print(
        distribution.head(20)
        .round(4)
        .to_string(index=False)
    )

    for intent in [
        "ACKNOWLEDGMENT",
        "AGREEMENT",
        "AFFIRMATIVE_ANSWER",
        "STATEMENT",
    ]:

        subset = distribution.loc[
            distribution[TARGET_COLUMN].eq(
                intent
            )
        ]

        print(
            f"\n=== {intent}: "
            "TOP 10 FREQUENT EXPRESSIONS ==="
        )

        print(
            subset.head(10)
            .round(4)
            .to_string(index=False)
        )

    distribution.to_csv(
        EXPERIMENTS_DIR
        / f"{EXPERIMENT}_weight_distribution.csv",
        index=False,
        encoding="utf-8"
    )


def analyze_source_weights(
    dataset,
    source_labels,
    all_weights
):

    print("\n=== SOURCE WEIGHT ANALYSIS ===")

    rows = []

    labels = dataset[
        TARGET_COLUMN
    ].to_numpy()

    for strategy, weights in all_weights.items():

        for source in [
            "switchboard",
            "domain"
        ]:

            mask = source_labels == source

            rows.append({
                "strategy": strategy,
                "source": source,
                "records": int(mask.sum()),
                "total_weight": (
                    weights[mask].sum()
                ),
                "weight_share": (
                    weights[mask].sum()
                    / weights.sum()
                ),
                "mean_weight": (
                    weights[mask].mean()
                ),
            })

    summary = pd.DataFrame(rows)

    print(
        summary.round(4)
        .to_string(index=False)
    )

    summary.to_csv(
        EXPERIMENTS_DIR
        / f"{EXPERIMENT}_source_weights.csv",
        index=False,
        encoding="utf-8"
    )

    # Source weights within each intent

    class_rows = []

    for strategy, weights in all_weights.items():

        for intent in INTENT_CLASSES:

            class_mask = labels == intent

            total = weights[class_mask].sum()

            for source in [
                "switchboard",
                "domain"
            ]:

                mask = (
                    class_mask
                    & (source_labels == source)
                )

                class_rows.append({
                    "strategy": strategy,
                    "intent": intent,
                    "source": source,
                    "records": int(mask.sum()),
                    "total_weight": (
                        weights[mask].sum()
                    ),
                    "class_weight_share": (
                        weights[mask].sum()
                        / total
                    ),
                })

    class_summary = pd.DataFrame(
        class_rows
    )

    print(
        "\nDomain weight share "
        "within each intent:"
    )

    print(
        class_summary.loc[
            class_summary["source"].eq(
                "domain"
            )
        ]
        .round(4)
        .to_string(index=False)
    )

    class_summary.to_csv(
        EXPERIMENTS_DIR
        / f"{EXPERIMENT}_source_class_weights.csv",
        index=False,
        encoding="utf-8"
    )


def train_weighted_model(
    reference_model,
    frozen_features,
    X_train,
    y_train,
    sample_weights,
    model_name
):

    print("\n" + "=" * 70)

    print(f"TRAINING: {model_name}")

    print("=" * 70)

    classifier = clone(
        reference_model.named_steps[
            "classifier"
        ]
    )

    # Balanced class weights are already
    # included in sample_weights.

    classifier.set_params(
        class_weight=None
    )

    print(classifier)

    start = perf_counter()

    with warnings.catch_warnings(
        record=True
    ) as caught:

        warnings.simplefilter(
            "always",
            ConvergenceWarning
        )

        classifier.fit(
            X_train,
            y_train,
            sample_weight=sample_weights
        )

    training_time = (
        perf_counter() - start
    )

    convergence_warnings = [
        warning
        for warning in caught
        if issubclass(
            warning.category,
            ConvergenceWarning
        )
    ]

    for warning in caught:

        print(
            f"Training warning: "
            f"{warning.message}"
        )

    print(
        f"Training time: "
        f"{training_time:.4f} seconds"
    )

    print(
        f"Convergence warnings: "
        f"{len(convergence_warnings)}"
    )

    if convergence_warnings:

        raise RuntimeError(
            f"{model_name} did not converge."
        )

    print(
        f"Classifier iterations: "
        f"{classifier.n_iter_}"
    )

    expected_shape = (
        len(INTENT_CLASSES),
        X_train.shape[1]
    )

    if (
        classifier.coef_.shape
        != expected_shape
    ):

        raise ValueError(
            "Unexpected coefficient shape."
        )

    print(
        f"Coefficient shape: "
        f"{classifier.coef_.shape}"
    )

    # Reuse the fitted feature extractor.
    # Pipeline.fit() is deliberately not called.

    model = Pipeline([
        ("tfidf", frozen_features),
        ("classifier", classifier),
    ])

    model_path = (
        MODELS_DIR
        / f"{EXPERIMENT}{model_name}.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    print(f"Model saved: {model_path}")

    print(
        f"Model SHA256: "
        f"{calculate_sha256(model_path)}"
    )

    return model, training_time


def verify_uniform_baseline(
    reference,
    candidate,
    dataset,
    classifier_name,
    dataset_name
):

    print(
        "\n=== UNIFORM BASELINE VERIFICATION ==="
    )

    texts = dataset[TEXT_COLUMN]

    old_predictions = reference.predict(
        texts
    )

    new_predictions = candidate.predict(
        texts
    )

    matching = (
        old_predictions == new_predictions
    )

    agreement = matching.mean()

    print(
        f"Classifier: {classifier_name}"
    )

    print(
        f"Dataset: {dataset_name}"
    )

    print(
        f"Matching predictions: "
        f"{matching.sum()} / {len(dataset)}"
    )

    print(
        f"Prediction agreement: "
        f"{agreement:.6f}"
    )

    if classifier_name == "lr":

        old_scores = (
            reference.predict_proba(texts)
        )

        new_scores = (
            candidate.predict_proba(texts)
        )

    else:

        old_scores = (
            reference.decision_function(texts)
        )

        new_scores = (
            candidate.decision_function(texts)
        )

    old_classes = reference.named_steps[
        "classifier"
    ].classes_

    new_classes = candidate.named_steps[
        "classifier"
    ].classes_

    if not np.array_equal(
        old_classes,
        new_classes
    ):

        raise ValueError(
            "Class order mismatch."
        )

    max_difference = float(
        np.abs(
            old_scores - new_scores
        ).max()
    )

    print(
        f"Maximum score difference: "
        f"{max_difference:.8f}"
    )

    if not matching.all():

        print(
            "WARNING: Uniform baseline "
            "predictions are not identical."
        )

    return {
        "classifier": classifier_name,
        "dataset": dataset_name,
        "records": len(dataset),
        "matching_predictions": int(
            matching.sum()
        ),
        "prediction_agreement": agreement,
        "max_score_difference": (
            max_difference
        ),
    }


def compare_results(results):

    print("\n=== FREQUENCY WEIGHTING COMPARISON ===")

    columns = [
        "model",
        "dataset",
        "correct",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "top3_accuracy",
        "statement_false_positives",
        "training_seconds",
    ]

    print(
        results[columns]
        .round(4)
        .to_string(index=False)
    )

    differences = []

    metrics = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "top3_accuracy",
    ]

    for classifier in ["lr", "svm"]:

        for dataset in [
            "validation",
            "benchmark"
        ]:

            subset = results.loc[
                results["dataset"].eq(dataset)
            ].set_index("model")

            baseline = subset.loc[
                f"a_{classifier}"
            ]

            for strategy in ["b", "c"]:

                candidate = subset.loc[
                    f"{strategy}_{classifier}"
                ]

                for metric in metrics:

                    differences.append({
                        "classifier": classifier,
                        "dataset": dataset,
                        "strategy": strategy,
                        "metric": metric,
                        "baseline": baseline[metric],
                        "candidate": candidate[metric],
                        "difference_pp": (
                            candidate[metric]
                            - baseline[metric]
                        ) * 100,
                    })

    differences = pd.DataFrame(
        differences
    )

    print("\n=== METRIC DIFFERENCES ===")

    print(
        differences.round(4)
        .to_string(index=False)
    )

    differences.to_csv(
        EXPERIMENTS_DIR
        / f"{EXPERIMENT}_metric_differences.csv",
        index=False,
        encoding="utf-8"
    )


def compare_benchmark_classes(
    all_classes
):

    print("\n=== BENCHMARK CLASS COMPARISON ===")

    comparison = pd.DataFrame(
        index=INTENT_CLASSES
    )

    for name, summary in all_classes.items():

        comparison[
            f"{name}_recall"
        ] = summary["recall"]

        comparison[
            f"{name}_f1"
        ] = summary["f1"]

        comparison[
            f"{name}_top3"
        ] = summary["top3_accuracy"]

    print(
        comparison.round(4)
        .to_string()
    )

    comparison.to_csv(
        EXPERIMENTS_DIR
        / f"{EXPERIMENT}_class_comparison.csv",
        encoding="utf-8"
    )


def compare_benchmark_predictions(
    all_predictions
):

    print("\n=== BENCHMARK PREDICTION COMPARISON ===")

    reference = all_predictions["a_lr"]

    comparison = reference[
        ["id", "text", "expected", "group"]
    ].copy()

    for name, predictions in all_predictions.items():

        for column in [
            "id",
            "model_text",
            "expected"
        ]:

            if not reference[column].equals(
                predictions[column]
            ):

                raise ValueError(
                    f"Prediction mismatch: {column}"
                )

        comparison[
            f"{name}_prediction"
        ] = predictions["predicted"]

        comparison[
            f"{name}_correct"
        ] = predictions["correct"]

        comparison[
            f"{name}_top3"
        ] = predictions["top3_correct"]

    for classifier in ["lr", "svm"]:

        old = comparison[
            f"a_{classifier}_correct"
        ]

        for strategy in ["b", "c"]:

            name = (
                f"{strategy}_{classifier}"
            )

            new = comparison[
                f"{name}_correct"
            ]

            status_column = (
                f"{name}_status"
            )

            comparison[status_column] = (
                np.select(
                    [
                        old & new,
                        ~old & new,
                        old & ~new,
                    ],
                    [
                        "BOTH_CORRECT",
                        "IMPROVED",
                        "REGRESSED",
                    ],
                    default="BOTH_INCORRECT"
                )
            )

            print(f"\n{name.upper()}")

            print(
                comparison[status_column]
                .value_counts()
                .to_string()
            )

            for status in [
                "IMPROVED",
                "REGRESSED"
            ]:

                print(f"\n--- {status} ---")

                subset = comparison.loc[
                    comparison[
                        status_column
                    ].eq(status)
                ]

                print(
                    subset[
                        [
                            "id",
                            "text",
                            "expected",
                            f"a_{classifier}_prediction",
                            f"{name}_prediction",
                        ]
                    ].to_string(index=False)
                )

    comparison.to_csv(
        EXPERIMENTS_DIR
        / f"{EXPERIMENT}_prediction_comparison.csv",
        index=False,
        encoding="utf-8"
    )


def main():

    print(
        "=== E09.07.5: "
        "TRAINING FREQUENCY WEIGHTING ==="
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    EXPERIMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Protect previous models and datasets

    protected_paths = [
        MODELS_DIR
        / "E09_05a_domain_adapted.joblib",

        MODELS_DIR
        / "E09_07_02_char_logreg.joblib",

        LR_MODEL_PATH,
        SVM_MODEL_PATH,
        BENCHMARK_PATH,

        SWITCHBOARD_DIR / "train.csv",
        SWITCHBOARD_DIR / "validation.csv",

        DOMAIN_PATH,
    ]

    hashes_before = {
        path: calculate_sha256(path)
        for path in protected_paths
    }

    (
        switchboard_train,
        domain_train,
        validation,
        benchmark

    ) = load_datasets()

    validate_datasets(
        switchboard_train,
        domain_train,
        validation,
        benchmark
    )

    combined, original_weights = (
        prepare_training_data(
            switchboard_train,
            domain_train
        )
    )

    if not np.allclose(
        original_weights,
        1.0
    ):

        raise ValueError(
            "Expected uniform original weights."
        )

    # Verify source order

    expected_combined = pd.concat(
        [
            switchboard_train,
            domain_train
        ],
        ignore_index=True
    )

    columns = [
        TEXT_COLUMN,
        TARGET_COLUMN
    ]

    if not combined[columns].reset_index(
        drop=True
    ).equals(
        expected_combined[columns]
    ):

        raise ValueError(
            "Unexpected combined dataset order."
        )

    source_labels = np.array(
        ["switchboard"] * len(
            switchboard_train
        )
        + ["domain"] * len(domain_train)
    )

    frequencies = calculate_frequencies(
        combined
    )

    all_weights = {}

    all_statistics = []

    for key, (name, alpha) in STRATEGIES.items():

        print(
            f"\n=== STRATEGY {key.upper()}: "
            f"{name.upper()} ==="
        )

        print(f"Alpha: {alpha}")

        weights, statistics = (
            calculate_sample_weights(
                combined,
                frequencies,
                alpha
            )
        )

        all_weights[key] = weights

        statistics["strategy"] = key

        statistics["strategy_name"] = name

        statistics["alpha"] = alpha

        all_statistics.append(
            statistics
        )

        print(
            statistics.round(4)
            .to_string(index=False)
        )

    pd.concat(
        all_statistics,
        ignore_index=True
    ).to_csv(
        EXPERIMENTS_DIR
        / f"{EXPERIMENT}_class_weight_statistics.csv",
        index=False,
        encoding="utf-8"
    )

    analyze_weight_distribution(
        combined,
        frequencies,
        all_weights
    )

    analyze_source_weights(
        combined,
        source_labels,
        all_weights
    )

    # Load previous models

    reference_models = {
        "lr": joblib.load(
            LR_MODEL_PATH
        ),
        "svm": joblib.load(
            SVM_MODEL_PATH
        ),
    }

    frozen_features = (
        reference_models["lr"]
        .named_steps["tfidf"]
    )

    svm_features = (
        reference_models["svm"]
        .named_steps["tfidf"]
    )

    lr_vectorizers = dict(
        frozen_features.transformer_list
    )

    svm_vectorizers = dict(
        svm_features.transformer_list
    )

    if list(lr_vectorizers) != list(
        svm_vectorizers
    ):

        raise ValueError(
            "Transformer names differ."
        )

    if (
        frozen_features.transformer_weights
        != svm_features.transformer_weights
    ):

        raise ValueError(
            "Transformer weights differ."
        )

    for name, vectorizer in lr_vectorizers.items():

        reference = svm_vectorizers[name]

        if vectorizer.get_params() != (
            reference.get_params()
        ):

            raise ValueError(
                f"Vectorizer parameters differ: {name}"
            )

        if vectorizer.vocabulary_ != (
            reference.vocabulary_
        ):

            raise ValueError(
                f"Vocabulary mismatch: {name}"
            )

        if not np.allclose(
            vectorizer.idf_,
            reference.idf_,
            rtol=1e-12,
            atol=1e-12
        ):

            raise ValueError(
                f"IDF mismatch: {name}"
            )

    print(
        "\nReference feature spaces match: YES"
    )

    # Transform training data only once

    print(
        "\n=== FROZEN FEATURE TRANSFORMATION ==="
    )

    feature_hash_before = joblib.hash(
        frozen_features
    )

    start = perf_counter()

    X_train = frozen_features.transform(
        combined[TEXT_COLUMN]
    )

    transformation_time = (
        perf_counter() - start
    )

    y_train = combined[
        TARGET_COLUMN
    ].to_numpy()

    print(
        f"Feature matrix shape: "
        f"{X_train.shape}"
    )

    print(
        f"Nonzero features: "
        f"{X_train.nnz:,}"
    )

    print(
        f"Transformation time: "
        f"{transformation_time:.4f} seconds"
    )

    if X_train.shape != (
        len(combined),
        177043
    ):

        raise ValueError(
            "Unexpected feature matrix shape."
        )

    # Reuse E09.07.4 evaluation functions.
    # Redirect output to the new experiment.

    evaluation.EXPERIMENT = EXPERIMENT

    results = []

    baseline_checks = []

    benchmark_classes = {}

    benchmark_predictions = {}

    for classifier_name, reference in (
        reference_models.items()
    ):

        for strategy, weights in (
            all_weights.items()
        ):

            model_name = (
                f"{strategy}_{classifier_name}"
            )

            model, training_time = (
                train_weighted_model(
                    reference,
                    frozen_features,
                    X_train,
                    y_train,
                    weights,
                    model_name
                )
            )

            for dataset_name, dataset in [
                ("validation", validation),
                ("benchmark", benchmark)
            ]:

                if strategy == "a":

                    check = (
                        verify_uniform_baseline(
                            reference,
                            model,
                            dataset,
                            classifier_name,
                            dataset_name
                        )
                    )

                    baseline_checks.append(
                        check
                    )

                (
                    metrics,
                    predictions,
                    class_summary

                ) = evaluation.evaluate_model(
                    model,
                    model_name,
                    dataset,
                    dataset_name
                )

                metrics["training_seconds"] = (
                    training_time
                )

                metrics["strategy"] = (
                    STRATEGIES[strategy][0]
                )

                metrics["alpha"] = (
                    STRATEGIES[strategy][1]
                )

                metrics["classifier"] = (
                    classifier_name
                )

                results.append(metrics)

                if dataset_name == "benchmark":

                    benchmark_classes[
                        model_name
                    ] = class_summary

                    benchmark_predictions[
                        model_name
                    ] = predictions

    results = pd.DataFrame(results)

    results.to_csv(
        EXPERIMENTS_DIR
        / f"{EXPERIMENT}_results.csv",
        index=False,
        encoding="utf-8"
    )

    pd.DataFrame(
        baseline_checks
    ).to_csv(
        EXPERIMENTS_DIR
        / f"{EXPERIMENT}_baseline_checks.csv",
        index=False,
        encoding="utf-8"
    )

    compare_results(results)

    compare_benchmark_classes(
        benchmark_classes
    )

    compare_benchmark_predictions(
        benchmark_predictions
    )

    # Integrity verification

    print("\n=== EXPERIMENT INTEGRITY ===")

    for path, original_hash in (
        hashes_before.items()
    ):

        unchanged = (
            calculate_sha256(path)
            == original_hash
        )

        status = (
            "PASS" if unchanged else "FAIL"
        )

        print(
            f"{status} | "
            f"{path.parent.name}/{path.name}"
        )

        if not unchanged:

            raise ValueError(
                f"Protected file changed: {path}"
            )

    if joblib.hash(frozen_features) != (
        feature_hash_before
    ):

        raise ValueError(
            "Frozen feature extractor changed."
        )

    print(
        "Frozen feature extractor unchanged: YES"
    )

    print(
        "Total class weights identical: YES"
    )

    print("Existing models modified: NO")
    print("Training datasets modified: NO")
    print("Benchmark modified: NO")
    print("Feature extractor retrained: NO")
    print("New classifiers trained: 6")
    print("Probability calibration performed: NO")
    print("Independent final test used: NO")

    print("\n=== EXPERIMENT COMPLETE ===")


if __name__ == "__main__":
    main()
