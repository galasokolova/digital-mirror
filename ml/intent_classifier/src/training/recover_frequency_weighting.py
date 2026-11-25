
from time import perf_counter
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning

import train_intent_frequency_weighted as frequency
import train_intent_combined_svm as evaluation


EXPERIMENT = "E09_07_05_01"

MODELS_DIR = frequency.MODELS_DIR
EXPERIMENTS_DIR = frequency.EXPERIMENTS_DIR

TEXT_COLUMN = frequency.TEXT_COLUMN
TARGET_COLUMN = frequency.TARGET_COLUMN

RECOVERY_MAX_ITER = 20000

SAVED_MODELS = {
    "a_lr": "E09_07_05a_lr.joblib",
    "b_lr": "E09_07_05b_lr.joblib",
    "c_lr": "E09_07_05c_lr.joblib",
    "a_svm": "E09_07_05a_svm.joblib",
    "b_svm": "E09_07_05b_svm.joblib",
}


def load_saved_models():

    print("\n=== SAVED MODEL VERIFICATION ===")

    models = {}

    for name, filename in SAVED_MODELS.items():

        path = MODELS_DIR / filename

        if not path.is_file():

            raise FileNotFoundError(path)

        model = joblib.load(path)

        classifier = model.named_steps[
            "classifier"
        ]

        if classifier.coef_.shape != (
            len(frequency.INTENT_CLASSES),
            177043
        ):

            raise ValueError(
                f"Invalid coefficient shape: {name}"
            )

        if set(classifier.classes_) != set(
            frequency.INTENT_CLASSES
        ):

            raise ValueError(
                f"Invalid classes: {name}"
            )

        if not np.isfinite(
            classifier.coef_
        ).all():

            raise ValueError(
                f"Invalid coefficients: {name}"
            )

        if not np.isfinite(
            classifier.intercept_
        ).all():

            raise ValueError(
                f"Invalid intercepts: {name}"
            )

        models[name] = model

        print(f"\nPASS | {name}")
        print(f"Path: {path}")

        print(
            f"SHA256: "
            f"{frequency.calculate_sha256(path)}"
        )

        print(
            f"Iterations: {classifier.n_iter_}"
        )

        print(
            f"Coefficient shape: "
            f"{classifier.coef_.shape}"
        )

        print(
            f"Parameters: "
            f"{classifier.get_params()}"
        )

    return models


def verify_feature_spaces(models, reference):

    print("\n=== FEATURE SPACE VERIFICATION ===")

    reference_features = (
        reference.named_steps["tfidf"]
    )

    reference_vectorizers = dict(
        reference_features.transformer_list
    )

    for name, model in models.items():

        features = model.named_steps[
            "tfidf"
        ]

        vectorizers = dict(
            features.transformer_list
        )

        if list(vectorizers) != list(
            reference_vectorizers
        ):

            raise ValueError(
                f"Transformer order mismatch: {name}"
            )

        if (
            features.transformer_weights
            != reference_features.transformer_weights
        ):

            raise ValueError(
                f"Transformer weight mismatch: {name}"
            )

        for key, vectorizer in vectorizers.items():

            expected = reference_vectorizers[key]

            if vectorizer.get_params() != (
                expected.get_params()
            ):

                raise ValueError(
                    f"Parameter mismatch: {name}/{key}"
                )

            if vectorizer.vocabulary_ != (
                expected.vocabulary_
            ):

                raise ValueError(
                    f"Vocabulary mismatch: {name}/{key}"
                )

            if not np.allclose(
                vectorizer.idf_,
                expected.idf_,
                rtol=1e-12,
                atol=1e-12
            ):

                raise ValueError(
                    f"IDF mismatch: {name}/{key}"
                )

        print(f"PASS | {name}")


def compare_classifier_parameters(
    reference,
    candidate,
    name
):

    print(
        f"\n=== CLASSIFIER PARAMETERS: "
        f"{name.upper()} ==="
    )

    old_params = reference.named_steps[
        "classifier"
    ].get_params()

    new_params = candidate.named_steps[
        "classifier"
    ].get_params()

    rows = []

    for parameter in sorted(
        set(old_params) | set(new_params)
    ):

        old = old_params.get(parameter)
        new = new_params.get(parameter)

        rows.append({
            "classifier": name,
            "parameter": parameter,
            "reference": str(old),
            "uniform": str(new),
            "identical": old == new,
        })

    result = pd.DataFrame(rows)

    print(
        result.to_string(index=False)
    )

    return result


def recover_strong_svm(
    reference,
    frozen_features,
    combined
):

    print("\n=== STRONG SVM RECOVERY ===")

    frequencies = (
        frequency.calculate_frequencies(
            combined
        )
    )

    sample_weights, statistics = (
        frequency.calculate_sample_weights(
            combined,
            frequencies,
            alpha=1.0
        )
    )

    statistics.to_csv(
        EXPERIMENTS_DIR
        / f"{EXPERIMENT}_strong_weights.csv",
        index=False,
        encoding="utf-8"
    )

    feature_hash_before = joblib.hash(
        frozen_features
    )

    start = perf_counter()

    X_train = frozen_features.transform(
        combined[TEXT_COLUMN]
    )

    print(
        f"Feature transformation: "
        f"{perf_counter() - start:.4f} seconds"
    )

    if X_train.shape != (
        len(combined),
        177043
    ):

        raise ValueError(
            "Unexpected feature matrix shape."
        )

    y_train = combined[
        TARGET_COLUMN
    ].to_numpy()

    classifier = clone(
        reference.named_steps["classifier"]
    )

    classifier.set_params(
        class_weight=None,
        max_iter=RECOVERY_MAX_ITER
    )

    print("\nClassifier parameters:")

    print(
        classifier.get_params()
    )

    print(
        f"\nTraining started: c_svm"
    )

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

    print(
        f"Training time: "
        f"{training_time:.4f} seconds"
    )

    print(
        f"Iterations: {classifier.n_iter_}"
    )

    for warning in caught:

        print(
            f"Training warning: "
            f"{warning.message}"
        )

    print(
        f"Convergence warnings: "
        f"{len(convergence_warnings)}"
    )

    if joblib.hash(frozen_features) != (
        feature_hash_before
    ):

        raise ValueError(
            "Frozen feature extractor changed."
        )

    if convergence_warnings:

        print(
            "FAILED | Strong SVM did not converge."
        )

        print(
            "The model will not be saved."
        )

        return None, training_time

    if classifier.coef_.shape != (
        len(frequency.INTENT_CLASSES),
        X_train.shape[1]
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

    model = Pipeline([
        ("tfidf", frozen_features),
        ("classifier", classifier),
    ])

    model_path = (
        MODELS_DIR
        / f"{EXPERIMENT}_c_svm.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        f"Model saved: {model_path}"
    )

    print(
        f"Model SHA256: "
        f"{frequency.calculate_sha256(model_path)}"
    )

    print(
        "PASS | Strong SVM recovery"
    )

    return model, training_time


def compare_available_results(results):

    print("\n=== FINAL MODEL COMPARISON ===")

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

    for dataset_name in [
        "validation",
        "benchmark"
    ]:

        subset = results.loc[
            results["dataset"].eq(
                dataset_name
            )
        ].sort_values(
            "macro_f1",
            ascending=False
        )

        print(
            f"\n=== {dataset_name.upper()} ==="
        )

        print(
            subset[columns]
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

    for dataset_name in [
        "validation",
        "benchmark"
    ]:

        subset = results.loc[
            results["dataset"].eq(
                dataset_name
            )
        ].set_index("model")

        for classifier in ["lr", "svm"]:

            baseline = subset.loc[
                f"a_{classifier}"
            ]

            for strategy in ["b", "c"]:

                name = (
                    f"{strategy}_{classifier}"
                )

                if name not in subset.index:

                    continue

                candidate = subset.loc[name]

                for metric in metrics:

                    differences.append({
                        "dataset": dataset_name,
                        "classifier": classifier,
                        "model": name,
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


def compare_available_predictions(
    all_predictions
):

    print(
        "\n=== PAIRED PREDICTION COMPARISON ==="
    )

    reference = all_predictions["a_lr"]

    comparison = reference[
        ["id", "text", "expected", "group"]
    ].copy()

    for name, predictions in (
        all_predictions.items()
    ):

        for column in [
            "id",
            "model_text",
            "expected"
        ]:

            if not reference[column].equals(
                predictions[column]
            ):

                raise ValueError(
                    f"Prediction mismatch: "
                    f"{name}/{column}"
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

        baseline = comparison[
            f"a_{classifier}_correct"
        ]

        for strategy in ["b", "c"]:

            name = (
                f"{strategy}_{classifier}"
            )

            if name not in all_predictions:

                continue

            candidate = comparison[
                f"{name}_correct"
            ]

            status = np.select(
                [
                    baseline & candidate,
                    ~baseline & candidate,
                    baseline & ~candidate,
                ],
                [
                    "BOTH_CORRECT",
                    "IMPROVED",
                    "REGRESSED",
                ],
                default="BOTH_INCORRECT"
            )

            comparison[
                f"{name}_status"
            ] = status

            print(f"\n{name.upper()}")

            print(
                pd.Series(status)
                .value_counts()
                .to_string()
            )

            for category in [
                "IMPROVED",
                "REGRESSED"
            ]:

                print(f"\n--- {category} ---")

                subset = comparison.loc[
                    comparison[
                        f"{name}_status"
                    ].eq(category)
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
        "=== E09.07.5.1: "
        "CONVERGENCE RECOVERY AND VERIFICATION ==="
    )

    EXPERIMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    protected_paths = [
        frequency.LR_MODEL_PATH,
        frequency.SVM_MODEL_PATH,
        frequency.BENCHMARK_PATH,
        frequency.DOMAIN_PATH,

        frequency.SWITCHBOARD_DIR
        / "train.csv",

        frequency.SWITCHBOARD_DIR
        / "validation.csv",

        MODELS_DIR
        / "E09_05a_domain_adapted.joblib",

        MODELS_DIR
        / "E09_07_02_char_logreg.joblib",
    ]

    protected_paths.extend([
        MODELS_DIR / filename
        for filename in SAVED_MODELS.values()
    ])

    hashes_before = {
        path: frequency.calculate_sha256(path)
        for path in protected_paths
    }

    try:

        (
            switchboard_train,
            domain_train,
            validation,
            benchmark

        ) = frequency.load_datasets()

        frequency.validate_datasets(
            switchboard_train,
            domain_train,
            validation,
            benchmark
        )

        combined, original_weights = (
            frequency.prepare_training_data(
                switchboard_train,
                domain_train
            )
        )

        if not np.allclose(
            original_weights,
            1.0
        ):

            raise ValueError(
                "Unexpected original weights."
            )

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
                "Unexpected training data order."
            )

        models = load_saved_models()

        reference_models = {
            "lr": joblib.load(
                frequency.LR_MODEL_PATH
            ),
            "svm": joblib.load(
                frequency.SVM_MODEL_PATH
            ),
        }

        verify_feature_spaces(
            {
                **models,
                "reference_svm": (
                    reference_models["svm"]
                )
            },
            reference_models["lr"]
        )

        parameter_comparisons = []

        baseline_checks = []

        for name in ["lr", "svm"]:

            reference = reference_models[name]

            candidate = models[f"a_{name}"]

            comparison = (
                compare_classifier_parameters(
                    reference,
                    candidate,
                    name
                )
            )

            parameter_comparisons.append(
                comparison
            )

            for dataset_name, dataset in [
                ("validation", validation),
                ("benchmark", benchmark)
            ]:

                check = (
                    frequency.verify_uniform_baseline(
                        reference,
                        candidate,
                        dataset,
                        name,
                        dataset_name
                    )
                )

                baseline_checks.append(check)

        pd.concat(
            parameter_comparisons,
            ignore_index=True
        ).to_csv(
            EXPERIMENTS_DIR
            / f"{EXPERIMENT}_parameter_comparison.csv",
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

        # Redirect evaluation outputs.
        # Previous experiment files are preserved.

        evaluation.EXPERIMENT = EXPERIMENT
        frequency.EXPERIMENT = EXPERIMENT

        results = []

        benchmark_classes = {}
        benchmark_predictions = {}

        def evaluate_saved_model(
            name,
            model,
            training_time=np.nan
        ):

            for dataset_name, dataset in [
                ("validation", validation),
                ("benchmark", benchmark)
            ]:

                (
                    metrics,
                    predictions,
                    class_summary

                ) = evaluation.evaluate_model(
                    model,
                    name,
                    dataset,
                    dataset_name
                )

                metrics["training_seconds"] = (
                    training_time
                )

                metrics["classifier"] = (
                    name.split("_")[1]
                )

                strategy = name.split("_")[0]

                metrics["strategy"] = (
                    frequency.STRATEGIES[strategy][0]
                )

                metrics["alpha"] = (
                    frequency.STRATEGIES[strategy][1]
                )

                results.append(metrics)

                if dataset_name == "benchmark":

                    benchmark_classes[name] = (
                        class_summary
                    )

                    benchmark_predictions[name] = (
                        predictions
                    )

            # Save progress after each model.

            pd.DataFrame(results).to_csv(
                EXPERIMENTS_DIR
                / f"{EXPERIMENT}_results.csv",
                index=False,
                encoding="utf-8"
            )

        # Evaluate the five existing models first.
        # No classifier training is performed here.

        for name, model in models.items():

            evaluate_saved_model(
                name,
                model
            )

        # Retry only the failed classifier.

        recovered_model, training_time = (
            recover_strong_svm(
                reference_models["svm"],
                reference_models["lr"]
                .named_steps["tfidf"],
                combined
            )
        )

        if recovered_model is not None:

            models["c_svm"] = recovered_model

            evaluate_saved_model(
                "c_svm",
                recovered_model,
                training_time
            )

        results = pd.DataFrame(results)

        compare_available_results(
            results
        )

        frequency.compare_benchmark_classes(
            benchmark_classes
        )

        compare_available_predictions(
            benchmark_predictions
        )

        print("\n=== RECOVERY STATUS ===")

        print("Existing models loaded: 5")
        print("Existing models retrained: 0")
        print("New training attempts: 1")

        print(
            f"Successfully evaluated models: "
            f"{len(models)}"
        )

        print(
            "Strong SVM converged: "
            + (
                "YES"
                if recovered_model is not None
                else "NO"
            )
        )

        print(
            f"Strong SVM max_iter: "
            f"{RECOVERY_MAX_ITER}"
        )

        print("Feature extractor retrained: NO")
        print("Probability calibration performed: NO")
        print("Independent final test used: NO")

        print(
            "Uniform baseline equivalence: "
            "NOT ASSUMED"
        )

        print(
            "Previous training times: "
            "NOT MEASURED DURING RECOVERY"
        )

    finally:

        # Executed even if an exception occurs.

        print("\n=== INTEGRITY VERIFICATION ===")

        failed = []

        for path, original_hash in (
            hashes_before.items()
        ):

            unchanged = (
                frequency.calculate_sha256(path)
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

                failed.append(path)

        if failed:

            raise RuntimeError(
                f"Protected files changed: {failed}"
            )

        print(
            "Protected files unchanged "
            "during recovery: YES"
        )

    print("\n=== RECOVERY COMPLETE ===")


if __name__ == "__main__":
    main()
