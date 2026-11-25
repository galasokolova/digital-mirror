
from time import perf_counter
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from train_intent_combined_logreg import (
    calculate_sha256,
    get_vectorizers,
    analyze_features,
    load_datasets,
    validate_datasets,
    prepare_training_data,
    calculate_metrics,
    INTENT_CLASSES,
    GROUPS,
    TEXT_COLUMN,
    TARGET_COLUMN,
    RANDOM_STATE,
    MODELS_DIR,
    EXPERIMENTS_DIR,
    SWITCHBOARD_DIR,
    DOMAIN_PATH,
    BENCHMARK_PATH,
)


EXPERIMENT = "E09_07_04"

WORD_MODEL_PATH = (
    MODELS_DIR
    / "E09_05a_domain_adapted.joblib"
)

CHAR_MODEL_PATH = (
    MODELS_DIR
    / "E09_07_02_char_logreg.joblib"
)

LR_MODEL_PATH = (
    MODELS_DIR
    / "E09_07_03_combined_logreg.joblib"
)

MODEL_PATH = (
    MODELS_DIR
    / f"{EXPERIMENT}_combined_svm.joblib"
)


def create_model(reference_model):

    features = clone(
        reference_model.named_steps["tfidf"]
    )

    classifier = LinearSVC(
        C=1.0,
        penalty="l2",
        loss="squared_hinge",
        dual="auto",
        class_weight="balanced",
        max_iter=5000,
        tol=1e-4,
        random_state=RANDOM_STATE,
    )

    return Pipeline([
        ("tfidf", features),
        ("classifier", classifier),
    ])


def get_class_scores(model, texts):

    if hasattr(model, "predict_proba"):

        scores = model.predict_proba(texts)

        score_type = "probability"

    elif hasattr(model, "decision_function"):

        scores = model.decision_function(texts)

        score_type = "decision_score"

    else:

        raise TypeError(
            "Unsupported classifier."
        )

    return np.asarray(scores), score_type


def evaluate_model(
    model,
    model_name,
    dataset,
    dataset_name
):

    print("\n" + "=" * 70)
    print(f"MODEL: {model_name}")
    print(f"DATASET: {dataset_name}")
    print("=" * 70)

    expected = dataset[
        TARGET_COLUMN
    ].to_numpy()

    texts = dataset[TEXT_COLUMN]

    classifier = model.named_steps[
        "classifier"
    ]

    classes = classifier.classes_

    if set(classes) != set(INTENT_CLASSES):

        raise ValueError(
            "Unexpected classifier classes."
        )

    start = perf_counter()

    scores, score_type = get_class_scores(
        model,
        texts
    )

    inference_time = perf_counter() - start

    if scores.shape != (
        len(dataset),
        len(classes)
    ):

        raise ValueError(
            "Unexpected score matrix shape."
        )

    if not np.isfinite(scores).all():

        raise ValueError(
            "Non-finite class scores detected."
        )

    ranked_indices = np.argsort(
        -scores,
        axis=1,
        kind="stable"
    )

    predicted = classes[
        ranked_indices[:, 0]
    ]

    native_predictions = model.predict(texts)

    if not np.array_equal(
        predicted,
        native_predictions
    ):

        raise ValueError(
            "Ranked predictions differ "
            "from native predictions."
        )

    top3_labels = classes[
        ranked_indices[:, :3]
    ]

    correct = expected == predicted

    top3_correct = np.any(
        top3_labels == expected[:, None],
        axis=1
    )

    metrics = calculate_metrics(
        expected,
        predicted,
        top3_correct
    )

    metrics.update({
        "model": model_name,
        "dataset": dataset_name,
        "score_type": score_type,
        "inference_seconds": inference_time,
        "ms_per_record":
            inference_time * 1000 / len(dataset),
    })

    predictions = dataset.copy()

    predictions["model"] = model_name
    predictions["expected"] = expected
    predictions["predicted"] = predicted
    predictions["correct"] = correct
    predictions["top3_correct"] = top3_correct
    predictions["score_type"] = score_type

    predictions["model_score"] = scores[
        np.arange(len(dataset)),
        ranked_indices[:, 0]
    ]

    for rank in range(3):

        predictions[
            f"top{rank + 1}_label"
        ] = top3_labels[:, rank]

        predictions[
            f"top{rank + 1}_score"
        ] = scores[
            np.arange(len(dataset)),
            ranked_indices[:, rank]
        ]

    predictions["score_gap"] = (
        predictions["top1_score"]
        - predictions["top2_score"]
    )

    print(f"\nScore type: {score_type}")

    print(
        "Ranked predictions match "
        "native predictions: YES"
    )

    # Feature analysis

    feature_statistics = analyze_features(
        model,
        dataset
    )

    total_nonzero = np.zeros(
        len(dataset),
        dtype=int
    )

    total_vocabulary = 0

    print("\n=== FEATURE STATISTICS ===")

    for name, stats in feature_statistics.items():

        coverage = stats["coverage"]

        nonzero = stats["nonzero_features"]

        vocabulary_size = stats[
            "vocabulary_size"
        ]

        total_nonzero += nonzero
        total_vocabulary += vocabulary_size

        predictions[
            f"{name}_coverage"
        ] = coverage

        predictions[
            f"{name}_nonzero_features"
        ] = nonzero

        metrics[
            f"{name}_feature_coverage"
        ] = float(coverage.mean())

        metrics[
            f"{name}_zero_vectors"
        ] = int((nonzero == 0).sum())

        print(f"\n{name.upper()} FEATURES")

        print(
            f"Vocabulary: {vocabulary_size:,}"
        )

        print(
            f"Mean coverage: {coverage.mean():.4f}"
        )

        print(
            f"Zero vectors: {(nonzero == 0).sum()}"
        )

    zero_vectors = total_nonzero == 0

    predictions["zero_vector"] = zero_vectors

    predictions["nonzero_features"] = (
        total_nonzero
    )

    metrics["vocabulary_size"] = (
        total_vocabulary
    )

    metrics["zero_vectors"] = int(
        zero_vectors.sum()
    )

    metrics["mean_nonzero_features"] = float(
        total_nonzero.mean()
    )

    print("\n=== OVERALL METRICS ===")

    for name, value in metrics.items():

        if isinstance(value, float):

            print(f"{name}: {value:.4f}")

        else:

            print(f"{name}: {value}")

    # Classification report

    print("\n=== CLASSIFICATION REPORT ===")

    print(
        classification_report(
            expected,
            predicted,
            labels=INTENT_CLASSES,
            digits=4,
            zero_division=0
        )
    )

    report = pd.DataFrame(
        classification_report(
            expected,
            predicted,
            labels=INTENT_CLASSES,
            output_dict=True,
            zero_division=0
        )
    ).transpose()

    matrix = pd.DataFrame(
        confusion_matrix(
            expected,
            predicted,
            labels=INTENT_CLASSES
        ),
        index=INTENT_CLASSES,
        columns=INTENT_CLASSES
    )

    print("\n=== CONFUSION MATRIX ===")

    print(matrix.to_string())

    # Class summary

    class_summary = (
        predictions.groupby("expected")
        .agg(
            records=("correct", "size"),
            correct=("correct", "sum"),
            recall=("correct", "mean"),
            top3_accuracy=(
                "top3_correct",
                "mean"
            ),
            zero_vectors=(
                "zero_vector",
                "sum"
            ),
        )
        .reindex(INTENT_CLASSES)
    )

    class_summary["precision"] = (
        report.loc[
            INTENT_CLASSES,
            "precision"
        ]
    )

    class_summary["f1"] = (
        report.loc[
            INTENT_CLASSES,
            "f1-score"
        ]
    )

    print("\n=== CLASS SUMMARY ===")

    print(
        class_summary.round(4).to_string()
    )

    # Group evaluation

    group_results = []

    if "group" in dataset.columns:

        print("\n=== BENCHMARK GROUP RESULTS ===")

        for group in GROUPS:

            mask = (
                dataset["group"].to_numpy()
                == group
            )

            group_metrics = calculate_metrics(
                expected[mask],
                predicted[mask],
                top3_correct[mask]
            )

            group_metrics.update({
                "model": model_name,
                "group": group,
                "zero_vectors": int(
                    zero_vectors[mask].sum()
                ),
            })

            group_results.append(
                group_metrics
            )

        print(
            pd.DataFrame(group_results)
            .round(4)
            .to_string(index=False)
        )

    # Error analysis

    errors = predictions.loc[
        ~predictions["correct"]
    ].copy()

    error_pairs = (
        errors.groupby(
            ["expected", "predicted"]
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            "count",
            ascending=False
        )
    )

    print("\n=== ERROR ANALYSIS ===")

    print(f"Errors: {len(errors)}")

    print("\nTop 15 error pairs:")

    print(
        error_pairs.head(15)
        .to_string(index=False)
    )

    recovered = int(
        errors["top3_correct"].sum()
    )

    print(
        "\nCorrect class recovered in top-3: "
        f"{recovered} / {len(errors)}"
    )

    # STATEMENT overprediction

    statement_mask = (
        predicted == "STATEMENT"
    )

    actual_statement_mask = (
        expected == "STATEMENT"
    )

    statement_predictions = int(
        statement_mask.sum()
    )

    statement_true_positives = int(
        (
            statement_mask
            & actual_statement_mask
        ).sum()
    )

    statement_false_positives = (
        statement_predictions
        - statement_true_positives
    )

    metrics.update({
        "statement_predictions":
            statement_predictions,

        "statement_true_positives":
            statement_true_positives,

        "statement_false_positives":
            statement_false_positives,
    })

    print("\n=== STATEMENT OVERPREDICTION ===")

    print(
        f"Predicted STATEMENT: "
        f"{statement_predictions}"
    )

    print(
        f"Actual STATEMENT: "
        f"{actual_statement_mask.sum()}"
    )

    print(
        f"True positives: "
        f"{statement_true_positives}"
    )

    print(
        f"False positives: "
        f"{statement_false_positives}"
    )

    if len(errors):

        statement_error_share = (
            statement_false_positives
            / len(errors)
        )

        print(
            f"Share of all errors: "
            f"{statement_error_share:.4f}"
        )

    # Save results

    prefix = (
        f"{EXPERIMENT}_"
        f"{model_name}_"
        f"{dataset_name}"
    )

    outputs = {
        "predictions": (predictions, False),
        "classification_report": (report, True),
        "confusion_matrix": (matrix, True),
        "class_summary": (class_summary, True),
        "errors": (errors, False),
        "error_pairs": (error_pairs, False),
    }

    if group_results:

        outputs["group_results"] = (
            pd.DataFrame(group_results),
            False
        )

    for name, (frame, save_index) in outputs.items():

        frame.to_csv(
            EXPERIMENTS_DIR
            / f"{prefix}_{name}.csv",
            index=save_index,
            encoding="utf-8"
        )

    return metrics, predictions, class_summary


def compare_predictions(all_predictions):

    print("\n=== BENCHMARK PREDICTION COMPARISON ===")

    baseline = all_predictions["combined_lr"]

    comparison = baseline[
        [
            "id",
            "text",
            "expected",
            "group",
        ]
    ].copy()

    for name, predictions in all_predictions.items():

        for column in [
            "id",
            "model_text",
            "expected",
        ]:

            if not baseline[column].equals(
                predictions[column]
            ):

                raise ValueError(
                    f"Prediction mismatch: {column}"
                )

        comparison[f"{name}_prediction"] = (
            predictions["predicted"]
        )

        comparison[f"{name}_correct"] = (
            predictions["correct"]
        )

        comparison[f"{name}_top3_correct"] = (
            predictions["top3_correct"]
        )

    old = comparison["combined_lr_correct"]

    new = comparison["combined_svm_correct"]

    comparison["status"] = np.select(
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

    print(
        comparison["status"]
        .value_counts()
        .to_string()
    )

    previous_correct = (
        comparison["word_correct"]
        | comparison["char_correct"]
        | comparison["combined_lr_correct"]
    )

    comparison["oracle_correct"] = (
        previous_correct | new
    )

    comparison["svm_only_correct"] = (
        ~previous_correct & new
    )

    print(
        "\nPrevious oracle accuracy: "
        f"{previous_correct.mean():.4f}"
    )

    print(
        "New oracle accuracy: "
        f"{comparison['oracle_correct'].mean():.4f}"
    )

    print(
        "Errors corrected only by SVM: "
        f"{comparison['svm_only_correct'].sum()}"
    )

    columns = [
        "id",
        "text",
        "expected",
        "word_prediction",
        "char_prediction",
        "combined_lr_prediction",
        "combined_svm_prediction",
    ]

    for status in [
        "IMPROVED",
        "REGRESSED"
    ]:

        print(f"\n--- {status} ---")

        subset = comparison.loc[
            comparison["status"].eq(status)
        ]

        print(
            subset[columns]
            .to_string(index=False)
        )

    print("\n=== SVM UNIQUE IMPROVEMENTS ===")

    unique_improvements = comparison.loc[
        comparison["svm_only_correct"]
    ]

    print(
        unique_improvements[columns]
        .to_string(index=False)
    )

    comparison.to_csv(
        EXPERIMENTS_DIR
        / f"{EXPERIMENT}_prediction_comparison.csv",
        index=False,
        encoding="utf-8"
    )


def compare_classes(all_classes):

    print("\n=== BENCHMARK CLASS COMPARISON ===")

    comparison = pd.DataFrame(
        index=INTENT_CLASSES
    )

    for name, summary in all_classes.items():

        comparison[f"{name}_recall"] = (
            summary["recall"]
        )

        comparison[f"{name}_f1"] = (
            summary["f1"]
        )

        comparison[f"{name}_top3"] = (
            summary["top3_accuracy"]
        )

    comparison["recall_difference_pp"] = (
        comparison["combined_svm_recall"]
        - comparison["combined_lr_recall"]
    ) * 100

    comparison["f1_difference_pp"] = (
        comparison["combined_svm_f1"]
        - comparison["combined_lr_f1"]
    ) * 100

    print(
        comparison.round(4).to_string()
    )

    comparison.to_csv(
        EXPERIMENTS_DIR
        / f"{EXPERIMENT}_class_comparison.csv",
        encoding="utf-8"
    )


def compare_metrics(results):

    print("\n=== LR VS SVM METRIC DIFFERENCES ===")

    rows = []

    metric_names = [
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
            results["dataset"].eq(dataset_name)
        ].set_index("model")

        for metric in metric_names:

            lr = subset.loc[
                "combined_lr", metric
            ]

            svm = subset.loc[
                "combined_svm", metric
            ]

            rows.append({
                "dataset": dataset_name,
                "metric": metric,
                "combined_lr": lr,
                "combined_svm": svm,
                "difference_pp":
                    (svm - lr) * 100,
            })

    differences = pd.DataFrame(rows)

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


def main():

    print(
        "=== E09.07.4: WORD + CHARACTER TF-IDF "
        "+ LINEAR SVM ==="
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    EXPERIMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    protected_paths = [
        WORD_MODEL_PATH,
        CHAR_MODEL_PATH,
        LR_MODEL_PATH,
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

    combined, sample_weights = (
        prepare_training_data(
            switchboard_train,
            domain_train
        )
    )

    print("\n=== EXISTING MODELS LOADING ===")

    word_model = joblib.load(
        WORD_MODEL_PATH
    )

    char_model = joblib.load(
        CHAR_MODEL_PATH
    )

    lr_model = joblib.load(
        LR_MODEL_PATH
    )

    model = create_model(lr_model)

    reference_features = (
        lr_model.named_steps["tfidf"]
    )

    new_features = model.named_steps["tfidf"]

    # Compare transformer parameters

    reference_vectorizers = get_vectorizers(
        lr_model
    )

    new_vectorizers = get_vectorizers(
        model
    )

    if list(reference_vectorizers) != list(
        new_vectorizers
    ):

        raise ValueError(
            "Transformer names or order differ."
        )

    for name, vectorizer in new_vectorizers.items():

        if vectorizer.get_params() != (
            reference_vectorizers[name].get_params()
        ):

            raise ValueError(
                f"Vectorizer parameters differ: {name}"
            )

        if hasattr(vectorizer, "vocabulary_"):

            raise ValueError(
                "Cloned vectorizer is already fitted."
            )

    if (
        reference_features.transformer_weights
        != new_features.transformer_weights
    ):

        raise ValueError(
            "Transformer weights differ."
        )

    print(
        "Vectorizer parameters match "
        "Combined LR: YES"
    )

    print(
        "Transformer weights match "
        "Combined LR: YES"
    )

    print("Cloned vectorizers are unfitted: YES")

    print("\n=== NEW MODEL CONFIGURATION ===")

    print(model)

    # Train the new classifier

    print("\n=== MODEL TRAINING ===")

    start = perf_counter()

    with warnings.catch_warnings(record=True) as caught:

        warnings.simplefilter(
            "always",
            ConvergenceWarning
        )

        model.fit(
            combined[TEXT_COLUMN],
            combined[TARGET_COLUMN],
            classifier__sample_weight=sample_weights
        )

    training_time = perf_counter() - start

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
            "SVM did not converge. "
            "Model was not saved."
        )

    # Verify trained feature space

    print("\n=== TRAINED FEATURE SPACE ===")

    vectorizers = get_vectorizers(model)

    total_features = 0

    for name, vectorizer in vectorizers.items():

        reference = reference_vectorizers[name]

        vocabulary_size = len(
            vectorizer.vocabulary_
        )

        total_features += vocabulary_size

        same_vocabulary = (
            vectorizer.vocabulary_
            == reference.vocabulary_
        )

        same_idf = np.allclose(
            vectorizer.idf_,
            reference.idf_,
            rtol=1e-12,
            atol=1e-12
        )

        print(
            f"{name}: {vocabulary_size:,}"
        )

        print(
            f"Vocabulary matches LR: "
            f"{same_vocabulary}"
        )

        print(
            f"IDF weights match LR: "
            f"{same_idf}"
        )

        if not same_vocabulary or not same_idf:

            raise ValueError(
                f"Feature space differs: {name}"
            )

    classifier = model.named_steps[
        "classifier"
    ]

    print(
        f"Total features: {total_features:,}"
    )

    print(
        f"Classifier iterations: "
        f"{classifier.n_iter_}"
    )

    print(
        f"Coefficient matrix shape: "
        f"{classifier.coef_.shape}"
    )

    if classifier.coef_.shape != (
        len(INTENT_CLASSES),
        total_features
    ):

        raise ValueError(
            "Unexpected coefficient matrix shape."
        )

    print(
        "Trained feature space matches "
        "Combined LR: YES"
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(f"\nModel saved: {MODEL_PATH}")

    # Evaluate all four models

    models = {
        "word": word_model,
        "char": char_model,
        "combined_lr": lr_model,
        "combined_svm": model,
    }

    results = []

    for dataset_name, dataset in [
        ("validation", validation),
        ("benchmark", benchmark),
    ]:

        all_predictions = {}

        all_classes = {}

        for name, current_model in models.items():

            (
                metrics,
                predictions,
                class_summary

            ) = evaluate_model(
                current_model,
                name,
                dataset,
                dataset_name
            )

            metrics["training_seconds"] = (
                training_time
                if name == "combined_svm"
                else np.nan
            )

            results.append(metrics)

            all_predictions[name] = predictions

            all_classes[name] = class_summary

        if dataset_name == "benchmark":

            compare_predictions(
                all_predictions
            )

            compare_classes(
                all_classes
            )

    results = pd.DataFrame(results)

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
        "zero_vectors",
        "vocabulary_size",
        "ms_per_record",
    ]

    print(
        results[columns]
        .round(4)
        .to_string(index=False)
    )

    results.to_csv(
        EXPERIMENTS_DIR
        / f"{EXPERIMENT}_results.csv",
        index=False,
        encoding="utf-8"
    )

    compare_metrics(results)

    # Integrity verification

    print("\n=== EXPERIMENT INTEGRITY ===")

    integrity_checks = []

    for path, original_hash in hashes_before.items():

        unchanged = (
            calculate_sha256(path)
            == original_hash
        )

        integrity_checks.append(unchanged)

        status = (
            "PASS" if unchanged else "FAIL"
        )

        print(
            f"{status} | "
            f"{path.parent.name}/{path.name}"
        )

    if not all(integrity_checks):

        raise ValueError(
            "Protected files have changed."
        )

    print("\nExisting models modified: NO")
    print("Training datasets modified: NO")
    print("Benchmark modified: NO")
    print("New SVM model trained: YES")
    print("Probability calibration performed: NO")
    print("Hyperparameter tuning performed: NO")
    print("Independent final test used: NO")

    print(
        f"New model SHA256: "
        f"{calculate_sha256(MODEL_PATH)}"
    )

    print("\n=== EXPERIMENT COMPLETE ===")


if __name__ == "__main__":
    main()
