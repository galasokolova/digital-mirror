
from pathlib import Path
from time import perf_counter
import hashlib

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion

from train_intent_char_logreg import (
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

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)


EXPERIMENT = "E09_07_03"

BASELINE_PATH = (
    MODELS_DIR
    / "E09_05a_domain_adapted.joblib"
)

CHAR_MODEL_PATH = (
    MODELS_DIR
    / "E09_07_02_char_logreg.joblib"
)

MODEL_PATH = (
    MODELS_DIR
    / f"{EXPERIMENT}_combined_logreg.joblib"
)


def calculate_sha256(path):

    sha256 = hashlib.sha256()

    with path.open("rb") as file:

        for chunk in iter(
            lambda: file.read(65536),
            b""
        ):

            sha256.update(chunk)

    return sha256.hexdigest()


def create_model():

    word_vectorizer = TfidfVectorizer(

        analyzer="word",

        ngram_range=(1, 2),

        min_df=2,

        max_features=100000,

        sublinear_tf=True,

        token_pattern=r"(?u)\b\w[\w']*\b",
    )

    char_vectorizer = TfidfVectorizer(

        analyzer="char",

        ngram_range=(3, 5),

        min_df=2,

        max_features=100000,

        sublinear_tf=True,
    )

    combined_features = FeatureUnion(

        transformer_list=[

            ("word", word_vectorizer),

            ("char", char_vectorizer),
        ],

        transformer_weights={

            "word": 1.0,

            "char": 1.0,
        },

        n_jobs=1,
    )

    classifier = LogisticRegression(

        C=1.0,

        max_iter=1000,

        solver="lbfgs",

        class_weight="balanced",

        random_state=RANDOM_STATE,
    )

    return Pipeline([

        ("tfidf", combined_features),

        ("classifier", classifier),
    ])


def get_vectorizers(model):

    transformer = model.named_steps["tfidf"]

    if isinstance(transformer, FeatureUnion):

        return dict(
            transformer.transformer_list
        )

    return {
        transformer.analyzer: transformer
    }


def analyze_features(model, dataset):

    vectorizers = get_vectorizers(model)

    statistics = {}

    for name, vectorizer in vectorizers.items():

        analyzer = vectorizer.build_analyzer()

        vocabulary = vectorizer.vocabulary_

        coverage_values = []

        for text in dataset[TEXT_COLUMN]:

            features = analyzer(text)

            known = sum(
                feature in vocabulary
                for feature in features
            )

            coverage = (

                known / len(features)

                if features

                else 0.0
            )

            coverage_values.append(coverage)

        vectors = vectorizer.transform(
            dataset[TEXT_COLUMN]
        )

        nonzero = np.asarray(
            vectors.getnnz(axis=1)
        ).ravel()

        statistics[name] = {

            "vocabulary_size": len(vocabulary),

            "coverage": np.asarray(
                coverage_values
            ),

            "nonzero_features": nonzero,
        }

    return statistics


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

    classifier = model.named_steps[
        "classifier"
    ]

    classes = classifier.classes_

    if set(classes) != set(INTENT_CLASSES):

        raise ValueError(
            "Unexpected model classes."
        )

    start = perf_counter()

    probabilities = model.predict_proba(
        dataset[TEXT_COLUMN]
    )

    inference_time = perf_counter() - start

    ranked_indices = np.argsort(

        -probabilities,

        axis=1,

        kind="stable"
    )

    predicted = classes[
        ranked_indices[:, 0]
    ]

    top3_labels = classes[
        ranked_indices[:, :3]
    ]

    top3_correct = np.any(

        top3_labels == expected[:, None],

        axis=1
    )

    correct = expected == predicted

    metrics = calculate_metrics(

        expected,

        predicted,

        top3_correct
    )

    metrics.update({

        "model": model_name,

        "dataset": dataset_name,

        "inference_seconds":
            inference_time,

        "ms_per_record":
            inference_time * 1000 / len(dataset),
    })

    predictions = dataset.copy()

    predictions["model"] = model_name

    predictions["expected"] = expected

    predictions["predicted"] = predicted

    predictions["correct"] = correct

    predictions["model_score"] = (
        probabilities.max(axis=1)
    )

    predictions["top3_correct"] = top3_correct

    for rank in range(3):

        predictions[
            f"top{rank + 1}_label"
        ] = top3_labels[:, rank]

        predictions[
            f"top{rank + 1}_score"
        ] = probabilities[

            np.arange(len(dataset)),

            ranked_indices[:, rank]
        ]

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

        vocabulary_size = stats["vocabulary_size"]

        total_nonzero += nonzero

        total_vocabulary += vocabulary_size

        predictions[
            f"{name}_coverage"
        ] = coverage

        predictions[
            f"{name}_nonzero_features"
        ] = nonzero

        predictions[
            f"{name}_zero_vector"
        ] = nonzero == 0

        metrics[
            f"{name}_feature_coverage"
        ] = float(coverage.mean())

        metrics[
            f"{name}_zero_vectors"
        ] = int((nonzero == 0).sum())

        metrics[
            f"{name}_vocabulary_size"
        ] = vocabulary_size

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

    predictions["nonzero_features"] = (
        total_nonzero
    )

    predictions["zero_vector"] = zero_vectors

    metrics["vocabulary_size"] = total_vocabulary

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
        report.loc[INTENT_CLASSES, "precision"]
    )

    class_summary["f1"] = (
        report.loc[INTENT_CLASSES, "f1-score"]
    )

    print("\n=== CLASS SUMMARY ===")

    print(
        class_summary.round(4).to_string()
    )

    print("\n=== ZERO FEATURE VECTORS ===")

    zero_subset = predictions.loc[
        predictions["zero_vector"]
    ]

    print(f"Zero vectors: {len(zero_subset)}")

    if not zero_subset.empty:

        print(

            zero_subset[
                [
                    TEXT_COLUMN,
                    "expected",
                    "predicted",
                ]
            ].head(20).to_string(index=False)
        )

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

                "zero_vectors":
                    int(zero_vectors[mask].sum()),
            })

            group_results.append(
                group_metrics
            )

        print(

            pd.DataFrame(group_results)

            .round(4)

            .to_string(index=False)
        )

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

    prefix = (

        f"{EXPERIMENT}_"

        f"{model_name}_"

        f"{dataset_name}"
    )

    outputs = {

        "predictions":
            (predictions, False),

        "classification_report":
            (report, True),

        "confusion_matrix":
            (matrix, True),

        "class_summary":
            (class_summary, True),

        "errors":
            (errors, False),

        "error_pairs":
            (error_pairs, False),
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


def compare_benchmark_predictions(
    all_predictions
):

    print("\n=== BENCHMARK PREDICTION COMPARISON ===")

    baseline = all_predictions["word"]

    comparison = baseline[

        [
            "id",
            "text",
            "expected",
            "group",
        ]

    ].copy()

    for name, predictions in all_predictions.items():

        if not baseline["id"].equals(
            predictions["id"]
        ):

            raise ValueError(
                "Prediction IDs do not match."
            )

        if not baseline["expected"].equals(
            predictions["expected"]
        ):

            raise ValueError(
                "Expected labels do not match."
            )

        comparison[f"{name}_prediction"] = (
            predictions["predicted"]
        )

        comparison[f"{name}_correct"] = (
            predictions["correct"]
        )

    old = comparison["word_correct"]

    new = comparison["combined_correct"]

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

    comparison["oracle_correct"] = (

        comparison["word_correct"]

        | comparison["char_correct"]

        | comparison["combined_correct"]
    )

    comparison["combined_only_correct"] = (

        ~comparison["word_correct"]

        & ~comparison["char_correct"]

        & comparison["combined_correct"]
    )

    print(

        "\nOracle accuracy across three models: "

        f"{comparison['oracle_correct'].mean():.4f}"
    )

    print(

        "Errors corrected only by combined model: "

        f"{comparison['combined_only_correct'].sum()}"
    )

    for status in [
        "IMPROVED",
        "REGRESSED"
    ]:

        print(f"\n--- {status} ---")

        subset = comparison.loc[
            comparison["status"].eq(status)
        ]

        print(

            subset[
                [
                    "id",
                    "text",
                    "expected",
                    "word_prediction",
                    "char_prediction",
                    "combined_prediction",
                ]
            ].to_string(index=False)
        )

    print("\n=== COMBINED MODEL UNIQUE IMPROVEMENTS ===")

    unique_improvements = comparison.loc[
        comparison["combined_only_correct"]
    ]

    print(

        unique_improvements[
            [
                "id",
                "text",
                "expected",
                "word_prediction",
                "char_prediction",
                "combined_prediction",
            ]
        ].to_string(index=False)
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

        comparison["combined_recall"]

        - comparison["word_recall"]

    ) * 100

    comparison["f1_difference_pp"] = (

        comparison["combined_f1"]

        - comparison["word_f1"]

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

    print("\n=== METRIC DIFFERENCES ===")

    difference_rows = []

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
            results["dataset"].eq(dataset_name)
        ].set_index("model")

        for metric in metrics:

            word = subset.loc[
                "word", metric
            ]

            char = subset.loc[
                "char", metric
            ]

            combined = subset.loc[
                "combined", metric
            ]

            difference_rows.append({

                "dataset": dataset_name,

                "metric": metric,

                "word": word,

                "char": char,

                "combined": combined,

                "combined_vs_word_pp":
                    (combined - word) * 100,

                "combined_vs_char_pp":
                    (combined - char) * 100,
            })

    differences = pd.DataFrame(
        difference_rows
    )

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

        "=== E09.07.3: WORD + CHARACTER TF-IDF "

        "+ LOGISTIC REGRESSION ==="
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

        BASELINE_PATH,

        CHAR_MODEL_PATH,

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
        BASELINE_PATH
    )

    char_model = joblib.load(
        CHAR_MODEL_PATH
    )

    model = create_model()

    classifier_params = (

        model.named_steps["classifier"]

        .get_params()
    )

    for name, existing_model in [

        ("word", word_model),

        ("char", char_model),
    ]:

        if (

            existing_model.named_steps["classifier"]

            .get_params()

            != classifier_params
        ):

            raise ValueError(

                f"Classifier parameters differ: {name}"
            )

    print(
        "Classifier parameters match both models: YES"
    )

    new_vectorizers = get_vectorizers(model)

    for name, existing_model in [

        ("word", word_model),

        ("char", char_model),
    ]:

        existing_vectorizer = (
            existing_model.named_steps["tfidf"]
        )

        if (

            existing_vectorizer.get_params()

            != new_vectorizers[name].get_params()
        ):

            raise ValueError(

                f"Vectorizer parameters differ: {name}"
            )

    print(
        "Vectorizer parameters match existing models: YES"
    )

    print("\n=== NEW MODEL CONFIGURATION ===")

    print(model)

    print("\n=== MODEL TRAINING ===")

    start = perf_counter()

    model.fit(

        combined[TEXT_COLUMN],

        combined[TARGET_COLUMN],

        classifier__sample_weight=sample_weights
    )

    training_time = perf_counter() - start

    print(
        f"Training time: {training_time:.4f} seconds"
    )

    vectorizers = get_vectorizers(model)

    total_features = 0

    print("\n=== TRAINED FEATURE SPACE ===")

    for name, vectorizer in vectorizers.items():

        vocabulary_size = len(
            vectorizer.vocabulary_
        )

        total_features += vocabulary_size

        print(
            f"{name}: {vocabulary_size:,}"
        )

    print(
        f"Total features: {total_features:,}"
    )

    classifier = model.named_steps[
        "classifier"
    ]

    print(
        f"Classifier iterations: {classifier.n_iter_}"
    )

    print(
        f"Coefficient matrix shape: {classifier.coef_.shape}"
    )

    if classifier.coef_.shape[1] != total_features:

        raise ValueError(
            "Unexpected coefficient matrix shape."
        )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(f"\nModel saved: {MODEL_PATH}")

    models = {

        "word": word_model,

        "char": char_model,

        "combined": model,
    }

    results = []

    for dataset_name, dataset in [

        ("validation", validation),

        ("benchmark", benchmark),
    ]:

        all_predictions = {}

        all_classes = {}

        for name, current_model in models.items():

            metrics, predictions, class_summary = (

                evaluate_model(

                    current_model,

                    name,

                    dataset,

                    dataset_name
                )
            )

            metrics["training_seconds"] = (

                training_time

                if name == "combined"

                else np.nan
            )

            results.append(metrics)

            all_predictions[name] = predictions

            all_classes[name] = class_summary

        if dataset_name == "benchmark":

            compare_benchmark_predictions(
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

    print("\n=== EXPERIMENT INTEGRITY ===")

    integrity_checks = []

    for path, original_hash in hashes_before.items():

        unchanged = (

            calculate_sha256(path)

            == original_hash
        )

        integrity_checks.append(unchanged)

        status = "PASS" if unchanged else "FAIL"

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

    print("New model trained: YES")

    print("Hyperparameter tuning performed: NO")

    print("Independent final test used: NO")

    print(

        f"New model SHA256: "

        f"{calculate_sha256(MODEL_PATH)}"
    )

    print("\n=== EXPERIMENT COMPLETE ===")


if __name__ == "__main__":
    main()
