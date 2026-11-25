
from pathlib import Path
from time import perf_counter
import hashlib
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


ML_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ML_ROOT / "src"))

from taxonomy.intent_mapping import INTENT_CLASSES


EXPERIMENT = "E09_07_02"

MODELS_DIR = ML_ROOT / "models"

EXPERIMENTS_DIR = ML_ROOT / "experiments"

PROCESSED_DIR = ML_ROOT / "data" / "processed"

SWITCHBOARD_DIR = PROCESSED_DIR / "intent_v1"

DOMAIN_PATH = (
    PROCESSED_DIR
    / "digital_mirror_v1"
    / "train_eligible.csv"
)

BENCHMARK_PATH = (
    ML_ROOT
    / "data"
    / "evaluation"
    / "digital_mirror_final"
    / "test_processed.csv"
)

BASELINE_PATH = (
    MODELS_DIR
    / "E09_05a_domain_adapted.joblib"
)

MODEL_PATH = (
    MODELS_DIR
    / f"{EXPERIMENT}_char_logreg.joblib"
)

TEXT_COLUMN = "model_text"

TARGET_COLUMN = "intent"

RANDOM_STATE = 42

GROUPS = [
    "natural",
    "short",
    "installation",
    "complex",
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


def load_datasets():

    print("\n=== DATASET LOADING ===")

    switchboard_train = load_dataset(
        SWITCHBOARD_DIR / "train.csv"
    )

    validation = load_dataset(
        SWITCHBOARD_DIR / "validation.csv"
    )

    domain_train = load_dataset(
        DOMAIN_PATH
    )

    benchmark = load_dataset(
        BENCHMARK_PATH
    )

    print(
        f"Switchboard training: "
        f"{len(switchboard_train):,}"
    )

    print(
        f"Domain training: "
        f"{len(domain_train):,}"
    )

    print(
        f"Switchboard validation: "
        f"{len(validation):,}"
    )

    print(
        f"Digital Mirror benchmark: "
        f"{len(benchmark):,}"
    )

    return (
        switchboard_train,
        domain_train,
        validation,
        benchmark
    )


def validate_datasets(
    switchboard_train,
    domain_train,
    validation,
    benchmark
):

    print("\n=== DATASET VALIDATION ===")

    switchboard_texts = set(
        switchboard_train[TEXT_COLUMN]
    )

    domain_texts = set(
        domain_train[TEXT_COLUMN]
    )

    validation_texts = set(
        validation[TEXT_COLUMN]
    )

    benchmark_texts = set(
        benchmark[TEXT_COLUMN]
    )

    datasets = [
        switchboard_train,
        domain_train,
        validation,
        benchmark,
    ]

    checks = {

        "Expected Switchboard training size":
            len(switchboard_train) == 135827,

        "Expected domain training size":
            len(domain_train) == 579,

        "Expected validation size":
            len(validation) == 16913,

        "Expected benchmark size":
            len(benchmark) == 240,

        "All intent classes present":
            all(
                set(dataset[TARGET_COLUMN])
                == set(INTENT_CLASSES)
                for dataset in datasets
            ),

        "No empty model texts":
            all(
                dataset[TEXT_COLUMN]
                .str.strip()
                .ne("")
                .all()
                for dataset in datasets
            ),

        "Unique benchmark texts":
            benchmark[TEXT_COLUMN].is_unique,

        "Balanced benchmark classes":
            benchmark[TARGET_COLUMN]
            .value_counts()
            .eq(20)
            .all(),

        "Valid benchmark groups":
            set(benchmark["group"])
            == set(GROUPS),

        "Balanced benchmark groups":
            benchmark["group"]
            .value_counts()
            .eq(60)
            .all(),

        "Balanced intent-group distribution":
            pd.crosstab(
                benchmark[TARGET_COLUMN],
                benchmark["group"]
            ).eq(5).all().all(),

        "No Switchboard conversation overlap":
            not (
                set(switchboard_train["conversation_no"])
                & set(validation["conversation_no"])
            ),

        "No domain-validation text overlap":
            not (
                domain_texts
                & validation_texts
            ),

        "No Switchboard-benchmark text overlap":
            not (
                switchboard_texts
                & benchmark_texts
            ),

        "No domain-benchmark text overlap":
            not (
                domain_texts
                & benchmark_texts
            ),

        "No validation-benchmark text overlap":
            not (
                validation_texts
                & benchmark_texts
            ),
    }

    for name, passed in checks.items():

        status = "PASS" if passed else "FAIL"

        print(f"{status} | {name}")

    print(
        f"\nChecks passed: "
        f"{sum(checks.values())} / {len(checks)}"
    )

    if not all(checks.values()):

        raise ValueError(
            "Dataset validation failed."
        )


def prepare_training_data(
    switchboard_train,
    domain_train
):

    print("\n=== TRAINING DATA PREPARATION ===")

    combined = pd.concat(

        [
            switchboard_train,
            domain_train
        ],

        ignore_index=True
    )

    sample_weights = np.ones(
        len(combined),
        dtype=float
    )

    print(
        f"Combined training records: "
        f"{len(combined):,}"
    )

    print("\nTraining class distribution:")

    distribution = pd.DataFrame({

        "switchboard":
            switchboard_train[TARGET_COLUMN]
            .value_counts(),

        "domain":
            domain_train[TARGET_COLUMN]
            .value_counts(),

        "combined":
            combined[TARGET_COLUMN]
            .value_counts(),
    }).reindex(INTENT_CLASSES)

    print(distribution.to_string())

    print("\nSwitchboard sample weight: 1.0")

    print("Domain sample weight: 1.0")

    return combined, sample_weights


def create_model():

    return Pipeline([

        (
            "tfidf",

            TfidfVectorizer(

                analyzer="char",

                ngram_range=(3, 5),

                min_df=2,

                max_features=100000,

                sublinear_tf=True,
            )
        ),

        (
            "classifier",

            LogisticRegression(

                C=1.0,

                max_iter=1000,

                solver="lbfgs",

                class_weight="balanced",

                random_state=RANDOM_STATE,
            )
        ),
    ])


def calculate_metrics(
    expected,
    predicted,
    top3_correct
):

    return {

        "records": len(expected),

        "correct": int(
            np.sum(expected == predicted)
        ),

        "accuracy": accuracy_score(
            expected,
            predicted
        ),

        "macro_precision": precision_score(
            expected,
            predicted,
            labels=INTENT_CLASSES,
            average="macro",
            zero_division=0
        ),

        "macro_recall": recall_score(
            expected,
            predicted,
            labels=INTENT_CLASSES,
            average="macro",
            zero_division=0
        ),

        "macro_f1": f1_score(
            expected,
            predicted,
            labels=INTENT_CLASSES,
            average="macro",
            zero_division=0
        ),

        "weighted_f1": f1_score(
            expected,
            predicted,
            labels=INTENT_CLASSES,
            average="weighted",
            zero_division=0
        ),

        "top3_accuracy": float(
            np.mean(top3_correct)
        ),
    }


def calculate_coverage(
    model,
    dataset
):

    vectorizer = model.named_steps["tfidf"]

    analyzer = vectorizer.build_analyzer()

    vocabulary = vectorizer.vocabulary_

    vectors = vectorizer.transform(
        dataset[TEXT_COLUMN]
    )

    nonzero_features = np.asarray(
        vectors.getnnz(axis=1)
    ).ravel()

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

    return (
        np.asarray(coverage_values),
        nonzero_features
    )


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

    expected = dataset[TARGET_COLUMN].to_numpy()

    classifier = model.named_steps["classifier"]

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

    coverage, nonzero_features = (
        calculate_coverage(
            model,
            dataset
        )
    )

    zero_vectors = nonzero_features == 0

    metrics = calculate_metrics(

        expected,

        predicted,

        top3_correct
    )

    metrics.update({

        "model": model_name,

        "dataset": dataset_name,

        "vocabulary_size":
            len(
                model.named_steps["tfidf"]
                .vocabulary_
            ),

        "mean_feature_coverage":
            float(coverage.mean()),

        "zero_vectors":
            int(zero_vectors.sum()),

        "inference_seconds":
            inference_time,

        "ms_per_record":
            inference_time * 1000 / len(dataset),
    })

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

    predictions = dataset.copy()

    predictions["model"] = model_name

    predictions["expected"] = expected

    predictions["predicted"] = predicted

    predictions["correct"] = correct

    predictions["model_score"] = (
        probabilities.max(axis=1)
    )

    predictions["top3_correct"] = top3_correct

    predictions["feature_coverage"] = coverage

    predictions["nonzero_features"] = (
        nonzero_features
    )

    predictions["zero_vector"] = zero_vectors

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

    print("\n=== CLASS SUMMARY ===")

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

            feature_coverage=(
                "feature_coverage",
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

    print(
        class_summary.round(4).to_string()
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

                "feature_coverage":
                    float(coverage[mask].mean()),
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

    print("\n=== ERROR ANALYSIS ===")

    print(f"Errors: {len(errors)}")

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

    return (
        metrics,
        predictions,
        class_summary
    )


def compare_predictions(
    baseline_predictions,
    new_predictions
):

    print("\n=== BENCHMARK PREDICTION COMPARISON ===")

    if not baseline_predictions["id"].equals(
        new_predictions["id"]
    ):

        raise ValueError(
            "Prediction IDs do not match."
        )

    comparison = baseline_predictions[
        [
            "id",
            "text",
            "expected",
            "group",
        ]
    ].copy()

    comparison["baseline_prediction"] = (
        baseline_predictions["predicted"]
    )

    comparison["char_prediction"] = (
        new_predictions["predicted"]
    )

    comparison["baseline_correct"] = (
        baseline_predictions["correct"]
    )

    comparison["char_correct"] = (
        new_predictions["correct"]
    )

    old = comparison["baseline_correct"]

    new = comparison["char_correct"]

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
                    "baseline_prediction",
                    "char_prediction",
                ]
            ].to_string(index=False)
        )

    comparison.to_csv(

        EXPERIMENTS_DIR
        / f"{EXPERIMENT}_prediction_comparison.csv",

        index=False,

        encoding="utf-8"
    )


def compare_classes(
    baseline_classes,
    new_classes
):

    print("\n=== BENCHMARK CLASS COMPARISON ===")

    comparison = pd.DataFrame({

        "baseline_recall":
            baseline_classes["recall"],

        "char_recall":
            new_classes["recall"],

        "baseline_f1":
            baseline_classes["f1"],

        "char_f1":
            new_classes["f1"],

        "baseline_top3":
            baseline_classes["top3_accuracy"],

        "char_top3":
            new_classes["top3_accuracy"],
    })

    comparison["recall_difference_pp"] = (

        comparison["char_recall"]

        - comparison["baseline_recall"]

    ) * 100

    comparison["f1_difference_pp"] = (

        comparison["char_f1"]

        - comparison["baseline_f1"]

    ) * 100

    print(
        comparison.round(4).to_string()
    )

    comparison.to_csv(

        EXPERIMENTS_DIR
        / f"{EXPERIMENT}_class_comparison.csv",

        encoding="utf-8"
    )


def main():

    print(
        "=== E09.07.2: CHARACTER TF-IDF "
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

    print("\n=== BASELINE MODEL LOADING ===")

    baseline = joblib.load(
        BASELINE_PATH
    )

    model = create_model()

    # Verify that classifier parameters
    # are identical to the baseline.

    if (
        baseline.named_steps["classifier"]
        .get_params()

        != model.named_steps["classifier"]
        .get_params()
    ):

        raise ValueError(
            "Classifier parameters differ "
            "from the baseline."
        )

    print(
        "Classifier parameters match baseline: YES"
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
        f"Training time: "
        f"{training_time:.4f} seconds"
    )

    print(
        "Vocabulary size: "
        f"{len(model.named_steps['tfidf'].vocabulary_):,}"
    )

    print(
        "Classifier iterations: "
        f"{model.named_steps['classifier'].n_iter_}"
    )

    print(
        "Coefficient matrix shape: "
        f"{model.named_steps['classifier'].coef_.shape}"
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(f"\nModel saved: {MODEL_PATH}")

    results = []

    for dataset_name, dataset in [

        ("validation", validation),

        ("benchmark", benchmark),
    ]:

        (
            baseline_metrics,
            baseline_predictions,
            baseline_classes

        ) = evaluate_model(

            baseline,

            "baseline",

            dataset,

            dataset_name
        )

        (
            new_metrics,
            new_predictions,
            new_classes

        ) = evaluate_model(

            model,

            "char_logreg",

            dataset,

            dataset_name
        )

        baseline_metrics["training_seconds"] = (
            np.nan
        )

        new_metrics["training_seconds"] = (
            training_time
        )

        results.extend([

            baseline_metrics,

            new_metrics,
        ])

        if dataset_name == "benchmark":

            compare_predictions(

                baseline_predictions,

                new_predictions
            )

            compare_classes(

                baseline_classes,

                new_classes
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

    print("\n=== METRIC DIFFERENCES ===")

    difference_rows = []

    for dataset_name in [
        "validation",
        "benchmark"
    ]:

        subset = results.loc[
            results["dataset"].eq(dataset_name)
        ].set_index("model")

        for metric in [

            "accuracy",

            "macro_precision",

            "macro_recall",

            "macro_f1",

            "weighted_f1",

            "top3_accuracy",
        ]:

            old_value = subset.loc[
                "baseline",
                metric
            ]

            new_value = subset.loc[
                "char_logreg",
                metric
            ]

            difference_rows.append({

                "dataset": dataset_name,

                "metric": metric,

                "baseline": old_value,

                "char_logreg": new_value,

                "difference_pp":
                    (new_value - old_value) * 100,
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

    print("\nBaseline model modified: NO")

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
