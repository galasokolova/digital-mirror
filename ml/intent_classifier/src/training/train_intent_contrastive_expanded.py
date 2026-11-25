
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone

import train_intent_contrastive_frozen as frozen


EXPERIMENT = "E09_07_06_05_02"

ML_ROOT = frozen.ML_ROOT
MODELS_DIR = frozen.MODELS_DIR

TEXT_COLUMN = frozen.TEXT_COLUMN
TARGET_COLUMN = frozen.TARGET_COLUMN

STRATEGIES = frozen.STRATEGIES

FROZEN_EXPERIMENT = "E09_07_06_05_01"

EXPECTED_TRAINING_RECORDS = 137605


def save_table(table, name):

    frozen.save_table(table, name)


def analyze_feature_changes(
    old_features,
    new_features
):

    print("\n=== FEATURE SPACE COMPARISON ===")

    old_transformers = dict(
        old_features.transformer_list
    )

    new_transformers = dict(
        new_features.transformer_list
    )

    if list(old_transformers) != list(new_transformers):

        raise ValueError(
            "Transformer structure mismatch."
        )

    if (
        old_features.transformer_weights
        != new_features.transformer_weights
    ):

        raise ValueError(
            "Transformer weight mismatch."
        )

    summary_rows = []
    idf_rows = []
    added_rows = []
    removed_rows = []

    for name, old_vectorizer in old_transformers.items():

        new_vectorizer = new_transformers[name]

        if (
            old_vectorizer.get_params()
            != new_vectorizer.get_params()
        ):

            raise ValueError(
                f"Vectorizer parameters changed: {name}"
            )

        old_vocab = set(
            old_vectorizer.vocabulary_
        )

        new_vocab = set(
            new_vectorizer.vocabulary_
        )

        shared = old_vocab & new_vocab
        added = new_vocab - old_vocab
        removed = old_vocab - new_vocab

        union = old_vocab | new_vocab

        jaccard = (
            len(shared) / len(union)
        )

        print(f"\nTransformer: {name}")
        print(f"Old features: {len(old_vocab)}")
        print(f"New features: {len(new_vocab)}")
        print(f"Shared features: {len(shared)}")
        print(f"Added features: {len(added)}")
        print(f"Removed features: {len(removed)}")
        print(f"Jaccard similarity: {jaccard:.6f}")

        differences = []

        for feature in sorted(shared):

            old_index = (
                old_vectorizer.vocabulary_[feature]
            )

            new_index = (
                new_vectorizer.vocabulary_[feature]
            )

            old_idf = float(
                old_vectorizer.idf_[old_index]
            )

            new_idf = float(
                new_vectorizer.idf_[new_index]
            )

            difference = new_idf - old_idf

            differences.append(difference)

            idf_rows.append({
                "transformer": name,
                "feature": feature,
                "old_idf": old_idf,
                "new_idf": new_idf,
                "idf_difference": difference,
                "absolute_difference": abs(difference),
            })

        differences = np.asarray(
            differences,
            dtype=float
        )

        for feature in sorted(added):

            index = (
                new_vectorizer.vocabulary_[feature]
            )

            added_rows.append({
                "transformer": name,
                "feature": feature,
                "idf": new_vectorizer.idf_[index],
            })

        for feature in sorted(removed):

            index = (
                old_vectorizer.vocabulary_[feature]
            )

            removed_rows.append({
                "transformer": name,
                "feature": feature,
                "idf": old_vectorizer.idf_[index],
            })

        summary_rows.append({
            "transformer": name,
            "old_features": len(old_vocab),
            "new_features": len(new_vocab),
            "shared_features": len(shared),
            "added_features": len(added),
            "removed_features": len(removed),
            "jaccard_similarity": jaccard,
            "mean_idf_difference": (
                differences.mean()
                if len(differences) else np.nan
            ),
            "mean_absolute_idf_difference": (
                np.abs(differences).mean()
                if len(differences) else np.nan
            ),
            "max_absolute_idf_difference": (
                np.abs(differences).max()
                if len(differences) else np.nan
            ),
        })

    summary = pd.DataFrame(summary_rows)

    print("\n=== FEATURE CHANGE SUMMARY ===")
    print(summary.to_string(index=False))

    save_table(
        summary,
        "feature_changes"
    )

    save_table(
        pd.DataFrame(
            added_rows,
            columns=["transformer", "feature", "idf"]
        ),
        "added_features"
    )

    save_table(
        pd.DataFrame(
            removed_rows,
            columns=["transformer", "feature", "idf"]
        ),
        "removed_features"
    )

    idf_table = pd.DataFrame(idf_rows)

    idf_table = idf_table.sort_values(
        "absolute_difference",
        ascending=False
    )

    save_table(
        idf_table,
        "idf_changes"
    )

    print("\n=== LARGEST IDF CHANGES ===")

    print(
        idf_table.head(20).to_string(index=False)
    )


def calculate_feature_coverage(
    vectorizer,
    texts
):

    analyzer = vectorizer.build_analyzer()

    vocabulary = vectorizer.vocabulary_

    total_features = 0
    known_features = 0
    zero_coverage = 0

    document_coverages = []

    for text in texts:

        features = analyzer(text)

        total = len(features)

        known = sum(
            feature in vocabulary
            for feature in features
        )

        total_features += total
        known_features += known

        coverage = (
            known / total
            if total else 0.0
        )

        document_coverages.append(
            coverage
        )

        if known == 0:

            zero_coverage += 1

    return {
        "records": len(texts),
        "total_feature_occurrences": total_features,
        "known_feature_occurrences": known_features,
        "feature_coverage": (
            known_features / total_features
            if total_features else 0.0
        ),
        "mean_document_coverage": (
            np.mean(document_coverages)
        ),
        "zero_coverage_records": zero_coverage,
    }


def analyze_coverage(
    old_features,
    new_features,
    datasets
):

    print("\n=== FEATURE COVERAGE ANALYSIS ===")

    rows = []

    for feature_space, extractor in [
        ("frozen", old_features),
        ("expanded", new_features),
    ]:

        for transformer, vectorizer in (
            extractor.transformer_list
        ):

            for dataset_name, dataset in datasets:

                statistics = (
                    calculate_feature_coverage(
                        vectorizer,
                        dataset[TEXT_COLUMN]
                    )
                )

                rows.append({
                    "feature_space": feature_space,
                    "transformer": transformer,
                    "dataset": dataset_name,
                    **statistics,
                })

    table = pd.DataFrame(rows)

    print(table.to_string(index=False))

    save_table(
        table,
        "feature_coverage"
    )

    comparison = (
        table.pivot(
            index=["transformer", "dataset"],
            columns="feature_space",
            values="feature_coverage"
        )
        .reset_index()
    )

    comparison["difference"] = (
        comparison["expanded"]
        - comparison["frozen"]
    )

    print("\n=== COVERAGE COMPARISON ===")

    print(
        comparison.to_string(index=False)
    )

    save_table(
        comparison,
        "coverage_comparison"
    )


def analyze_zero_vectors(
    matrix,
    dataset,
    name
):

    zero_mask = (
        matrix.getnnz(axis=1) == 0
    )

    print(
        f"{name} zero vectors: "
        f"{zero_mask.sum()}"
    )

    save_table(
        dataset.loc[zero_mask],
        f"{name}_zero_vectors"
    )


def main():

    print(
        "=== E09.07.6.5.2: "
        "EXPANDED FEATURE SPACE TRAINING ==="
    )

    # Redirect reused reporting functions
    # to the new experiment.

    frozen.EXPERIMENT = EXPERIMENT

    original_expected_features = (
        frozen.EXPECTED_FEATURES
    )

    frozen_model_paths = {
        strategy: (
            MODELS_DIR
            / f"{FROZEN_EXPERIMENT}{strategy}_lr.joblib"
        )
        for strategy in STRATEGIES
    }

    protected_paths = [
        frozen.SWITCHBOARD_DIR / "train.csv",
        frozen.SWITCHBOARD_DIR / "validation.csv",
        frozen.SWITCHBOARD_DIR / "test.csv",
        frozen.DOMAIN_PATH,
        frozen.CONTRASTIVE_PATH,
        frozen.BENCHMARK_PATH,
        frozen.REFERENCE_MODEL_PATH,
        *[
            MODELS_DIR / config["baseline"]
            for config in STRATEGIES.values()
        ],
        *frozen_model_paths.values(),
    ]

    original_hashes = {
        path: frozen.calculate_sha256(path)
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
        ) = frozen.load_datasets()

        combined = frozen.prepare_training_data(
            switchboard,
            domain,
            contrastive
        )

        frozen.validate_datasets(
            switchboard,
            domain,
            contrastive,
            combined,
            validation,
            benchmark
        )

        print("\n=== LOADING REFERENCE FEATURES ===")

        reference_model = joblib.load(
            frozen.REFERENCE_MODEL_PATH
        )

        old_features = (
            reference_model.named_steps["tfidf"]
        )

        old_feature_hash = joblib.hash(
            old_features
        )

        old_feature_count = len(
            old_features.get_feature_names_out()
        )

        print(
            f"Old feature count: "
            f"{old_feature_count}"
        )

        if old_feature_count != original_expected_features:

            raise ValueError(
                "Unexpected reference feature count."
            )

        # Create an unfitted feature extractor
        # with the same parameters.

        new_features = clone(
            old_features
        )

        print("\n=== REFITTING TF-IDF ===")

        started = time.perf_counter()

        training_matrix = (
            new_features.fit_transform(
                combined[TEXT_COLUMN]
            )
        )

        feature_fit_time = (
            time.perf_counter() - started
        )

        new_feature_count = len(
            new_features.get_feature_names_out()
        )

        print(
            f"TF-IDF fitting time: "
            f"{feature_fit_time:.4f} seconds"
        )

        print(
            f"New feature count: "
            f"{new_feature_count}"
        )

        print(
            f"Training matrix shape: "
            f"{training_matrix.shape}"
        )

        print(
            f"Non-zero elements: "
            f"{training_matrix.nnz}"
        )

        if training_matrix.shape != (
            EXPECTED_TRAINING_RECORDS,
            new_feature_count
        ):

            raise ValueError(
                "Unexpected training matrix shape."
            )

        if not np.isfinite(
            training_matrix.data
        ).all():

            raise ValueError(
                "Non-finite TF-IDF values."
            )

        new_feature_hash = joblib.hash(
            new_features
        )

        if joblib.hash(
            old_features
        ) != old_feature_hash:

            raise RuntimeError(
                "Original feature extractor changed."
            )

        # The reused training function validates
        # classifier dimensions against this value.

        frozen.EXPECTED_FEATURES = (
            new_feature_count
        )

        analyze_feature_changes(
            old_features,
            new_features
        )

        analyze_coverage(
            old_features,
            new_features,
            [
                ("contrastive", contrastive),
                ("digital_mirror", domain),
                ("validation", validation),
                ("benchmark", benchmark),
            ]
        )

        print("\n=== ZERO VECTOR ANALYSIS ===")

        analyze_zero_vectors(
            training_matrix,
            combined,
            "training"
        )

        # Transform evaluation data independently
        # using the two fitted feature spaces.

        evaluation_sets = []

        for name, dataset in [
            ("validation", validation),
            ("benchmark", benchmark),
        ]:

            old_matrix = old_features.transform(
                dataset[TEXT_COLUMN]
            )

            new_matrix = new_features.transform(
                dataset[TEXT_COLUMN]
            )

            analyze_zero_vectors(
                old_matrix,
                dataset,
                f"frozen_{name}"
            )

            analyze_zero_vectors(
                new_matrix,
                dataset,
                f"expanded_{name}"
            )

            evaluation_sets.append((
                name,
                dataset,
                old_matrix,
                new_matrix,
            ))

        frequencies = (
            frozen.calculate_frequencies(
                combined
            )
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

        for strategy, config in STRATEGIES.items():

            print("\n" + "=" * 60)

            print(
                f"STRATEGY: {config['name']} | "
                f"ALPHA: {config['alpha']}"
            )

            print("=" * 60)

            frozen_model = joblib.load(
                frozen_model_paths[strategy]
            )

            frozen.verify_feature_space(
                old_features,
                frozen_model.named_steps["tfidf"]
            )

            frozen_classifier = (
                frozen_model.named_steps[
                    "classifier"
                ]
            )

            weights = (
                frozen.calculate_sample_weights(
                    combined,
                    frequencies,
                    config["alpha"]
                )
            )

            frozen.analyze_weights(
                combined,
                weights,
                frequencies,
                strategy
            )

            model_name = (
                f"{EXPERIMENT}{strategy}_lr"
            )

            expanded_model, training_time = (
                frozen.train_model(
                    new_features,
                    frozen_classifier,
                    training_matrix,
                    labels,
                    weights,
                    model_name
                )
            )

            for (
                dataset_name,
                dataset,
                old_matrix,
                new_matrix,
            ) in evaluation_sets:

                (
                    frozen_metrics,
                    frozen_predictions,
                    frozen_classes,
                ) = frozen.evaluate_model(
                    frozen_model,
                    dataset,
                    old_matrix,
                    f"frozen_{strategy}_lr",
                    dataset_name
                )

                (
                    expanded_metrics,
                    expanded_predictions,
                    expanded_classes,
                ) = frozen.evaluate_model(
                    expanded_model,
                    dataset,
                    new_matrix,
                    model_name,
                    dataset_name
                )

                frozen_metrics.update({
                    "feature_space": "frozen",
                    "features": old_feature_count,
                    "alpha": config["alpha"],
                    "training_seconds": np.nan,
                    "feature_fit_seconds": np.nan,
                })

                expanded_metrics.update({
                    "feature_space": "expanded",
                    "features": new_feature_count,
                    "alpha": config["alpha"],
                    "training_seconds": training_time,
                    "feature_fit_seconds": feature_fit_time,
                })

                results.extend([
                    frozen_metrics,
                    expanded_metrics,
                ])

                frozen.compare_models(
                    frozen_metrics,
                    expanded_metrics,
                    frozen_predictions,
                    expanded_predictions,
                    frozen_classes,
                    expanded_classes,
                    strategy,
                    dataset_name
                )

                save_table(
                    pd.DataFrame(results),
                    "results"
                )

        if joblib.hash(
            old_features
        ) != old_feature_hash:

            raise RuntimeError(
                "Original feature extractor changed."
            )

        if joblib.hash(
            new_features
        ) != new_feature_hash:

            raise RuntimeError(
                "Refitted feature extractor changed "
                "during classifier training."
            )

        print("\n=== FINAL RESULTS ===")

        print(
            pd.DataFrame(results)
            .to_string(index=False)
        )

        print("\n=== EXPERIMENT SUMMARY ===")

        print("Models trained: 3")
        print("Training records: 137605")
        print("TF-IDF refitted: YES")
        print("TF-IDF fitted on training data only: YES")
        print(f"Old features: {old_feature_count}")
        print(f"New features: {new_feature_count}")
        print("Vectorizer parameters changed: NO")
        print("Classifier parameters changed: NO")
        print("Weighting formula changed: NO")
        print("Original feature extractor changed: NO")
        print("Switchboard Test evaluated: NO")
        print("Hyperparameter tuning: NO")

    finally:

        frozen.EXPECTED_FEATURES = (
            original_expected_features
        )

        print("\n=== FILE INTEGRITY ===")

        integrity_rows = []

        for path, original_hash in original_hashes.items():

            current_hash = (
                frozen.calculate_sha256(path)
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
