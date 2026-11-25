# Evidence classifier

E06b uses TF-IDF, pair features and logistic regression to return `SUPPORTS` or `REFUTES` for a claim and a supplied evidence passage. Spring Boot sends both texts to `POST /predict` and uses `p_supports` as the machine's score from 0 to 1. The curated label in the claim data supplies the reference value for distortion. For `REFUTES`, distortion equals the agreement score. For `SUPPORTS`, it equals one minus that score.

| Location | Contents |
|---|---|
| `src/data_preparation/` | Scripts for preparing source datasets. |
| `src/models/` | E06b training and feature extraction. |
| `src/final_model/` | FastAPI and the E06b predictor. |
| `src/evaluation/` | Diagnostic dataset builders, evaluators and an orchestrator smoke check. |
| `src/experiments/` | Earlier datasets, FEVER baselines, contrastive tests, NLI, and E10–E14 evidence robustness experiments. |
| `data/evaluation/datasets/` | Versioned diagnostic JSON fixtures. |
| `data/evaluation/results/` | Saved V2 and V3 diagnostic results. |
| `docs/` | Research reports and figures. |

## Restore or train E06b

Place a trusted existing bundle at `ml/evidence_classifier/models/fever_relationship_logreg.joblib` and use the same scikit-learn version that produced it. To train from FEVER, place `train.jsonl`, `shared_task_dev.jsonl` and `wiki-pages.zip` in `ml/evidence_classifier/data/raw/fever/`. From the repository root, with the ML Python environment active, run:

```bash
python -m ml.evidence_classifier.src.data_preparation.prepare_fever
python -m ml.evidence_classifier.src.data_preparation.prepare_fever_training_context
python -m ml.evidence_classifier.src.models.train_fever_relationship_model
```

Start FastAPI from the repository root:

```bash
python -m uvicorn final_model.api:app --app-dir ml/evidence_classifier/src --port 8000
```

Check `GET /health`. With FastAPI running, use `python ml/evidence_classifier/src/evaluation/evaluate_diagnostic_v3.py` to refresh the V3 report. The orchestrator smoke check additionally requires Spring Boot on port 8080:

```bash
python ml/evidence_classifier/src/evaluation/test_claims_api.py
```

Run research modules with shared imports from the repository root, for example `python -m ml.evidence_classifier.src.experiments.evidence_robustness.e13_negation_generalization`. Scripts in this group can require ignored inputs or results from earlier experiments. E14 uses E13's CSV and a pretrained NLI model.

## Research reports

- [Initial ML research](docs/ML_RESEARCH_REPORT.md)
- [FEVER experiments](docs/FEVER_EXPERIMENTS_REPORT.md)
- [E08 negation experiments](docs/E08_FEVER_REPORT.md)
- [E10 cross-domain evidence sensitivity](docs/E10_CROSS_DOMAIN_TEST.md)
- [E10 cross-domain and negation tests](docs/E10_NEGATION_TESTING.md)

The E06b report records 77.96% accuracy and 0.7788 macro F1 on FEVER dev. Treat these as FEVER results. Evaluate the installation's curated claims and evidence separately. The exported branch does not include E09 review batches, raw or processed FEVER data, or a trained bundle.
