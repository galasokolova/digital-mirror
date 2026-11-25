# Digital Mirror — FEVER Experiments Report

**Status:** Research in progress, through experiment E06d
**Project:** Digital Mirror
**Purpose:** Record the full research path, including unsuccessful experiments, decisions, measured results, and remaining problems.

> This document continues `docs/ML_RESEARCH_REPORT.md`. The original report covers EUvsDisinfo, ClaimReview, and LIAR. It should be preserved, not replaced.

## 1. Why I changed direction

My first model used TF-IDF and Logistic Regression to predict `TRUE` or `FALSE` from a political statement alone. It reached **66.90% accuracy** on the LIAR validation set, but it made incorrect predictions on factual statements and was insensitive to some negations. I concluded that learning patterns in a claim is not the same as verifying it.

**New research question:** Can a model I train myself distinguish whether a supplied piece of evidence **supports** or **refutes** a claim?

This is **evidence-based classification**, not independent fact-checking. A correct prediction depends on the quality and relevance of the supplied evidence.

## 2. FEVER dataset and preparation

FEVER provides claims, Wikipedia evidence references, and three labels: `SUPPORTS`, `REFUTES`, and `NOT ENOUGH INFO` (NEI).

| Source / stage | Records | Notes |
|---|---:|---|
| Official FEVER train | 145,449 claims | Original JSONL |
| Official FEVER shared-task dev | 19,998 claims | Original JSONL |
| Prepared train evidence | 219,697 rows | 108,888 distinct verified claims |
| Prepared dev evidence | 24,454 rows | 13,202 distinct verified claims |
| Wikipedia archive | 109 JSONL files | Used to resolve referenced evidence |

**Preparation:** I resolved FEVER evidence references against Wikipedia pages, assembled complete evidence sets, and saved them in `ml/evidence_classifier/data/processed/fever/`. The prepared verified files contain `claim_id`, `claim`, `evidence_set_id`, `evidence`, `label`, and `split`. Later I added `wiki_titles` and `context_evidence` (Wikipedia title followed by evidence text).

**Important decisions and limitations:**

- For model training and evaluation, I selected **one evidence set per distinct claim** (sorting by `claim_id` and `evidence_set_id` and retaining the first). This reduces repeated-claim leakage within each split but discards alternative evidence sets.
- I trained and evaluated binary `SUPPORTS` / `REFUTES` classification. **NEI was not included** in these models; this is not yet a three-class FEVER system.
- The train and dev claim IDs were checked to be disjoint.
- An evidence integrity check of **300 selected dev claims** found **300/300** matches to the original FEVER reference, Wikipedia title, and evidence text. This supports the sampled extraction, not a guarantee that every row is perfect.
- Evidence is **provided from FEVER**. The project does **not yet retrieve evidence automatically** for an arbitrary new claim.

## 3. Early experiments: text alone versus evidence

I started with TF-IDF + Logistic Regression because I could train and inspect the model myself.

| Experiment | Approach | Accuracy | Macro F1 | What I learned |
|---|---|---:|---:|---|
| E01 | Claim Only, FEVER Dev | 0.7394 | 0.7385 | Strong baseline, but no evidence reasoning |
| E01 | Claim + Evidence, FEVER Dev | 0.6994 | 0.6988 | Simply adding evidence **made results worse** |
| E04 | Claim Only, balanced 300-claim sample | 0.7600 | — | Small-sample comparison only |
| E04 | Claim + Evidence, same sample | 0.7000 | — | Repeated the same problem |

**Difficulty:** Adding relevant text did not automatically teach the model how the two texts relate. The classifier could still rely on wording patterns in the claim.

## 4. Pretrained model as a research reference (E03/E04)

I tested the pretrained `cross-encoder/nli-deberta-v3-small` to see whether a model built for sentence relationships could do better. **I did not train DeBERTa myself; it is a benchmark, not my own model.**

| Input to pretrained DeBERTa | Accuracy | Macro F1 | Evaluation |
|---|---:|---:|---|
| Original evidence | 0.7633 | 0.8225 | 300 selected dev claims |
| Evidence with correct Wikipedia article title | **0.8933** | **0.9257** | Same 300 claims |
| Evidence with random Wikipedia title | 0.7467 | 0.7938 | Same 300 claims |

**Interpretation:** The article title carried useful context for the pretrained model. A random title degraded performance, which motivated adding correct titles to my own training data. These numbers are from **300 examples**, not the full 13,202-claim dev evaluation, so they must not be presented as directly comparable leaderboard results. Neutral predictions were counted as incorrect in the binary task; the reported macro F1 covered the two target classes.

## 5. My first context-aware model (E05a)

I trained my own TF-IDF + Logistic Regression model using claim text, evidence with Wikipedia title, and six manually designed relationship features (word overlap, title overlap, negation difference, and numeric matches/mismatches).

| Evaluation | Result |
|---|---:|
| Train claims | 108,888 |
| Dev claims | 13,202 |
| Accuracy | **0.7268** |
| Macro F1 | **0.7256** |

Confusion matrix (rows: actual `REFUTES`, `SUPPORTS`; columns: predicted `REFUTES`, `SUPPORTS`):

```text
[[4364, 2241],
 [1366, 5231]]
```

**Difficulty:** Despite extra information and engineered features, this version did not beat the stronger Claim Only baseline. I needed to separate the claim and evidence representations and test whether the model actually used evidence.

## 6. Controlled feature comparison (E05b)

To make the comparison fair, I used the same train/dev splits, TF-IDF settings, and Logistic Regression configuration across four variants.

| Variant | Input | Accuracy | Macro F1 |
|---|---|---:|---:|
| A | Claim Only | **0.7369** | **0.7359** |
| B | Claim + Evidence in one text | 0.6865 | 0.6858 |
| C | Claim + title/context in one text | 0.6941 | 0.6938 |
| D | Separate claim and context TF-IDF vectors | 0.7357 | 0.7347 |

**Finding:** Keeping separate text representations recovered most of the lost performance, but still did not outperform Claim Only. The title was not a universal fix. The E05b Claim Only result differs slightly from E01 because the experimental feature settings differed.

## 7. Does evidence influence the model? (E05c)

I kept the claims unchanged and replaced their evidence with evidence from other dev claims.

| Evidence condition | Accuracy | Macro F1 | Predictions changed vs original |
|---|---:|---:|---:|
| Original | 0.7357 | 0.7347 | — |
| Random replacement | 0.7246 | 0.7238 | 13.51% |
| Replacement from an opposite-label claim | 0.7126 | 0.7116 | 13.79% |

Mean absolute change in `P(SUPPORTS)`: **0.0865** (random) and **0.0873** (opposite-label).

**Finding:** Evidence influenced predictions, but many predictions remained unchanged. An opposite-label example is **not necessarily a logically contradictory evidence statement** for the original claim. Also, swapping `context_evidence` changes both the title and evidence, so this test cannot isolate their effects.

## 8. First controlled contradiction test (E06a)

I manually constructed five pairs: each pair had the **same claim and Wikipedia title**, but one evidence sentence supported the claim and the other contradicted it. These are synthetic diagnostic statements, **not verified source quotations**.

**Results using E05a:** **5/10** correct predictions; **0/5** pairs changed predicted class. In all five pairs, the model predicted `SUPPORTS` for both evidence variants. Nevertheless, `P(SUPPORTS)` fell for every contradictory variant.

**Problem identified:** The model was sensitive to changed words but could not reliably turn that sensitivity into a correct `REFUTES` decision.

## 9. Relationship model (E06b)

I redesigned my own model with separate TF-IDF representations for the claim and evidence, plus seven numeric relationship features:

1. Claim token overlap
2. Jaccard-style token overlap
3. Claim/title overlap
4. Negation mismatch
5. Number overlap
6. Number mismatch
7. TF-IDF cosine similarity between claim and evidence

All features were calculated from the training data; the scaler was fitted on train only. I trained a balanced Logistic Regression model and evaluated it on the same 13,202 dev claims.

| Metric | E05b separate TF-IDF | **E06b relationship model** |
|---|---:|---:|
| Accuracy | 0.7357 | **0.7796** |
| Macro F1 | 0.7347 | **0.7788** |

**Improvement:** **+4.39 percentage points** in accuracy over the directly comparable separate-TF-IDF baseline.

E06b confusion matrix (actual rows / predicted columns: `REFUTES`, `SUPPORTS`):

```text
[[4759, 1846],
 [1064, 5533]]
```

**Remaining weakness:** The model incorrectly classified **1,846** refuting examples as supporting, compared with **1,064** supporting examples incorrectly classified as refuting. The aggregate improvement does not establish robust contradiction understanding.

Saved model: `ml/evidence_classifier/models/fever_relationship_logreg.joblib`.

## 10. Repeating the contradiction test (E06c)

I used the same five controlled pairs as E06a, this time with the E06b model.

| Claim (shortened) | P(SUPPORTS), supporting evidence | P(SUPPORTS), contradictory evidence | Correct pair? |
|---|---:|---:|---|
| Marie Curie born in Poland | 0.8814 | 0.8343 | No |
| Eiffel Tower located in Paris | 0.5057 | 0.3236 | **Yes** |
| Einstein born in Germany | 0.8959 | 0.7837 | No |
| *Titanic* released in 1997 | 0.9880 | 0.7650 | No |
| Water freezes at 0°C | 0.9681 | 0.5632 | No |

| Measure | E06a | E06c |
|---|---:|---:|
| Correct individual predictions | 5/10 | **6/10** |
| Pairs with a changed predicted class | 0/5 | **1/5** |
| Pairs with both predictions correct | 0/5 | **1/5** |

**Finding:** In **all five** pairs, contradictory evidence lowered `P(SUPPORTS)`. However, only the Eiffel Tower pair crossed the classification threshold. The model still often favored `SUPPORTS` despite explicit contradiction. Five synthetic pairs are a useful debugging check, not a statistically representative benchmark.

Saved results: `ml/evidence_classifier/data/experiments/fever/e06c_controlled_contradictions.csv`.

## 11. Which features actually helped? (E06d ablation)

I retrained seven variants with the same dataset and model settings, removing one group of relationship features at a time.

| Variant | Accuracy | Macro F1 | Accuracy vs full |
|---|---:|---:|---:|
| TF-IDF Only | 0.7357 | 0.7347 | −4.39 pp |
| **Full (all seven)** | **0.7796** | **0.7788** | — |
| No Cosine | 0.7795 | 0.7787 | −0.01 pp |
| No Numbers | 0.7607 | 0.7600 | **−1.89 pp** |
| No Negation | 0.7793 | 0.7785 | −0.03 pp |
| No Overlap | 0.7722 | 0.7715 | −0.74 pp |
| No Title Feature | 0.7797 | 0.7789 | +0.01 pp |

**Finding:** The **numeric features** contributed the largest measurable improvement in this ablation; token overlap also helped. Removing cosine similarity, negation mismatch, or title overlap barely changed aggregate accuracy in this setup. These tiny differences are not evidence that those features never matter. Feature groups can be correlated, and this experiment did not quantify statistical uncertainty.

**Important detail:** `No Title Feature` removes only the **numeric title-overlap feature**. The Wikipedia title is still included in the evidence TF-IDF text; this is **not** a full test of removing article titles.

Saved results: `ml/evidence_classifier/data/experiments/fever/e06d_feature_ablation.csv`.

## 12. Current conclusions

**What worked**

- I built and trained my own evidence-aware model with TF-IDF, handcrafted relationship features, and Logistic Regression.
- E06b reached **77.96% accuracy / 0.7788 Macro F1** on the held-out FEVER dev claims.
- Relationship features improved accuracy by **4.39 percentage points** over the comparable separate-TF-IDF baseline.
- Numeric comparison and word overlap were particularly useful.

**What did not work yet**

- Simply concatenating claim and evidence made the classifier worse.
- Correct Wikipedia titles helped a pretrained DeBERTa benchmark substantially, but did not automatically fix my own models.
- Even E06b correctly handled **only 1/5** manually constructed contradiction pairs.
- The model does not handle `NOT ENOUGH INFO` and does not automatically retrieve or verify external evidence.

**Interpretation for Digital Mirror:** The model's output represents a learned **SUPPORTS-versus-REFUTES classification for supplied evidence**, not a trustworthy probability that a claim is true in the real world. The installation should not label its score as objective truth or present it as independently verified fact.

## 13. Next research steps (planned, not completed)

**E07 — Better contradiction learning.** Investigate verified, carefully constructed contrastive evidence examples and training strategies that make the model respond to changed entities, dates, numbers, and negation. Keep E06b as a saved baseline. Avoid assigning labels to automatically swapped evidence without checking whether the new evidence actually supports or refutes the original claim.

**Evaluate beyond accuracy.** Reuse the controlled contradiction pairs, add a larger reviewed test set, examine class-specific errors, and consider confidence calibration. Avoid repeatedly optimizing on the same FEVER dev set without a separate final test set.

**Longer-term system work.** Explore evidence retrieval and NEI handling. Only after these components are validated should the installation use the model for unseen claims without manually supplied evidence.

## 14. Reproducibility and files

| File | Purpose |
|---|---|
| `ml/evidence_classifier/src/data_preparation/prepare_fever.py` | Prepare FEVER evidence sets |
| `ml/evidence_classifier/src/data_preparation/prepare_fever_training_context.py` | Add Wikipedia article titles |
| `ml/evidence_classifier/src/experiments/fever_baselines/train_fever_context_model.py` | Train E05a |
| `ml/evidence_classifier/src/experiments/fever_baselines/compare_fever_features.py` | Run E05b |
| `ml/evidence_classifier/src/experiments/fever_baselines/test_fever_evidence_sensitivity.py` | Run E05c |
| `ml/evidence_classifier/src/experiments/fever_baselines/test_controlled_contradictions.py` | Run E06a |
| `ml/evidence_classifier/src/models/train_fever_relationship_model.py` | Train E06b |
| `ml/evidence_classifier/src/evaluation/test_fever_relationship_contradictions.py` | Run E06c |
| `ml/evidence_classifier/src/experiments/fever_baselines/ablate_fever_relationship_features.py` | Run E06d |
| `ml/evidence_classifier/models/fever_relationship_logreg.joblib` | Saved E06b model |
| `ml/evidence_classifier/data/experiments/fever/` | Experiment CSV outputs |

**Reporting note:** Figures above are from recorded experiment outputs. Where only rounded metrics were retained, results are shown to four decimals; no uncertainty estimates have been computed yet.
