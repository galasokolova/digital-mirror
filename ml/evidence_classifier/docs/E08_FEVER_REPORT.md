
**Latest confirmed commit:**  `Add FEVER contrastive review and E08 negation evaluation`  


## 1. Current baseline — E06b

- **Dataset:** FEVER, using SUPPORTS and REFUTES examples with Wikipedia evidence/context.
- **Model:** separate claim and evidence TF-IDF representations + balanced Logistic Regression.
- **Development accuracy:** **0.7796** (approximately 77.96%).
- **Development macro F1:** **0.7788**.
- **Saved model (local, gitignored):** `ml/evidence_classifier/models/fever_relationship_logreg.joblib`.
- **Training script:** `ml/evidence_classifier/src/models/train_fever_relationship_model.py`.
- **Model components saved:** `model`, `claim_vectorizer`, `evidence_vectorizer`, `similarity_vectorizer`, `scaler`, `feature_version=e06b`.

- **Limitation:** binary classifier *must* choose SUPPORTS or REFUTES; it has not been trained on FEVER's NOT ENOUGH INFO class. Predicted probabilities are not evidence of calibrated certainty.

## 2. Experiments completed

| Experiment | Purpose | Result / takeaway |
|---|---|---|
| Earlier LIAR baseline | Classify claim text | Accuracy ~0.6690; weak sensitivity to negation; not suitable as a reliable fact checker |
| E06b | FEVER claim–evidence relationship classifier | Dev accuracy 0.7796; macro F1 0.7788 |
| E06c | Five controlled claim-fixed, evidence-changed pairs | 1/5 predicted-label flips; 6/10 individual predictions correct; 1/5 pairs both correct |
| E06d | Ablation of relationship features | Full accuracy 0.779579; TF-IDF-only 0.735722; number features and overlap had the clearest positive contribution |
| E07a–c | Mine and audit similar opposite-label FEVER claims | 1,000 candidates; 932 after deduplication; 160 pairs with `not`/`never` in added tokens |
| E07d | Manual review workflow | Review script created; first three cases annotated, review unfinished |
| E08 | Fixed supporting evidence, compare SUPPORTS claim against REFUTES counterpart | 160 pairs / 320 predictions; 147/160 prediction flips; 307/320 individual predictions match assigned labels; 147/160 pairs both correct |

### E06d ablation details

| Variant | Dev accuracy | Macro F1 |
|---|---:|---:|
| Baseline (TF-IDF only) | 0.735722 | 0.734736 |
| Full (all features) | 0.779579 | 0.778819 |
| No Cosine | 0.779503 | 0.778741 |
| No Numbers | 0.760718 | 0.759994 |
| No Negation | 0.779276 | 0.778519 |
| No Overlap | 0.772231 | 0.771472 |
| No Title Feature | 0.779730 | 0.778935 |



## 3. E08 methodology and limitations

- Input: `ml/evidence_classifier/data/experiments/fever/e07c_audited_pairs.csv`.
- Selection: `added_tokens` contains `not` or `never` (160 candidate pairs).
- For each pair, test the SUPPORTS claim and the REFUTES claim **against the same `supports_evidence` and `supports_titles`**.
- Context string constructed as `Wikipedia article: {wiki_titles}. {evidence}`.
- Test script: `ml/evidence_classifier/src/experiments/contrastive/test_fever_negation_pairs.py`.
- Results committed: `ml/evidence_classifier/data/experiments/fever/e08_negation_results.csv` (320 rows, 9 columns).
- Columns: `pair_id`, `claim`, `evidence`, `wiki_titles`, `context_evidence`, `expected`, `prediction`, `p_supports`, `correct`.

**Do not overclaim E08 performance:** 95.94% agreement with assigned labels is a diagnostic result on *selected* pairs, **not** a validated independent fact-checking benchmark. Possible FEVER training overlap was not ruled out. The selected evidence may not suffice for both claims. Some pairs change more than simple negation. Titles/context construction may not exactly reproduce original FEVER records.

## 4. E08 — 13 apparent errors to audit in E09

All 13 assigned-label mismatches occur on the SUPPORTS member: the model predicted REFUTES. Their REFUTES counterparts were predicted REFUTES. These are **errors relative to E08's assigned labels**, pending evidence review.

| Pair ID | SUPPORTS claim (abbreviated) | P(SUPPORTS) | Initial review hypothesis |
|---|---|---:|---|
| 14 | Annie Parisse is her professional name | 0.4660 | Evidence supports claim; model error likely |
| 19 | Nepal is 17 miles away from Bangladesh | 0.1519 | Approximate vs exact distance; ambiguity |
| 33 | Adderall's active ingredient is amphetamine | 0.4406 | Evidence supports substance relation; model error likely |
| 35 | Fred Trump's father's name was Frederick Trump | 0.3951 | Evidence does not identify Fred Trump's father |
| 40 | Joffrey Baratheon's mother is Cersei Lannister | 0.4699 | Evidence supports claim; model error likely |
| 48 | Czech Republic's capital is Prague | 0.3628 | Evidence says 'It'; needs title/coreference |
| 57 | Czech Republic's largest city is Prague | 0.3437 | Same title/coreference issue |
| 61 | Monica Seles is ethnically Hungarian | 0.4003 | Evidence says 'she'; needs title/coreference |
| 67 | Halle Berry's middle name is Maria | 0.4936 | Evidence supports claim; borderline score |
| 72 | Brown bear is a carnivore | 0.3800 | 'Carnivoran' (taxonomy) vs 'carnivore' (diet): ambiguous |
| 80 | Rafael Nadal first man to win Monte-Carlo Masters ten times | 0.4357 | Evidence supports claim; model error likely |
| 156 | Gold is mined in China | 0.4853 | Evidence supports claim; borderline score |
| 160 | Brown bear is a carnivorous mammal | 0.2470 | Evidence says 'carnivoran'; sufficiency unclear; counterpart is not pure negation |

**All classifications in the last column are preliminary, not final adjudications.**

### Pair 35 — inspected in detail

- `wiki_titles`: `Fred Trump`.
- Claim A (assigned SUPPORTS): `Fred Trump's father's name was Frederick Trump.`
- Claim B (assigned REFUTES): `Fred Trump's father's name was not Frederick Trump.`
- Shared evidence: `Frederick Christ Trump ... was ... father of Donald Trump ...`.
- The passage identifies Fred Trump / Frederick Christ Trump and **his children**, but does **not** identify *his own father*.
- **Assessment for this exact E08 evidence:** `INSUFFICIENT` for determining either claim; neither SUPPORTS nor REFUTES is justified by this passage alone.
- This does **not** prove the original FEVER annotation is wrong: the original item may have used other evidence. Check original FEVER record before making that claim.
- The model's REFUTES output is still unsupported by the supplied passage, even if the E08 assigned SUPPORTS label is also unsupported.

## 5. E09 — next work (do not repeat prior audits)

**Research question:** How many of the 13 E08 mismatches are genuine classifier failures versus insufficient, ambiguous, or incorrectly assembled evidence?

1. **Audit the exact evidence used by E08**, including `wiki_titles` and `context_evidence`; compare with original FEVER evidence when possible.
2. Annotate the 13 cases separately from E08's original results. Suggested fields:
   - `pair_id`
   - `evidence_assessment`: `SUFFICIENT`, `INSUFFICIENT`, `AMBIGUOUS`, or `IRRELEVANT`
   - `justified_label`: `SUPPORTS`, `REFUTES`, or `NOT_ENOUGH_INFO` (only if evidence supports a clear determination)
   - `error_category`: e.g. `MODEL_ERROR`, `EVIDENCE_GAP`, `COREFERENCE`, `SEMANTIC_AMBIGUITY`, `TEST_CONSTRUCTION`
   - `review_notes`
3. Start from **pair 35** (already provisionally assessed as insufficient), then review the other 12; pair 72 is a useful next case for scientific terminology.
4. Save the new annotations in a separate E09 CSV; **do not overwrite E08 results or retroactively change its reported metrics**. Report adjusted findings only with explicit annotation criteria.
5. Only then design and evaluate a real `NOT ENOUGH INFO` capability. A simple probability threshold is an abstention heuristic, **not** the same as evidence sufficiency. FEVER contains NEI examples, but training and evaluation will require carefully assembled claim/evidence inputs and an independent test.

**Current stopping point:** E08 committed and pushed. E09 analysis discussed but no E09 code/annotations created yet.

## 6. Existing Ukraine/Russia external datasets — reuse later, not now

Files in `ml/evidence_classifier/data/processed/` (local; check `.gitignore` before assuming they exist on a fresh clone):

| File | Rows | Relevant columns |
|---|---:|---|
| `euvsdisinfo_false.csv` | 730 | `claim`, `label`, `date`, `original_text`, `factcheck_text`, `factcheck_url` |
| `claimreview_true_candidates.csv` | 86 | `claim`, `label`, `date`, `original_rating`, `fact_checker`, `factcheck_url` |
| `claims_review.csv` | 50 | `claim`, `original_text`, `factcheck_text`, `factcheck_url`, `checkable`, `notes` |
| `relations_candidates.csv` | 571 | `claim`, `label`, `date`, `original_rating`, `fact_checker`, `factcheck_url` |

All 50 `claims_review.csv` records match `euvsdisinfo_false.csv` by `factcheck_url`; manual checkability labels: **22 YES, 17 UNCERTAIN, 11 NO**.

**Caution:** `factcheck_text` often describes a disinformation narrative instead of supplying directly sufficient factual evidence. `claimreview_true_candidates.csv` has fact-check URLs but no evidence text. Source ratings are heterogeneous, and candidate labels should not be treated as independently verified ground truth without review. Some datasets may overlap.

**Future plan:** after E09 and NEI work, build an evidence-retrieval pipeline with provenance, source URL, publication/event dates, and evidence sufficiency checks. Evaluate on selected, independently reviewed external claims. Do not re-run the early claim-only classification experiment.

## 7. Git and file locations

```text
Branch: experiment/e09-evidence-sufficiency
Remote: origin/experiment/e09-evidence-sufficiency

ml/evidence_classifier/src/models/train_fever_relationship_model.py
ml/evidence_classifier/src/evaluation/test_fever_relationship_contradictions.py
ml/evidence_classifier/src/experiments/fever_baselines/ablate_fever_relationship_features.py
ml/evidence_classifier/src/experiments/contrastive/review_fever_contrastive_pairs.py
ml/evidence_classifier/src/experiments/contrastive/test_fever_negation_pairs.py
ml/evidence_classifier/data/experiments/fever/e07c_audited_pairs.csv
ml/evidence_classifier/data/experiments/fever/e08_negation_results.csv
ml/evidence_classifier/models/fever_relationship_logreg.joblib
```

