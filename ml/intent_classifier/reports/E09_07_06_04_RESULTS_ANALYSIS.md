# E09.07.6.4 — Preprocessing and Overlap Analysis

## Objective

The contrastive dataset was prepared for controlled model training using the existing text normalization pipeline.

Internal and external text overlaps were analyzed to prevent duplicate training records, conflicting labels, and evaluation data leakage.

## Preprocessing Results

| Metric | Result |
|---|---:|
| Original records | 1,200 |
| Processed records | 1,200 |
| Original columns | 8 |
| Processed columns | 10 |
| Normalized records changed | 1,180 |
| Punctuation removal changes | 1,200 |
| Unique model texts | 1,200 |
| Internal duplicates | 0 |
| Internal label conflicts | 0 |
| Validation checks | 14/14 PASS |

All original records, labels, and metadata were preserved.

## Normalization Diagnostics

A normalization mismatch was detected during reference dataset loading.

Raw Switchboard transcripts contained transcription annotations that had been removed during an earlier cleaning stage.

| Transformation | Matching | Mismatching |
|---|---:|---:|
| Raw text to model text | 66,072 | 69,755 |
| Normalized text to model text | 135,827 | 0 |
| Raw text to normalized text | 0 | 135,827 |

The reference loading procedure was corrected to use the existing normalized text column for Switchboard datasets.

No source datasets were modified.

## External Overlap Results

| Reference dataset | Overlapping records |
|---|---:|
| Switchboard Train | 0 |
| Switchboard Validation | 0 |
| Switchboard Test | 0 |
| Digital Mirror Original | 1 |
| Digital Mirror Eligible | 1 |
| Installation Development | 0 |
| Installation Benchmark | 0 |

One existing domain training duplicate was identified and excluded.

No exact text overlaps with evaluation datasets were detected.

## Final Training Dataset

| Metric | Result |
|---|---:|
| Original records | 1,200 |
| Excluded records | 1 |
| Eligible records | 1,199 |
| Retention rate | 99.92% |
| Intent classes | 12 |
| Preserved contrastive groups | 300 |
| Unchanged groups | 299 |
| Modified groups | 1 |
| Groups with fewer than two intents | 0 |
| Eligibility checks | 16/16 PASS |
| File integrity checks | 8/8 PASS |

The QUESTION class contains 99 records. All other classes contain 100 records.

The excluded record was not replaced.

## Output Dataset

`data/processed/digital_mirror_contrastive_v1/train_eligible.csv`

SHA256:

`bc2fc93bff2b9ef8106cd09728d57f5956eba6dad551e2569b5976766541ccd8`

## Limitations

Only exact normalized text overlaps were analyzed. Semantic similarity and paraphrased duplicates were not evaluated.

The dataset contains repeated linguistic templates and context-dependent expressions identified during previous analysis.

Only 1.25% of the original contrastive records contain three words or fewer.

Full semantic validation has not been completed.

## Conclusion

The contrastive dataset was successfully prepared for controlled training.

All eligibility and file integrity checks were passed. No models were loaded, trained, or evaluated during this experiment.

The effect of the additional contrastive training data will be evaluated in E09.07.6.5.