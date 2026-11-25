# E09.07.6 — Contrastive Dataset and Feature Space Adaptation

## Objective

The effect of contrastive training examples and TF-IDF feature space adaptation on intent classification was investigated.

## Dataset Preparation

A synthetic contrastive dataset containing 1,200 expressions across 12 intent classes was created.

Structural validation, semantic sample review, text preprocessing, and exact text overlap analysis were performed.

One duplicate expression was excluded. A total of 1,199 eligible records were retained.

The extended training dataset contained:

| Dataset | Records |
|---|---:|
| Switchboard | 135,827 |
| Digital Mirror V1 | 579 |
| Contrastive V1 | 1,199 |
| Total | 137,605 |

## Experimental Design

Two feature space configurations were investigated:

- Frozen Feature Space: the existing TF-IDF extractor was preserved.
- Expanded Feature Space: TF-IDF was refitted on the extended training dataset.

Three frequency weighting strategies were applied to each configuration.

Classifier and vectorizer parameters were preserved.

## Feature Space Results

| Parameter | Frozen | Expanded |
|---|---:|---:|
| Word features | 77,043 | 77,515 |
| Character features | 100,000 | 100,000 |
| Total features | 177,043 | 177,515 |
| Contrastive word coverage | 76.50% | 98.61% |
| Benchmark word coverage | 77.56% | 79.16% |

A substantial improvement in training-domain feature coverage was observed.

However, only a limited improvement in classification performance was obtained.

## Installation Benchmark Results

| Model | Accuracy | Macro F1 | Top-3 Accuracy |
|---|---:|---:|---:|
| Frozen A | 67.92% | 68.59% | 87.50% |
| Expanded A | 67.08% | 67.98% | 88.33% |
| Frozen B | 70.42% | 72.11% | 91.25% |
| Expanded B | 70.42% | 72.24% | 91.25% |
| Frozen C | 70.83% | 72.45% | 91.25% |
| Expanded C | 71.25% | 73.02% | 91.25% |

The highest benchmark accuracy and Macro F1 were obtained with Expanded C.

However, only one additional correct prediction was obtained compared with Frozen C.

## Remaining Classification Problems

| Intent | Expanded C Recall | Target |
|---|---:|---:|
| AGREEMENT | 60% | 80% |
| ACKNOWLEDGMENT | 50% | 75% |

The recognition of agreement and acknowledgment remains insufficient.

Limited linguistic diversity and the low proportion of short conversational expressions were identified as potential contributing factors.

## Research Integrity

All 15 data validation checks and all 13 protected file integrity checks were passed during the expanded feature space experiment.

All three expanded classifiers converged successfully.

TF-IDF was fitted exclusively on training data.

The Switchboard test dataset was not evaluated.

## Limitations

The Installation Benchmark has been repeatedly used for model development and comparison. An independent final evaluation is therefore required.

The observed improvement after TF-IDF refitting is too small to establish a statistically reliable advantage.

The synthetic contrastive dataset contains repeated linguistic structures and context-dependent expressions.

## Conclusion

Contrastive domain adaptation produced a modest improvement in installation-specific classification.

Refitting TF-IDF substantially improved feature coverage but did not consistently improve classification performance across weighting strategies.

Expanded C was retained as the primary development candidate. Frozen C and Expanded B were retained as alternatives.

Further improvement will be investigated through the introduction of natural conversational expressions, particularly short utterances.

## Next Experiment

E09.07.7 — Natural Small Talk Dataset

A new training dataset will be created to improve linguistic diversity and the representation of short conversational expressions.

The effect of the additional dataset will be evaluated through controlled experiments.