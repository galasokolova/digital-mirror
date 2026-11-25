# E09.07.6.5.1 — Frozen Feature Space Training

## Objective

The effect of additional contrastive training data was investigated while keeping the existing TF-IDF feature space unchanged.

## Experimental Configuration

| Parameter | Value |
|---|---|
| Algorithm | Logistic Regression |
| Feature extraction | Word and Character TF-IDF |
| Feature space | Frozen |
| Training records | 137,605 |
| New contrastive records | 1,199 |
| Features | 177,043 |
| Intent classes | 12 |
| Weighting strategies | Uniform, Moderate, Strong |

## Installation Benchmark Results

| Strategy | Previous Accuracy | New Accuracy | Previous Macro F1 | New Macro F1 | New Top-3 |
|---|---:|---:|---:|---:|---:|
| Uniform | 62.08% | 67.92% | 62.90% | 68.59% | 87.50% |
| Moderate | 68.75% | 70.42% | 70.58% | 72.11% | 91.25% |
| Strong | 70.00% | 70.83% | 71.73% | 72.45% | 91.25% |

## Main Findings

- Improved benchmark accuracy and Macro F1 were observed across all three weighting strategies.
- The largest accuracy improvement was obtained with Uniform Weighting (+5.83 percentage points).
- The highest benchmark accuracy was obtained with Strong Weighting (70.83%).
- An accuracy of 80% and a Top-3 accuracy of 95% were achieved on the installation-specific expression group by both Moderate and Strong Weighting.
- A reduction in STATEMENT false positives was observed across all strategies. However, STATEMENT recall was also reduced.
- Consistent improvements were observed for AFFIRMATIVE_ANSWER and NEGATIVE_ANSWER.
- Consistent improvements were not achieved for AGREEMENT and ACKNOWLEDGMENT.

## Validation Results

The highest Switchboard validation accuracy (83.19%) and Macro F1 (52.51%) were obtained with Moderate Weighting.

Strong Weighting achieved slightly better installation benchmark results but showed substantially lower recall for affirmative and negative answers on Switchboard validation.

## Training Weight Distribution

Although the contrastive dataset represented only 0.87% of the training records, its share of the total sample weight reached:

- Uniform: 12.85%
- Moderate: 20.10%
- Strong: 23.01%

The combined weight share of both synthetic datasets reached 34.08% under Strong Weighting.

## Experimental Integrity

| Check | Result |
|---|---|
| Data validation | 15/15 passed |
| File integrity | 10/10 passed |
| Models trained | 3/3 |
| Convergence warnings | 0 |
| Zero training vectors | 0 |
| TF-IDF refitted | No |
| Protected files modified | No |
| Switchboard test evaluated | No |
| Hyperparameter tuning performed | No |

## Limitations

The installation benchmark was used for model development and does not represent an independent final evaluation.

The effect of additional contrastive examples cannot be completely separated from the changes in sample weight distribution.

Repeated linguistic templates and context-dependent expressions may limit generalization.

The small differences between Moderate and Strong Weighting do not establish a reliable superiority of either strategy.

## Conclusion

The hypothesis was partially supported. Additional contrastive training data improved overall benchmark performance without refitting the feature extractor.

However, the main difficulties in distinguishing closely related conversational intents were not fully resolved.

Moderate Weighting was retained as the primary dialogue-oriented candidate. Strong Weighting was retained as an alternative candidate.

## Next Step

The TF-IDF feature extractor will be refitted using the extended training dataset.

The resulting models will be compared with their corresponding frozen-feature models using the same weighting strategies and evaluation datasets.

Additional vocabulary and feature coverage will be investigated before further dataset expansion.