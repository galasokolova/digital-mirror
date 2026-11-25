# E09.07.2 — Character TF-IDF Experiment

## Objective

Character-level TF-IDF was evaluated as an alternative to word-level TF-IDF for human intent classification.

## Configuration

- Training records: 136,406
- Character n-grams: (3, 5)
- Maximum features: 100,000
- Classifier: Logistic Regression
- Class weighting: balanced
- Training time: 132.4 seconds

The training datasets and classifier parameters were kept unchanged.

## Results

| Metric | Word TF-IDF | Character TF-IDF |
|---|---:|---:|
| Validation Accuracy | 75.83% | 72.56% |
| Validation Macro F1 | 49.85% | 45.17% |
| Benchmark Accuracy | 62.08% | 54.17% |
| Benchmark Macro F1 | 61.56% | 54.59% |
| Benchmark Top-3 Accuracy | 80.00% | 80.00% |
| AGREEMENT Recall | 15% | 30% |
| DISAGREEMENT Recall | 70% | 80% |
| ACKNOWLEDGMENT Recall | 0% | 0% |

## Findings

- Overall classification performance was reduced.
- AGREEMENT and DISAGREEMENT recognition was improved.
- Fourteen previous errors were corrected, but 33 new errors were introduced.
- Benchmark zero vectors were eliminated.
- Short utterances containing fewer than three characters could not be represented by the selected character n-gram configuration.
- The ACKNOWLEDGMENT classification problem remained unresolved.

## Conclusion

Character TF-IDF was not selected as a replacement for the baseline model. However, complementary information was identified in character-level features.

A combination of word-level and character-level TF-IDF will be evaluated in the next experiment.

The benchmark was used for model development. Independent final evaluation was not performed.

All dataset validation and file integrity checks were passed.