# E09.07.3 — Combined Word and Character TF-IDF

## Objective

Word-level and character-level TF-IDF features were combined to improve human intent classification.

## Configuration

- Training records: 136,406
- Word n-grams: (1, 2)
- Character n-grams: (3, 5)
- Feature combination: FeatureUnion
- Transformer weights: 1.0 and 1.0
- Classifier: Logistic Regression
- Class weighting: balanced
- Total features: 177,043
- Training time: 191.63 seconds

The training datasets and classifier parameters were kept unchanged.

## Results

| Metric | Word | Character | Combined |
|---|---:|---:|---:|
| Validation Accuracy | 75.83% | 72.56% | 76.34% |
| Validation Macro F1 | 49.85% | 45.17% | 51.45% |
| Benchmark Accuracy | 62.08% | 54.17% | 61.25% |
| Benchmark Macro F1 | 61.56% | 54.59% | 61.94% |
| Benchmark Top-3 Accuracy | 80.00% | 80.00% | 83.33% |
| AGREEMENT Recall | 15% | 30% | 30% |
| DISAGREEMENT Recall | 70% | 80% | 80% |
| ACKNOWLEDGMENT Recall | 0% | 0% | 0% |
| ACKNOWLEDGMENT Top-3 Recall | 0% | 0% | 15% |
| Short Utterance Accuracy | 61.67% | 60.00% | 70.00% |

## Findings

- The highest Macro F1 and Top-3 Accuracy among the three evaluated architectures were achieved.
- Benchmark Accuracy was slightly reduced compared with Word TF-IDF.
- Zero feature vectors were eliminated on both evaluation datasets.
- Short utterance classification was improved.
- Eleven previous errors were corrected, while thirteen new errors were introduced.
- Two utterances were correctly classified only by the combined model.
- ACKNOWLEDGMENT remained unresolved.
- Excessive STATEMENT prediction remained a major classification problem.

## Conclusion

Combined TF-IDF was retained as a candidate architecture for further experiments.

The results indicate that feature combination alone is insufficient to resolve the main classification problems.

A Linear SVM classifier will be evaluated next. Training frequency weighting and additional domain-specific training data will subsequently be investigated.

All dataset validation and file integrity checks were passed.

The benchmark was used for model development. Independent final evaluation was not performed.