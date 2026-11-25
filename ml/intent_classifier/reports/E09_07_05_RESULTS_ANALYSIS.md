# E09.07.5 — Training Frequency Weighting

## Objective

The effect of repeated expressions on intent classification was investigated.

Three weighting strategies were compared using Logistic Regression and Linear SVM:

- A — Uniform weighting
- B — Inverse square root frequency weighting
- C — Inverse frequency weighting

## Experimental Setup

| Parameter | Value |
|---|---|
| Training records | 136,406 |
| Switchboard records | 135,827 |
| Digital Mirror records | 579 |
| Validation records | 16,913 |
| Digital Mirror benchmark | 240 |
| Intent classes | 12 |
| TF-IDF features | 177,043 |

The existing word and character TF-IDF feature space was preserved. No new training records were added.

## Digital Mirror Results

| Model | Accuracy | Macro F1 | Top-3 Accuracy |
|---|---:|---:|---:|
| A — LR | 62.08% | 0.6290 | 82.92% |
| B — LR | 68.75% | 0.7058 | 89.58% |
| C — LR | 70.00% | 0.7173 | 89.58% |
| A — SVM | 58.33% | 0.6218 | 92.50% |
| B — SVM | 64.17% | 0.6857 | 92.08% |
| C — SVM | 63.33% | 0.6764 | 92.50% |

## Main Improvements

The strongest benchmark results were obtained with inverse frequency weighting and Logistic Regression.

| Metric | Uniform LR | Strong LR |
|---|---:|---:|
| Accuracy | 62.08% | 70.00% |
| Macro F1 | 0.6290 | 0.7173 |
| ACKNOWLEDGMENT recall | 0% | 55% |
| AGREEMENT recall | 30% | 50% |
| UNCERTAINTY recall | 40% | 60% |
| STATEMENT false positives | 59 | 34 |

A total of 28 previous classification errors were corrected, while 9 new errors were introduced.

## Convergence Recovery

The strongly weighted SVM initially failed to converge within 5,000 iterations.

During E09.07.5.1, the iteration limit was increased to 20,000. Convergence was achieved after 5,097 iterations.

| Verification | Result |
|---|---|
| Dataset validation | 16/16 passed |
| Existing models retrained | 0 |
| Models evaluated | 6 |
| Convergence warnings | 0 |
| File integrity checks | 13/13 passed |

All protected files remained unchanged during recovery.

## Limitations

The uniform models did not reproduce all predictions from the previous automatically class-weighted models.

Strong frequency weighting improved the Digital Mirror results but reduced Macro F1 on Switchboard validation.

The Digital Mirror benchmark was used for model development. Independent final evaluation has not yet been performed.

## Conclusion

Frequency weighting improved domain-specific intent classification without adding new training records.

The strongly weighted Logistic Regression model was retained as the primary candidate. The moderately weighted model was retained as an alternative.

## Next Step

E09.07.6 — Contrastive Domain Dataset

New training examples will be introduced to improve the distinction between similar conversational intents and reduce dependence on topic-specific vocabulary.