# E10 — Cross-Domain Evidence Sensitivity Test

**Model:** E06b — TF-IDF + Logistic Regression  
**Dataset:** EUvsDisinfo  
**Date:** 2026-11-22

## Objective

Test whether the existing FEVER-trained model can correctly interpret contradictory evidence from a real-world disinformation case.

## Method

One claim from EUvsDisinfo was tested with two opposing evidence statements.

**Claim:** Dutch court recognised the conflict in Donbas in 2014 as non-international.

| Test | Evidence meaning | Expected | Predicted | P(SUPPORTS) |
|---|---|---|---|---|
| 1 | The conflict was international | REFUTES | REFUTES | 0.1942 |
| 2 | The conflict was non-international | SUPPORTS | REFUTES | 0.3099 |

The first evidence statement was extracted from the fact-checking text. The second was artificially constructed to test the model's sensitivity to contradictory evidence.

## Results

- Correct predictions: **1/2**
- Prediction changed: **No**
- Change in P(SUPPORTS): **+0.1157**

The model responded to the changed evidence but continued predicting REFUTES.

## Conclusion

E06b demonstrates some sensitivity to evidence changes, but this test reveals a limitation in recognising contradictory statements.

A single test is insufficient to measure cross-domain accuracy.

## Next Steps

1. Prepare a small, manually verified claim–evidence test set from EUvsDisinfo and AI4Debunk.
2. Evaluate cross-domain performance.
3. Investigate errors before deciding whether retraining is necessary.

**Model status:** E06b remains unchanged. No retraining was performed.