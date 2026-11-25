# E09.07.6.3.1 — Semantic Review

## Objective

A semantic review was conducted to examine intent correctness, conversational naturalness, and context dependency before model training.

## Methodology

A stratified sample of 180 expressions was reviewed using an LLM-assisted annotation procedure.

Three expressions from each intent-style combination were included.

The expressions were classified before comparison with the original intent labels. The supplied conversational context was considered during classification.

## Results

| Metric | Result |
|---|---:|
| Reviewed records | 180 |
| Intent classes | 12 |
| Annotation agreement | 100% |
| Matching labels | 180 |
| Conflicting labels | 0 |
| Natural expressions | 125 |
| Awkward expressions | 55 |
| Context-dependent expressions | 51 |
| Accepted without reservations | 97 |
| Flagged for revision | 83 |
| Rejected expressions | 0 |

## Findings

No conflicting intent labels were identified in the reviewed sample.

However, several expressions were found to be formulaic or awkward for natural conversation.

Context dependency was identified in 28.33% of the reviewed expressions.

This finding supports the proposed separation between text-based intent classification and context-aware decision-making.

## Limitations

The semantic review was performed using an LLM rather than independent human annotators.

Only 15% of the dataset was reviewed.

Although all intent classes and linguistic styles were represented, only four of the twelve topics were included.

Annotation agreement should not be interpreted as model accuracy or proof of complete dataset correctness.

## Decision

The existing contrastive dataset will be retained for controlled experimentation.

A separate Natural Small Talk Dataset will be developed to investigate the effect of shorter and more natural conversational expressions.

## Next Stage

E09.07.6.4 — Preprocessing and Overlap Analysis

No ML model was loaded or trained during this stage.