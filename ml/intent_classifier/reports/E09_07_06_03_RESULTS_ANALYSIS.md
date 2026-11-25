# E09.07.6.3 — Contrastive Dataset Validation

## Objective

The generated contrastive dataset was evaluated before model training. Structural integrity, lexical diversity, template repetition, topic distribution, linguistic styles, and context availability were examined.

## Results

| Metric | Result |
|---|---:|
| Total records | 1,200 |
| Intent classes | 12 |
| Linguistic styles | 5 |
| Contrastive groups | 300 |
| Topics | 12 |
| Structural checks passed | 20/20 |
| Duplicate normalized texts | 0 |
| Conflicting labels | 0 |
| Cramér's V | 0.012388 |
| Records with context | 800 |
| Semantic review sample | 180 |

## Main Findings

All structural checks were passed. Intent classes and linguistic styles were fully balanced.

A very weak association between topic and intent was observed. All intent classes were represented across all 12 topics.

However, substantial template repetition was identified.

The ACKNOWLEDGMENT class contained only 47 unique tokens across 886 total tokens, resulting in a type-token ratio of 0.0530.

Within several intent-style combinations, the same approximate template signature was identified in 17 out of 20 records.

In addition, 60 out of 240 short expressions contained more than five words.

## Identified Limitations

The absence of duplicate texts did not guarantee linguistic diversity.

Different sentences were frequently generated using the same grammatical construction with different topic-related words.

Exact-text frequency weighting would not directly address this type of repetition.

Semantic correctness and overlap with existing datasets have not yet been verified.

## Decision

The current dataset was not approved for model training.

A linguistic diversity improvement stage was introduced before preprocessing and training.

The existing dataset structure, class balance, topic distribution, and contrastive organization will be preserved where possible.

## Next Experiment

E09.07.6.3.1 — Linguistic Diversity Improvement

A revised dataset will be prepared with more varied grammatical constructions, natural conversational expressions, and improved representation of short responses.

The revised dataset will be validated using the same diagnostic procedure.

## Integrity

The source dataset remained unchanged.

No model was loaded or trained.

No evaluation data was loaded.