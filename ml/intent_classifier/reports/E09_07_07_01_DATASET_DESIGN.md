# E09.07.7.1 — Natural Small Talk Dataset Design

## Objective

The recognition of natural conversational expressions will be improved through additional domain-specific training data.

Particular attention will be given to short utterances, linguistic diversity, and closely related conversational intents.

## Research Hypothesis

Increasing the diversity of natural conversational expressions, particularly short utterances, improves intent classification performance.

## Motivation

Previous experiments revealed several remaining limitations:

- Insufficient recognition of agreement and acknowledgment.
- Repeated linguistic structures in synthetic training data.
- Limited representation of short conversational expressions.
- Ambiguity between context-dependent intents.
- Excessive prediction of the STATEMENT class.

## Dataset Configuration

| Parameter | Value |
|---|---|
| Dataset name | Natural Small Talk V1 |
| Dataset type | Synthetic conversational data |
| Planned records | 600 |
| Intent classes | 12 |
| Records per class | 50 |
| Language | English |
| Purpose | Supervised training |
| Generation batches | 12 |

## Target Length Distribution

| Expression length | Records | Percentage |
|---|---:|---:|
| 1–3 words | 300 | 50% |
| 4–6 words | 180 | 30% |
| 7–10 words | 90 | 15% |
| 11 or more words | 30 | 5% |

The distribution will be treated as a design target rather than a proven optimal configuration.

## Conversational Styles

Natural expressions will be represented through casual, neutral, hesitant, emphatic, fragmented, and polite speech.

Rigid linguistic templates will be avoided.

## Dataset Schema

The following fields will be included:

- id
- text
- intent
- context
- scenario
- style
- context_required
- source

Context information will be retained as metadata and will not be used as a feature by the current text-only classifier.

## Quality Assurance

The following procedures will be performed:

1. Structural validation.
2. Class and length distribution analysis.
3. Lexical diversity analysis.
4. Repeated linguistic pattern analysis.
5. Full semantic and naturalness review.
6. Context dependence analysis.
7. Text preprocessing.
8. Exact text overlap analysis.
9. Training eligibility validation.

The existing evaluation datasets will not be used as sources for generating training expressions.

## Experimental Design

The eligible records will be added to the existing training dataset.

Frozen and refitted TF-IDF feature spaces will be investigated using three frequency weighting strategies.

Classifier and vectorizer parameters will be preserved.

## Evaluation

The resulting models will be compared using:

- Accuracy
- Macro F1
- Top-3 Accuracy
- Class-level Precision, Recall, and F1
- STATEMENT false positives
- Expression group performance
- Paired prediction analysis

## Limitations

Synthetic expressions may not fully represent spontaneous human speech.

Some conversational intents cannot be reliably distinguished without context.

LLM-assisted semantic review does not replace independent human annotation.

The existing installation benchmark will be used for development only.

An independent final evaluation will be performed after model selection.