# Dataset Preparation — Digital Mirror

## 1. Project Goal

The goal of this project is to teach a machine to recognize human reactions during a conversation.

Two separate ML models are proposed:

- **Communicative Intent Model** — identifies what a person is trying to communicate.
- **Emotional Reaction Model** — identifies emotions expressed in a person's words.

The model predictions will be used to select appropriate machine responses during the interaction.

## 2. Datasets

Two existing datasets were selected instead of generating artificial training data.

| Dataset | Content | Purpose |
|---|---|---|
| Switchboard | Human telephone conversations | Intent classification |
| GoEmotions | Reddit comments with emotion labels | Emotion classification |

### Switchboard

- 223,606 original records.
- 1,155 conversations.
- 303 original label variants.

Source: https://compprag.christopherpotts.net/swda.html

### GoEmotions

- 54,263 records in the downloaded dataset.
- 28 categories (27 emotions and neutral).
- Multiple emotion labels can be assigned to a single text.

Source: https://github.com/google-research/google-research/tree/master/goemotions

## 3. Data Preparation Experiments

| Experiment | Task | Result |
|---|---|---|
| E01 | Switchboard Exploration | Dataset structure and labels were examined |
| E02 | Label Semantics | Useful dialogue categories were identified |
| E03 | DAMSL Normalization | 303 label variants were reduced to 43 categories |
| E04 | Utterance Reconstruction | 18,691 continuation segments were joined |
| E05 | Text Preprocessing | Transcription symbols were removed and empty texts were identified |
| E06 | Label Ambiguity | Identical texts with different labels were identified |
| E07 | GoEmotions Exploration | Emotion labels and multilabel structure were examined |
| E08a | Intent Dataset Preparation | 12 custom categories were created |
| E08a.2 | Text Representation Comparison | The effect of capitalization and punctuation was examined |
| E08a.3 | Dataset Split | Training, validation and test sets were created |
| E08b | Emotion Dataset Preparation | 10 emotional categories were selected |

## 4. Text Preprocessing

Unnecessary transcription symbols, technical annotations and extra spaces were removed.

The following elements were preserved:

- Short answers.
- Negations.
- English contractions.
- Repeated words.
- Natural speech patterns.

Examples:

| Original text | Cleaned text | Model text |
|---|---|---|
| Right <laughter>. / | Right. | right |
| Yeah, that's right. / | Yeah, that's right. | yeah that's right |
| I don't know. / | I don't know. | i don't know |

The original texts were preserved for future experiments.

## 5. Communicative Intent Dataset

After cleaning and category selection, 170,706 records were retained.

| Intent | Records |
|---|---:|
| STATEMENT | 101,928 |
| ACKNOWLEDGMENT | 38,378 |
| AGREEMENT | 11,157 |
| QUESTION | 8,913 |
| AFFIRMATIVE_ANSWER | 3,891 |
| GOODBYE | 2,584 |
| NEGATIVE_ANSWER | 1,681 |
| UNCERTAINTY | 1,227 |
| DISAGREEMENT | 345 |
| CLARIFICATION | 299 |
| GREETING | 225 |
| GRATITUDE | 78 |
| **Total** | **170,706** |

One intent label is assigned to each text.

This task is defined as **multiclass classification**.

## 6. Emotional Reaction Dataset

Ten categories were selected from GoEmotions.

| Emotion | Records |
|---|---:|
| NEUTRAL | 17,772 |
| APPROVAL | 3,687 |
| GRATITUDE | 3,372 |
| ANNOYANCE | 3,093 |
| AMUSEMENT | 2,895 |
| CURIOSITY | 2,723 |
| DISAPPROVAL | 2,581 |
| CONFUSION | 1,673 |
| REALIZATION | 1,382 |
| SURPRISE | 1,330 |

Multiple emotion labels can be assigned to a single text.

This task is defined as **multilabel classification**.

| Label distribution | Records |
|---|---:|
| No selected emotions | 16,556 |
| One selected emotion | 34,985 |
| Multiple selected emotions | 2,722 |
| **Total** | **54,263** |

All records were preserved, including texts without selected emotions.

These texts were not automatically classified as neutral. They were kept as negative examples for the selected categories.

## 7. Dataset Split

Both datasets were divided into training, validation and test sets.

| Dataset | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Intent | 135,827 | 16,913 | 17,966 | 170,706 |
| Emotion | 43,410 | 5,426 | 5,427 | 54,263 |

Switchboard was split by complete conversations to prevent parts of the same conversation from being included in different sets.

The original GoEmotions split was preserved.

The presence of all selected categories was confirmed in every set.

## 8. Research Findings

### Class Imbalance

A significant difference between category sizes was identified.

For example:

- STATEMENT: 101,928 examples.
- GRATITUDE: 78 examples.

Because of this imbalance, accuracy alone cannot be used to evaluate model performance.

Macro F1 was selected as the main evaluation metric.

### Text Ambiguity

Identical texts with different intent labels were identified.

For example, 1,835 occurrences of the word "yes" were found in the normalized Switchboard dataset.

| Intent | Percentage |
|---|---:|
| AGREEMENT | 35.97% |
| ACKNOWLEDGMENT | 34.66% |
| AFFIRMATIVE_ANSWER | 28.39% |
| Other intents | 0.98% |

The importance of conversation context was highlighted by these results.

### Capitalization and Punctuation

Three text representations were compared.

| Representation | Ambiguous records | Empirical accuracy ceiling |
|---|---:|---:|
| Original cleaned text | 32.05% | 92.97% |
| Lowercase | 32.35% | 92.85% |
| Lowercase without most punctuation | 33.29% | 92.52% |

Only a small difference in the calculated accuracy ceiling was observed after punctuation was removed.

Lowercase text without most punctuation was selected as the first training configuration because speech recognition will be used in the installation.

English contractions were preserved.

These values were calculated from the dataset. They were not obtained through model training.

## 9. Known Limitations

The following limitations were identified:

- Some categories are represented by very few examples.
- Different labels can be assigned to identical texts.
- Identical texts were found in different dataset splits.
- GoEmotions was collected from written comments rather than spoken conversations.
- Actual internal human feelings cannot be determined from emotion labels alone.
- Two GoEmotions training texts became empty after normalization. These records will be excluded during training.
- The selected custom categories have not yet been validated in real interaction scenarios.

## 10. Planned Architecture

```text
Human Speech
      |
      v
Speech-to-Text
      |
      v
+-------------------+-------------------+
| Communicative     | Emotional         |
| Intent Model      | Reaction Model    |
+-------------------+-------------------+
      |
      v
Java Decision Engine
      +
Conversation Context
      |
      v
Machine Response
```

Human speech will be converted into text.

The text will be analyzed by two independently trained ML models.

Model predictions and conversation context will be processed by the Java Decision Engine.

An appropriate machine response will then be selected.

## 11. Project Progress

### Completed

- [x] Both datasets were explored.
- [x] Texts were cleaned and prepared.
- [x] Custom categories were created.
- [x] Class imbalance was analyzed.
- [x] Text ambiguity was analyzed.
- [x] Training, validation and test sets were prepared.

### Planned

- [ ] Communicative Intent Model will be trained.
- [ ] Emotional Reaction Model will be trained.
- [ ] Model performance will be evaluated.
- [ ] Models will be tested using real interaction scenarios.
- [ ] Models will be integrated into Digital Mirror.

## 12. Project Structure

```text
intent_classifier/
├── data/
│   ├── raw/
│   │   ├── switchboard/
│   │   └── goemotions/
│   └── processed/
│       ├── intent_v1/
│       │   ├── train.csv
│       │   ├── validation.csv
│       │   └── test.csv
│       └── emotion_v1/
│           ├── train.csv
│           ├── validation.csv
│           └── test.csv
├── src/
│   ├── preprocessing/
│   ├── taxonomy/
│   └── training/
├── models/
├── experiments/
├── reports/
└── README.md
```

Detailed experiment results are stored in the `reports/` directory.

## Current Status

Dataset preparation has been completed.

Model training and evaluation will be performed in the next stage.
