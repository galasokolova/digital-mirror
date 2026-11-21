# Digital Mirror — ML Research Report

**Goal:** Train a machine learning model to predict whether a political statement is true or false. The model will be used in an interactive installation to compare human trust with AI predictions.

## 1. Datasets explored

| Dataset | How I used it | Results |
|---|---|---|
| **EUvsDisinfo** | Collected false claims about Ukraine. | 14,497 records. Selected 730 relevant false claims. |
| **ClaimReview** | Searched for true and false claims about Russia–Ukraine relations from 2014 onwards. | From 281,496 records, selected 571 relevant claims: 457 false, 22 true, 92 other. |
| **LIAR** | Used political statements with verified labels to train a binary classifier. | Prepared 4,542 statements: 2,049 true and 2,493 false. |

**Main problem:** EUvsDisinfo and ClaimReview contained many false claims but very few true ones. LIAR provided a better balance for initial training.

## 2. Model training

I trained a **TF-IDF + Logistic Regression** model using LIAR.

TF-IDF converts text into numerical features. Logistic Regression uses these features to predict `TRUE` or `FALSE`.

| Metric | Simple baseline | Trained model |
|---|---:|---:|
| Accuracy | 60.88% | **66.90%** |
| Macro F1 | 0.3784 | **0.6635** |

The trained model performed better than a classifier that always predicts the most common class.

However, it still made 143 mistakes out of 432 validation statements.

## 3. Testing the model

I tested the model on statements about Russia and Ukraine that were not part of LIAR.

| Statement | AI prediction | Actual fact |
|---|---|---|
| Russia invaded Ukraine in February 2022. | TRUE (57.29%) | TRUE |
| Ukraine became a NATO member in 2015. | TRUE (52.81%) | FALSE |
| The EU has never imposed sanctions on Russia. | TRUE (54.54%) | FALSE |

The model sometimes predicted false statements as true.

I also tested opposite statements:

| Statement | P(TRUE) |
|---|---:|
| Crimea was annexed by Russia in 2014. | 55.16% |
| Crimea was **not** annexed by Russia in 2014. | 57.21% |

The model gave a higher truth score to the false version.

## 4. What I learned

- The model learned patterns in political language.
- It did not reliably understand negation or changes in factual meaning.
- Good results on LIAR did not guarantee good predictions about Russia and Ukraine.
- **Text alone is not enough for reliable fact-checking.**

## 5. Next steps

I plan to move from text-only classification to **evidence-based fact verification**.

Instead of analysing only a statement, the model will compare it with supporting information.

I plan to explore the **FEVER dataset**, which contains statements, evidence and three labels:

- `SUPPORTS` — evidence supports the statement.
- `REFUTES` — evidence contradicts the statement.
- `NOT ENOUGH INFO` — there is not enough evidence.

I will then test this approach on verified statements about Russia–Ukraine relations.

**Conclusion:** The first model works as a basic text classifier, but it cannot reliably check facts. The next version will focus on comparing claims with evidence, while keeping the original Digital Mirror concept of visualising differences between human trust and machine predictions.