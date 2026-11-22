# Findings

Numbers from `make train` (tables in `reports/tables/`).

## Data
- 50,000 reviews → **49,582 after removing 418 exact duplicates**. Without
  this step identical reviews could land in both train and test.
- Stratified split: 35,698 train / 3,967 validation / 9,917 test.

## Model comparison (validation set)
| Model | Accuracy | ROC AUC | Train time |
|---|---|---|---|
| Word + char TF-IDF, linear SVM | **91.6%** | 0.974 | 75 s |
| Word 1–2-gram TF-IDF, logistic regression | 91.1% | 0.973 | 24 s |
| NB-weighted logistic regression | 89.9% | 0.968 | 26 s |
| Bag of words, Naive Bayes | 83.2% | 0.905 | 7 s |

The best model was chosen on **validation**, refit on train + validation and
scored **once** on test:

- **Test accuracy 92.0%** (95% bootstrap CI 91.5–92.6%), F1 0.921, ROC AUC 0.975.
- Errors are balanced: 432 false positives vs 357 false negatives.

The SVM beats logistic regression by only 0.6 points and takes 3x longer to
train. If speed or interpretability mattered more, I would ship the logistic
regression.

## What the model learned
- Strongest negative terms: *worst, bad, awful, boring, the worst, poor*.
  Strongest positive: *great, excellent, amazing, perfect, wonderful*.
- **"7/10" is the 3rd most positive term.** The labels were made from star
  ratings, and many reviewers write their rating in the text. This is a mild
  form of label leakage: it helps on this dataset, but would not carry over
  to reviews without ratings.
- **Negation marking works**: *boring* has weight -10.6, but *NOT_boring*
  ("not boring") is +0.7. Likewise *great* +10.3 vs *NOT_great* +1.3.

## Where it fails
- **Mixed reviews**: reviews containing *but / however / although* are 91.4%
  accurate vs **94.2%** for reviews without them. A turn-around in the middle
  of the review is the main hard case for bag-of-n-gram models.
- Length matters less than expected: accuracy is 91.5–92.5% for anything
  under 800 words and 93.8% for very long reviews.
- The most confident mistakes include clear **label noise** (an enthusiastic
  review labelled negative) and sarcasm.

## Next steps
- Fine-tune a small transformer (DistilBERT) for the mixed-review cases.
  Published results put it around 93–94% on this dataset.
- Remove explicit ratings (`\d+/10`) from the text to measure the model
  without that shortcut.
