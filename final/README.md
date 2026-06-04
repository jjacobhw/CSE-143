# Spam Detection with TF-IDF and Linear SVM

A CSE 143 (NLP) final project. We build a text spam detector and compare how
it performs on two different domains: **SMS text messages** and **emails**.

## What the project does

We turn raw text into TF-IDF features and train a **Linear Support Vector
Machine (LinearSVC)** to classify each message as `ham` (not spam) or `spam`.
We run the same pipeline on both datasets and compare the results. We also try
two simple baselines (Naive Bayes and Logistic Regression) for comparison.

## Datasets

The data files live in the `data/` folder:

- `data/sms_spam_collection.tsv` — SMS Spam Collection. Tab-separated, no
  header. Column 1 is the label (`ham`/`spam`), column 2 is the message text.
- `data/enron_spam_data.csv` — Enron email dataset. We combine the `Subject`
  and `Message` columns into one text field and use `Spam/Ham` as the label.

> Note: the raw data files are not committed. Place them in the `data/` folder
> before running the notebook.

## Methods

- **Features:** `TfidfVectorizer` (word and optional character n-grams).
- **Main model:** `LinearSVC` with `class_weight="balanced"`.
- **No data leakage:** everything is wrapped in a scikit-learn `Pipeline`, so
  TF-IDF is fit only on the training split.
- **Split:** `train_test_split` with `stratify=y` and `random_state=42`.

## Metrics

Accuracy, precision, recall, F1, and **F2** (weights recall more heavily),
plus a confusion matrix and 5-fold cross-validation.

## Project structure

```
final/
  data/        # the two dataset files
  src/         # reusable functions (preprocessing, modeling, evaluation)
  results/     # saved confusion matrices and metrics summary CSV
  spam_detection_final.ipynb   # the full story: experiments + charts
  requirements.txt
  README.md
```

## How to run

1. Install the requirements:

   ```bash
   pip install -r requirements.txt
   ```

2. Open the notebook:

   ```bash
   jupyter notebook spam_detection_final.ipynb
   ```

3. Run all cells from top to bottom.
