# Spam Detection with TF-IDF and Linear SVM

My CSE 143 (NLP) final project. It builds a text spam detector and checks how it
does on two kinds of data: SMS text messages and emails.

## What the project does

We turn the raw text into TF-IDF features and train a Linear SVM (`LinearSVC`)
to label each message as `ham` (not spam) or `spam`. We run the same steps on
both datasets and compare the results. We also try two simpler models (Naive
Bayes and Logistic Regression) to see if the SVM is actually better.

## Datasets

The data files go in the `data/` folder:

- `data/sms_spam_collection.tsv` — SMS Spam Collection. Tab-separated, no header.
  Column 1 is the label (`ham`/`spam`), column 2 is the message.
- `data/enron_spam_data.csv` — Enron emails. We join the `Subject` and `Message`
  columns into one text field and use `Spam/Ham` as the label.

Note: the raw data files are not committed. Put them in the `data/` folder before
running the notebook.

## Methods

- **Features:** `TfidfVectorizer` (words, with an option for character n-grams).
- **Main model:** `LinearSVC` with `class_weight="balanced"`.
- **No data leakage:** everything is in a scikit-learn `Pipeline`, so TF-IDF is
  only fit on the training data.
- **Split:** `train_test_split` with `stratify=y` to keep the class balance.
- **Cross-validation:** `StratifiedKFold` with `shuffle=True` (5 folds).

## Metrics

Accuracy, precision, recall, F1, and F2 (F2 weights recall more), plus a
confusion matrix. `spam` is the positive class.

## N-Gram Experiment

We compared three SVM setups on the SMS data: word TF-IDF, character TF-IDF,
and a combined word + character TF-IDF model. Character n-grams look at
patterns inside words, which can matter when spam uses weird spelling like
"fr33" or "w1nner". The results are shown in the notebook and saved in
`results/ngram_comparison.csv`.

## Model Interpretability

Because we use a Linear SVM, we can inspect the feature weights it learned.
Positive weights are associated with `spam`, and negative weights are
associated with `non-spam`. The notebook shows the strongest features for each
dataset, and the Streamlit app shows the top model signals for each message you
check (see `src/interpretability.py`).

## Project structure

```
final/
  data/        # the two dataset files
  src/         # helper functions (preprocessing, modeling, evaluation)
  results/     # saved confusion matrices and the metrics summary CSV
  spam_detection_final.ipynb   # the experiments and charts
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

3. Run all the cells from top to bottom.
