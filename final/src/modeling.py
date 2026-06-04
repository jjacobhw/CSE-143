"""Model pipelines and train/evaluate helpers.

Everything uses a scikit-learn Pipeline so the TF-IDF vectorizer is fit only
on the training data. This avoids data leakage (the test set never influences
the vocabulary or IDF weights).
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_svm_pipeline(ngram_range=(1, 2), analyzer="word"):
    """TF-IDF + Linear SVM. This is our main model.

    - analyzer="word": normal words/word-pairs (uses English stop words).
    - analyzer="char": character n-grams (useful for tricky/short text).
    """
    # Stop words only make sense for word-level features.
    stop_words = "english" if analyzer == "word" else None

    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        stop_words=stop_words,
    )
    # class_weight="balanced" helps when one class is rarer than the other.
    classifier = LinearSVC(class_weight="balanced")

    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


def build_nb_pipeline(ngram_range=(1, 2)):
    """Simple baseline: TF-IDF + Multinomial Naive Bayes."""
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, stop_words="english")
    return Pipeline([("tfidf", vectorizer), ("clf", MultinomialNB())])


def build_logreg_pipeline(ngram_range=(1, 2)):
    """Simple baseline: TF-IDF + Logistic Regression."""
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, stop_words="english")
    classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


def train_test_model(df, model, test_size=0.2, random_state=42):
    """Split the data, train the model, and predict on the test set.

    stratify=y keeps the same ham/spam ratio in train and test, which matters
    for imbalanced data. random_state=42 makes the split reproducible.
    """
    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, X_train, X_test, y_train, y_test, y_pred


def cross_validate_model(df, model, cv=5):
    """Run k-fold cross-validation and report mean/std for each metric.

    We treat "spam" as the positive class. F2 weights recall more than
    precision, because missing spam (a false negative) is usually worse than
    a false alarm in this setting.
    """
    X = df["text"]
    y = df["label"]

    # Custom scorers that know "spam" is the positive label.
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(_safe_score, metric="precision"),
        "recall": make_scorer(_safe_score, metric="recall"),
        "f1": make_scorer(fbeta_score, beta=1, pos_label="spam", zero_division=0),
        "f2": make_scorer(fbeta_score, beta=2, pos_label="spam", zero_division=0),
    }

    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring)

    # Summarize each metric as mean and standard deviation across folds.
    summary = {}
    for metric in scoring:
        scores = cv_results[f"test_{metric}"]
        summary[f"{metric}_mean"] = float(np.mean(scores))
        summary[f"{metric}_std"] = float(np.std(scores))
    return summary


def _safe_score(y_true, y_pred, metric):
    """Small helper so precision/recall scorers share one definition."""
    from sklearn.metrics import precision_score, recall_score

    if metric == "precision":
        return precision_score(y_true, y_pred, pos_label="spam", zero_division=0)
    return recall_score(y_true, y_pred, pos_label="spam", zero_division=0)
