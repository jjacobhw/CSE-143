# Builds the models and the helper functions for training and testing.
# Everything is wrapped in a scikit-learn Pipeline so TF-IDF only learns from
# the training data. This avoids leaking info from the test set

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


# Our main model: TF-IDF turns text into numbers, then LinearSVC classifies it
# analyzer="word" uses words, analyzer="char" uses character n-grams
def build_svm_pipeline(ngram_range=(1, 2), analyzer="word"):
    # Stop words only make sense for words, not characters.
    stop_words = "english" if analyzer == "word" else None

    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        stop_words=stop_words,
    )
    # class_weight="balanced" helps when one class is rarer than the other
    classifier = LinearSVC(class_weight="balanced")

    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


# Baseline model: TF-IDF + Naive Bayes
def build_nb_pipeline(ngram_range=(1, 2)):
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, stop_words="english")
    return Pipeline([("tfidf", vectorizer), ("clf", MultinomialNB())])


# Baseline model: TF-IDF + Logistic Regression
def build_logreg_pipeline(ngram_range=(1, 2)):
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, stop_words="english")
    classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


# Splits the data, trains the model, and predicts on the test set
# stratify=y keeps the same ham/spam balance in train and test
def train_test_model(df, model, test_size=0.2, random_state=42):
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


# Runs cross-validation and returns the mean and std for each metric
# StratifiedKFold with shuffle=True keeps the ham/spam balance in each fold and
# shuffles first so the folds are not based on the original file order
def cross_validate_model(df, model, cv=5):
    X = df["text"]
    y = df["label"]

    # spam is the positive class for precision, recall, F1, and F2
    # F2 weights recall more than precision (missing spam is worse than a false alarm)
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(_safe_score, metric="precision"),
        "recall": make_scorer(_safe_score, metric="recall"),
        "f1": make_scorer(fbeta_score, beta=1, pos_label="spam", zero_division=0),
        "f2": make_scorer(fbeta_score, beta=2, pos_label="spam", zero_division=0),
    }

    cv_strategy = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=42,
    )

    cv_results = cross_validate(model, X, y, cv=cv_strategy, scoring=scoring)

    # Get the mean and std for each metric across the folds
    summary = {}
    for metric in scoring:
        scores = cv_results[f"test_{metric}"]
        summary[f"{metric}_mean"] = float(np.mean(scores))
        summary[f"{metric}_std"] = float(np.std(scores))
    return summary


# Small helper so precision and recall share one function (spam is positive)
def _safe_score(y_true, y_pred, metric):
    from sklearn.metrics import precision_score, recall_score

    if metric == "precision":
        return precision_score(y_true, y_pred, pos_label="spam", zero_division=0)
    return recall_score(y_true, y_pred, pos_label="spam", zero_division=0)
