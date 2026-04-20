import csv
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from tokenization import get_tokenizer

DATA_DIR = Path(__file__).parent.parent / "data" / "cleaned data"


def load_split(split: str) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    with open(DATA_DIR / f"{split}.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row["text"])
            labels.append(int(row["label"]))
    return texts, labels


def build_vectorizer(
    vectorizer_type: str = "tfidf",
    tokenizer_method: str = "split",
    ngram_range: tuple = (1, 2),
    min_df: int = 3,
):
    tokenizer = get_tokenizer(tokenizer_method)

    if vectorizer_type == "tfidf":
        return TfidfVectorizer(
            tokenizer=tokenizer,
            token_pattern=None,
            lowercase=False,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=True,
        )
    elif vectorizer_type == "count":
        return CountVectorizer(
            tokenizer=tokenizer,
            token_pattern=None,
            lowercase=False,
            ngram_range=ngram_range,
            min_df=min_df,
        )
    else:
        raise ValueError("vectorizer_type must be 'tfidf' or 'count'")


def build_classifier(
    model_type: str = "logreg",
    C: float = 1.0,
    alpha: float = 1.0,
):
    if model_type == "logreg":
        return LogisticRegression(max_iter=1000, C=C)
    elif model_type == "svm":
        return LinearSVC(C=C, max_iter=2000)
    elif model_type == "nb":
        return MultinomialNB(alpha=alpha)
    else:
        raise ValueError("model_type must be 'logreg', 'svm', or 'nb'")


def build_model(
    model_type: str = "logreg",
    vectorizer_type: str = "tfidf",
    tokenizer_method: str = "split",
    ngram_range: tuple = (1, 2),
    min_df: int = 3,
    C: float = 1.0,
    alpha: float = 1.0,
):
    vectorizer = build_vectorizer(
        vectorizer_type=vectorizer_type,
        tokenizer_method=tokenizer_method,
        ngram_range=ngram_range,
        min_df=min_df,
    )
    clf = build_classifier(
        model_type=model_type,
        C=C,
        alpha=alpha,
    )
    return vectorizer, clf


def train(vectorizer, clf, texts: list[str], labels: list[int]):
    X = vectorizer.fit_transform(texts)
    clf.fit(X, labels)


def predict(vectorizer, clf, texts: list[str]):
    X = vectorizer.transform(texts)
    return clf.predict(X)