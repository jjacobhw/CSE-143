import csv
from pathlib import Path

import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

DATA_DIR = Path(__file__).parent.parent / "data" / "cleaned data"


def load_split(split: str) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    with open(DATA_DIR / f"{split}.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row["text"])
            labels.append(int(row["label"]))
    return texts, labels


def tokenize(text: str) -> str:
    return " ".join(nltk.word_tokenize(text))


def build_model(
    ngram_range: tuple = (1, 2),
    min_df: int = 3,
    alpha: float = 1.0,
) -> tuple[CountVectorizer, MultinomialNB]:
    vectorizer = CountVectorizer(ngram_range=ngram_range, min_df=min_df)
    clf = MultinomialNB(alpha=alpha)
    return vectorizer, clf


def train(vectorizer: CountVectorizer, clf: MultinomialNB, texts: list[str], labels: list[int]):
    nltk.download("punkt_tab", quiet=True)
    tokenized = [tokenize(t) for t in texts]
    X = vectorizer.fit_transform(tokenized)
    clf.fit(X, labels)


def predict(vectorizer: CountVectorizer, clf: MultinomialNB, texts: list[str]) -> list[int]:
    tokenized = [tokenize(t) for t in texts]
    X = vectorizer.transform(tokenized)
    return clf.predict(X)
