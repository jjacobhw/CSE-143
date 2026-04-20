import csv
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from tokenization import get_tokenizer

DATA_DIR = Path(__file__).parent.parent / "data" / "cleaned data"


def load_split(split: str) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    with open(DATA_DIR / f"{split}.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row["text"])
            labels.append(int(row["label"]))
    return texts, labels


def build_model(
    tokenizer_method: str = "split",
    ngram_range: tuple = (1, 2),
    min_df: int = 3,
    C: float = 1.0,
) -> tuple[TfidfVectorizer, LogisticRegression]:
    vectorizer = TfidfVectorizer(
        tokenizer=get_tokenizer(tokenizer_method),
        token_pattern=None,
        lowercase=False,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True,
    )
    clf = LogisticRegression(max_iter=1000, C=C)
    return vectorizer, clf


def train(vectorizer: TfidfVectorizer, clf: LogisticRegression, texts: list[str], labels: list[int]):
    X = vectorizer.fit_transform(texts)
    clf.fit(X, labels)


def predict(vectorizer: TfidfVectorizer, clf: LogisticRegression, texts: list[str]) -> list[int]:
    X = vectorizer.transform(texts)
    return clf.predict(X)