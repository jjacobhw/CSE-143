"""Data loading and text cleaning helpers for the spam detection project.

These functions keep the messy parts (file formats, label names, cleaning)
in one place so the notebook can stay short and easy to read.
"""

import re

import pandas as pd

# Simple regex patterns used by clean_text.
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
EMAIL_PATTERN = re.compile(r"\S+@\S+")
NUMBER_PATTERN = re.compile(r"\d+")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase text and replace URLs, emails, and numbers with tokens.

    Turning specific URLs/emails/numbers into generic tokens helps the model
    learn the *pattern* (e.g. "spam often has links") instead of memorizing
    one exact link.
    """
    # Handle missing values (NaN) and make sure we are working with a string.
    if pd.isna(text):
        return ""
    text = str(text).lower()

    # Order matters: emails contain "@" so handle them before numbers.
    text = URL_PATTERN.sub(" URL ", text)
    text = EMAIL_PATTERN.sub(" EMAIL ", text)
    text = NUMBER_PATTERN.sub(" NUMBER ", text)

    # Collapse repeated spaces/newlines into a single space.
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def load_sms_data(path: str) -> pd.DataFrame:
    """Load the SMS spam collection TSV into a (text, label) dataframe."""
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"])

    # Standardize: keep labels as lowercase strings "ham"/"spam".
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    return df[["text", "label"]]


def load_enron_data(path: str) -> pd.DataFrame:
    """Load the Enron email CSV into a (text, label) dataframe.

    The email text is the subject and message glued together, since spam
    clues can live in either part.
    """
    df = pd.read_csv(path)

    # Combine subject + message into a single text field (fill blanks first).
    subject = df["Subject"].fillna("")
    message = df["Message"].fillna("")
    df["text"] = subject.astype(str) + " " + message.astype(str)

    # Standardize labels to lowercase "ham"/"spam".
    df["label"] = df["Spam/Ham"].astype(str).str.strip().str.lower()
    return df[["text", "label"]]


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a (text, label) dataframe so it is ready for modeling."""
    df = df.copy()

    # Drop rows missing either field, then clean the text.
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].apply(clean_text)

    # Drop rows that became empty after cleaning (nothing for the model to use).
    df = df[df["text"].str.strip() != ""]

    return df[["text", "label"]].reset_index(drop=True)
