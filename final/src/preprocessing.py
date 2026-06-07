# Functions for loading the data and cleaning the text
# These keep the messy file and cleaning details out of the notebook

import re

import pandas as pd

# Simple patterns used by clean_text.
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
EMAIL_PATTERN = re.compile(r"\S+@\S+")
NUMBER_PATTERN = re.compile(r"\d+")
WHITESPACE_PATTERN = re.compile(r"\s+")


# Lowercases the text and swaps URLs, emails, and numbers for simple tokens
# This way the model learns the pattern instead of one exact link or number
def clean_text(text):
    # Handle missing values and make sure we have a string.
    if pd.isna(text):
        return ""
    text = str(text).lower()

    # replacing these different potential patterns with tokens
    text = URL_PATTERN.sub(" URL ", text)
    text = EMAIL_PATTERN.sub(" EMAIL ", text)
    text = NUMBER_PATTERN.sub(" NUMBER ", text)

    # Turn repeated spaces/newlines into a single space to make clean sentences
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


# Loads the SMS data into a text/label table
def load_sms_data(path):
    # standardizing the labels for consistency
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"])

    # Make labels lowercase "ham"/"spam"
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    return df[["text", "label"]]


# Loads the Enron emails into a text/label table
# We glue the subject and message together since spam clues can be in either
def load_enron_data(path):
    df = pd.read_csv(path)

    # handling plan subjects and messages
    subject = df["Subject"].fillna("")
    message = df["Message"].fillna("")

    # combining the subject the subject and message of the email into one text
    df["text"] = subject.astype(str) + " " + message.astype(str)

    # Make labels lowercase "ham"/"spam"
    df["label"] = df["Spam/Ham"].astype(str).str.strip().str.lower()
    return df[["text", "label"]]


# Cleans a text/label table so it is ready for the model
def prepare_dataset(df):
    df = df.copy()

    # Drop rows missing text or label, then clean the text
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].apply(clean_text)

    # Drop rows that became empty after cleaning.
    df = df[df["text"].str.strip() != ""]

    return df[["text", "label"]].reset_index(drop=True)
