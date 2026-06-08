# Trains the SMS and email spam models and saves them to the models/ folder

import os

import joblib

from src.preprocessing import (
    load_enron_data,
    load_sms_data,
    prepare_dataset,
)
from src.modeling import build_svm_pipeline

# Where the data lives and where we want to save the trained models
SMS_DATA_PATH = os.path.join("data", "sms_spam_collection.tsv")
EMAIL_DATA_PATH = os.path.join("data", "enron_spam_data.csv")
MODELS_DIR = "models"
SMS_MODEL_PATH = os.path.join(MODELS_DIR, "sms_spam_model.joblib")
EMAIL_MODEL_PATH = os.path.join(MODELS_DIR, "email_spam_model.joblib")


def train_sms_model():
    # Load and clean the SMS data, then train on all of it
    print("Loading SMS data...")
    df = load_sms_data(SMS_DATA_PATH)
    df = prepare_dataset(df)

    print("Training SMS model...")
    model = build_svm_pipeline()
    model.fit(df["text"], df["label"])

    joblib.dump(model, SMS_MODEL_PATH)
    print("Saved SMS model to", SMS_MODEL_PATH)


def train_email_model():
    # Load and clean the Enron email data, then train on all of it
    print("Loading email data...")
    df = load_enron_data(EMAIL_DATA_PATH)
    df = prepare_dataset(df)

    print("Training email model...")
    model = build_svm_pipeline()
    model.fit(df["text"], df["label"])

    joblib.dump(model, EMAIL_MODEL_PATH)
    print("Saved email model to", EMAIL_MODEL_PATH)


def main():
    # Make sure the models/ folder exists before we save anything
    os.makedirs(MODELS_DIR, exist_ok=True)

    train_sms_model()
    train_email_model()
    print("Done! Both models are trained and saved.")


if __name__ == "__main__":
    main()
