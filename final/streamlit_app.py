import os

import joblib
from dotenv import load_dotenv

load_dotenv()
import streamlit as st

from src.preprocessing import clean_text
from src.rag import explain_ham_stream, explain_spam_stream, load_spam_corpus, retrieve_similar_spam

SMS_MODEL_PATH = os.path.join("models", "sms_spam_model.joblib")
SMS_DATA_PATH = os.path.join("data", "sms_spam_collection.tsv")


@st.cache_resource
def load_model():
    return joblib.load(SMS_MODEL_PATH)


@st.cache_resource
def load_corpus(_pipeline):
    # underscore prefix tells Streamlit not to hash this arg
    return load_spam_corpus(SMS_DATA_PATH, _pipeline)


sms_model = load_model()
spam_texts, spam_vectors, vectorizer = load_corpus(sms_model)

st.title("SMS Spam Detector")
st.write(
    "This app predicts whether an SMS message is ham (not spam) or spam. "
    "If spam is detected, a RAG agent will explain the red flags and how to "
    "protect yourself."
)

user_text = st.text_area("Paste your SMS message here")

if st.button("Check message"):
    if user_text.strip() == "":
        st.warning("Please paste a message first.")
    else:
        cleaned = clean_text(user_text)
        prediction = sms_model.predict([cleaned])[0]

        if prediction == "spam":
            st.error("Prediction: Spam")

            similar = retrieve_similar_spam(
                user_text, spam_texts, spam_vectors, vectorizer, k=5
            )

            st.subheader("Why this is spam — and how to stay safe")
            if not os.environ.get("OPENROUTER_API_KEY"):
                st.warning(
                    "Set the OPENROUTER_API_KEY environment variable to enable "
                    "AI-powered explanations."
                )
            else:
                st.write_stream(explain_spam_stream(user_text, similar))
        else:
            st.success("Prediction: Ham")

            similar = retrieve_similar_spam(
                user_text, spam_texts, spam_vectors, vectorizer, k=5
            )

            st.subheader("Why this message looks legitimate")
            if not os.environ.get("OPENROUTER_API_KEY"):
                st.warning(
                    "Set the OPENROUTER_API_KEY environment variable to enable "
                    "AI-powered explanations."
                )
            else:
                st.write_stream(explain_ham_stream(user_text, similar))
