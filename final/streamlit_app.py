# Simple Streamlit demo for the SMS spam detector.
# It loads the saved SMS model and predicts spam or non-spam for what the user pastes.

import os

import joblib
import streamlit as st

from src.preprocessing import clean_text
from src.interpretability import explain_single_prediction

SMS_MODEL_PATH = os.path.join("models", "sms_spam_model.joblib")


# st.cache_resource keeps the model in memory so it loads only once,
# not every time the user clicks the button.
@st.cache_resource
def load_model():
    return joblib.load(SMS_MODEL_PATH)


sms_model = load_model()

st.title("SMS Spam Detector")
st.write(
    "This app predicts whether an SMS message is non-spam or spam, "
    "and shows which words pushed the model toward its decision."
)

user_text = st.text_area("Paste your SMS message here")

if st.button("Check message"):
    if user_text.strip() == "":
        st.warning("Please paste a message first.")
    else:
        # Clean the input the same way we cleaned the training data
        cleaned = clean_text(user_text)
        prediction = sms_model.predict([cleaned])[0]

        # the model labels are "ham"/"spam", but we display "Non-spam"/"Spam"
        if prediction == "spam":
            st.error("Prediction: Spam")
        else:
            st.success("Prediction: Non-spam")

        # Top model signals come straight from the TF-IDF + Linear SVM model:
        # for each feature in the message, contribution = tfidf value * SVM weight
        st.subheader("Top model signals")
        st.caption(
            "These signals come from the TF-IDF + Linear SVM model weights. "
            "Positive contributions push toward spam, negative toward non-spam."
        )
        signals = explain_single_prediction(sms_model, cleaned, top_n=8)
        if signals.empty:
            st.write("No strong known TF-IDF features were found for this message.")
        else:
            st.dataframe(signals)
