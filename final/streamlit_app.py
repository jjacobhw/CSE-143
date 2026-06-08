# Simple Streamlit demo for the SMS spam detector
# It loads the saved SMS model and predicts spam or ham for what the user pastes

import os

import joblib
import streamlit as st

from src.preprocessing import clean_text

# Path to the SMS model we trained with train_and_save_models.py
SMS_MODEL_PATH = os.path.join("models", "sms_spam_model.joblib")


# st.cache_resource keeps the model in memory so it loads only once,
# not every time the user clicks the button
@st.cache_resource
def load_model():
    return joblib.load(SMS_MODEL_PATH)


sms_model = load_model()

# Page content
st.title("SMS Spam Detector")
st.write(
    "This app predicts whether an SMS message is ham (not spam) or spam. "
    "Paste your message and click the button."
)

# Textbox for the SMS message
user_text = st.text_area("Paste your SMS message here")

# Button that runs the prediction
if st.button("Check message"):
    if user_text.strip() == "":
        st.warning("Please paste a message first.")
    else:
        # Clean the input the same way we cleaned the training data
        cleaned = clean_text(user_text)

        # The model returns a list, so we take the first prediction
        prediction = sms_model.predict([cleaned])[0]

        # Show the result clearly
        if prediction == "spam":
            st.error("Prediction: Spam")
        else:
            st.success("Prediction: Ham")
