import os

import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import clean_text, load_sms_data, prepare_dataset


def load_spam_corpus(data_path, pipeline):
    """Load all spam examples from the training set and vectorize them."""
    df = load_sms_data(data_path)
    df = prepare_dataset(df)
    spam_df = df[df["label"] == "spam"].reset_index(drop=True)

    vectorizer = pipeline.named_steps["tfidf"]
    spam_vectors = vectorizer.transform(spam_df["text"])

    return spam_df["text"].tolist(), spam_vectors, vectorizer


def retrieve_similar_spam(query_text, spam_texts, spam_vectors, vectorizer, k=5):
    """Return the K training spam examples most similar to query_text."""
    cleaned = clean_text(query_text)
    query_vec = vectorizer.transform([cleaned])

    sims = cosine_similarity(query_vec, spam_vectors).flatten()
    top_idx = np.argsort(sims)[::-1][:k]

    return [spam_texts[i] for i in top_idx]


def _stream(prompt):
    """Send a prompt via OpenRouter and yield text chunks."""
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    stream = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        max_tokens=1024,
        stream=True,
        messages=[{"role": "user", "content": prompt}],
    )
    for chunk in stream:
        text = chunk.choices[0].delta.content
        if text:
            yield text


def explain_spam_stream(message, similar_examples):
    """Yield streamed text chunks explaining why the message is spam."""
    examples_block = "\n".join(
        f"{i + 1}. {ex}" for i, ex in enumerate(similar_examples)
    )

    prompt = f"""You are a fraud-awareness expert. A machine learning spam detector has flagged the following message as spam.

FLAGGED MESSAGE:
"{message}"

SIMILAR KNOWN SPAM MESSAGES RETRIEVED FROM OUR DATABASE:
{examples_block}

Using the flagged message and the retrieved examples as context, provide:

**Why this message is spam**
Identify the specific red flags, psychological tactics, and patterns present in the flagged message (e.g., false urgency, prize claims, impersonation, suspicious links, requests for personal or financial information).

**How to protect yourself**
Give 3–4 concrete, actionable tips a person can follow to avoid falling for this type of fraudulent message in the future."""

    yield from _stream(prompt)


def explain_ham_stream(message, similar_spam_examples):
    """Yield streamed text chunks explaining why the message is legitimate."""
    examples_block = "\n".join(
        f"{i + 1}. {ex}" for i, ex in enumerate(similar_spam_examples)
    )

    prompt = f"""You are a fraud-awareness expert. A machine learning spam detector has classified the following message as legitimate (not spam).

MESSAGE:
"{message}"

FOR CONTEXT — SPAM MESSAGES WITH SIMILAR VOCABULARY FROM OUR DATABASE:
{examples_block}

Using the message and the spam examples as context, provide:

**Why this message appears legitimate**
Point out the specific features that distinguish it from spam (e.g., no urgency tactics, no prize claims, no suspicious links, no requests for personal information, natural conversational tone).

**What to still watch out for**
Give 2–3 brief reminders about how fraudsters sometimes disguise spam as legitimate messages, so the reader stays cautious."""

    yield from _stream(prompt)
