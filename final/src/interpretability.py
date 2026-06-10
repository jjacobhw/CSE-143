# Simple interpretability helpers for the TF-IDF + Linear SVM model.
# Since the SVM is linear, we can just look at its learned weights:
# positive weights push toward spam, negative weights push toward non-spam.

import pandas as pd

# Column order for the single-prediction explanation table
EXPLAIN_COLUMNS = ["feature", "tfidf", "weight", "contribution", "direction"]


# Returns the strongest spam and non-spam features the model learned
def get_top_features(model, top_n=20):
    # get the words/phrases from TF-IDF and the matching SVM weights
    feature_names = model.named_steps["tfidf"].get_feature_names_out()
    weights = model.named_steps["clf"].coef_[0]

    df = pd.DataFrame({"feature": feature_names, "weight": weights})

    # biggest positive weights = strongest spam signals
    spam_features = (
        df.sort_values("weight", ascending=False).head(top_n).reset_index(drop=True)
    )

    # most negative weights = strongest non-spam signals
    nonspam_features = (
        df.sort_values("weight", ascending=True).head(top_n).reset_index(drop=True)
    )

    return spam_features, nonspam_features


# Explains which features in one message pushed the model toward spam or non-spam
def explain_single_prediction(model, text, top_n=10):
    vectorizer = model.named_steps["tfidf"]
    weights = model.named_steps["clf"].coef_[0]

    # turn the message into TF-IDF values
    vec = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()

    # only look at the features that actually appear in this message
    rows = []
    for idx, tfidf_value in zip(vec.indices, vec.data):
        contribution = tfidf_value * weights[idx]
        rows.append(
            {
                "feature": feature_names[idx],
                "tfidf": tfidf_value,
                "weight": weights[idx],
                "contribution": contribution,
                "direction": "spam" if contribution > 0 else "non-spam",
            }
        )

    # if the message has no known features, return an empty table
    if not rows:
        return pd.DataFrame(columns=EXPLAIN_COLUMNS)

    df = pd.DataFrame(rows, columns=EXPLAIN_COLUMNS)

    # sort by absolute contribution so the strongest signals come first
    df = df.reindex(df["contribution"].abs().sort_values(ascending=False).index)

    return df.head(top_n).reset_index(drop=True)
