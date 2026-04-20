import nltk
from nltk.tokenize import ToktokTokenizer, word_tokenize, WordPunctTokenizer

# Create tokenizer objects once so we can reuse them
toktok = ToktokTokenizer()
wordpunct = WordPunctTokenizer()

# basic split tokenization approach
def split_tokenize(text):
    return text.split()

# NLTK's toktok tokenizer 
def toktok_tokenize(text):
    return toktok.tokenize(text)

# NLTK's word_tokenizer
def nltk_word_tokenize_text(text):
    return word_tokenize(text)

# NLTK wordpunct_tokenizer
def wordpunct_tokenize_text(text):
    return wordpunct.tokenize(text)


# function for getting the tokenization method
def get_tokenizer(method):
    if method == "split":
        return split_tokenize
    elif method == "toktok":
        return toktok_tokenize
    elif method == "word_tokenize":
        return nltk_word_tokenize_text
    elif method == "wordpunct":
        return wordpunct_tokenize_text
    else:
        raise ValueError(
            f"Unknown tokenization method: {method}. "
            "Choose from: split, toktok, word_tokenize, wordpunct"
        )


# tokenizing single string
def tokenize_text(text, method="split"):
    tokenizer = get_tokenizer(method)
    return tokenizer(text)

# tokenize list of strings
def tokenize_texts(texts, method="split"):
    tokenizer = get_tokenizer(method)
    return [tokenizer(text) for text in texts]

# takes one piece of text, returning the tokens created for each different 
# tokenization method 
def compare_tokenizers_on_text(text):
    return {
        "split": split_tokenize(text),
        "toktok": toktok_tokenize(text),
        "word_tokenize": nltk_word_tokenize_text(text),
        "wordpunct": wordpunct_tokenize_text(text),
    }