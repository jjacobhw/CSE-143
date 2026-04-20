# Homework 1

CSE 143: Intro to Natural Language Processing

University of California, Santa Cruz

*This assignment is to be done with your group, in Python 3, with code and
prose that you write yourself, rather than the output of a robot. You need to be
able to explain what you did, and why.*

## Introduction

In this assignment, you'll be doing some basic text classification, developing a
simple sentiment analysis system.

You're provided with a dataset (or rather, a tool for downloading it
automatically), and it's your job to figure out good ways to tokenize the given
data and turn it into features, and then run a classification algorithm and
report your classification results on the test set.

For this homework, we'll use the NLTK library for its tokenizers, and
scikit-learn for basic machine learning functionality. You'll probably want to
install them in a virtual environment or `uv`.

## Dataset

For this assignment, you're working with Stanford's Large Movie Review Dataset
(described here: https://ai.stanford.edu/~amaas/data/sentiment/ ), and it is
your job to tokenize and turn it into sensible vectors that you can use for
machine learning.

You can download the dataset with the following commands, after you've copied
the `download_and_split_data.py` file into your `hw1` directory:

```
  $ mkdir data
  $ python3 download_and_split_data.py
```

Once you've run these commands, you'll have a nice pre-split dataset, separated
out into `train.csv`, `dev.csv`, and `test.csv`, which are the training,
development and test sets respectively. You should avoid using the test set
until you are satisfied with your results on the development set, like we
discussed in class!

Each of these CSV files contains (after the CSV header) a single movie review
per line, with a label of 1 for a positive review or 0 for a negative review.
You should load up each line of the datasets with the standard Python CSV reader
(`import csv`) or `pandas`, then tokenize the texts and turn them into vectors
suitable for machine learning. These next steps are described in the following
sections.

## Tokenization and preprocessing

Taking a peek at the data, you may notice that there are HTML tags (specifically
`<br />`) in the text -- you should probably take those out as a preprocessing
step! Also, do you think that case matters for this task? What would happen if
you lowercase everything as a preprocessing step, or if you don't?

Next up, we'll experiment with a few different tokenizers, and see how they feel
qualitatively, or whether they end up changing your classification results
downstream.

Tokenizers to try:

  * Simply splitting the string with `"".split()`
  * [NLTK's TokTok tokenizer](https://www.nltk.org/api/nltk.tokenize.toktok.html#nltk.tokenize.toktok.ToktokTokenizer)
  * NLTK's default `word_tokenize` method, `nltk.word_tokenize` (may require
    running `nltk.download()` first)
  * Perhaps some other tokenization strategy from NLTK that seems interesting?

Randomly sample five reviews from the training set and run them through the
tokenization strategies that you've chosen -- do you like the tokenization
results, qualitatively? Do you think this will be helpful for making feature
vectors for machine learning?

## Feature extraction

Now that we've tokenized our text, it's time to produce vectors that we can use
to train classifiers!

Try at least a few different feature extraction strategies here -- as a
baseline, you ought to turn the text into a basic *bag of words* model like we
discussed in class. One straightforward way to do this is with
scikit-learn's
[`DictVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html)
(there are good examples of how to do this in the scikit-learn documentation) --
you'll take a pass over your training set and turn it into a list of
dictionaries mapping from *word types* to the *counts* of each of the words in
the dictionary.

Notably in this case you'll want to *fit* the vectorizer first, so that you can
use it again to *transform* later datasets (the dev and test sets) -- you need
to do preprocessing and feature extraction the same way for the training set
*and* for the dev and test sets.

Other things to try:

  * If you want to get fancier, try out the scikit-learn `CountVectorizer`,
    which can add counts of n-grams as features rather than just counts of
    individual words (which settings will you use?), or
    even the `TfidfVectorizer`, the latter of which will scale up the counts of
    words that are more unusual and informative -- we'll talk more about TF-IDF
    as a strategy later in class.
  * You should also try crafting your own features! You could write down a list
    of words or sequences that you think convey strong sentiment and use these
    as features.

## Training classifiers

Once you can turn these short documents into vectors, it's time to get training
some machine learning classifiers!

Use scikit-learn to train some simple classifiers (on the training set): you
should try, at least:

  * [logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
  * [multinomial naive
    bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
  * [linear support vector
    machines](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)

If you want to get fancy and flex your machine learning skills, you can do some
clever feature selection pipelines, or even build a neural network (scikit-learn
has neural networks too). In any case, try a few different classifiers with your
various feature settings, collect your results, and see how well you can do on
this classification problem!

This is a binary classification problem on a balanced dataset, so reporting
accuracy is fairly sensible. But, make sure to do some error analysis! Do your
models get any interesting instances wrong, in the test set?

## Deliverables

You should write up a short discussion of your process, either as a Markdown
document, like as a nice `README.md`, or as a PDF, and include both a
description of your code and a summary of your experimental results.

An important part of experimentation is error analysis -- your classifiers will
almost certainly make some interesting errors, so take a look at the text that
they misclassify! What went wrong?

In the writeup, be sure to describe your experimental procedure, give some
explanation for how you made your choices, and give a quick description of who
did which tasks in your group. Your work here should be written by you and your
teammates, and not by a robot.

Your code can be included as either Python files (with instructions for how to
run them), or as a Jupyter notebook.

# Submission Instructions

Check in all of your team's materials into your provided gitlab repository,
which should look something like:
`https://git.ucsc.edu/cse-143-spring-2026/team_XX` , into a new directory called
`hw1`. No need to check in your dataset for this homework -- we'll all have the
same one, based on the tool.

Post a link to your team's hw1 directory in the Canvas assignment, and make sure
to include your teammates on the turn-in! You only need to make one submission
per team.
