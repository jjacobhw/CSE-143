# Homework 2 Write-Up

## Introduction

For this assignment, we implemented unigram, bigram, and trigram language models on the provided dataset. We wanted to was to compare different n-gram models and smoothing methods and see how their perplexities changed between them and with appraoches like smoothing and linear interpolation

We experimented with three main model settings. First, we implemented standard maximum likelihood estimate models without smoothing. Then, we implemented additive smoothing with three different alpha values (0.01, 0.1, and 1). Finally, we implemented linear interpolation, where the final probability is a weighted combination of the unigram, bigram, and trigram probabilities. We used the development set to choose hyperparameters and then reported our final results on the test set.

## Data Preprocessing and Vocabulary

The dataset was already whitespace-tokenized, so we split the sentences on spaces to gather the tokens. During training, we counted word frequencies and built a vocabulary from the training set. In the standard experiment, tokens that appeared fewer than 3 times in the training data were replaced with `<UNK>`. We also added `<STOP>` to the end of every sentence and added `<START>` to the beginning of every sentence. However, `<START>` was not included as a predicted vocabulary item.

In the standard full-training experiment, the vocabulary size was 26602, which matched the expected vocabulary size from the assignment. This helped confirm that the vocabulary and `<UNK>` preprocessing were working correctly.

## Model Descriptions

For the unigram model, each token probability was estimated from its overall frequency in the training data. This model does not use context, so it only learns which words are common overall. 

For the bigram model, each probability was based on the previous token.

 For the trigram model, each probability was based on the previous two tokens. For the first real token after `<START>`, we had the trigram model use the bigram probability because there are not two previous tokens available yet.

Perplexity was calculated using log probabilities to avoid underflow. The token count `M` included `<STOP>` but did not include `<START>`.

## Sanity Check

We tested our models against the sentence `HDTV .`, as asked for by the instructions. Our standard full-vocabulary experiment gave us these results:

| Model | Our Perplexity | Expected Perplexity |
|---|---:|---:|
| Unigram MLE | 658.0445 | about 658 |
| Bigram MLE | 63.7076 | about 63.7 |
| Trigram MLE | 39.4787 | about 39.5 |
| Interpolation, lambdas = (0.1, 0.3, 0.6) | 48.1135 | about 48.1 |

These results we got matched up with the expected results, so we had some confidence that our modles were working correcting.

## Part 1: MLE Results Without Smoothing

The table below shows the MLE perplexities for the unigram, bigram, and trigram models using the standard full training data and the default `<UNK>` threshold of 3.

| Model | Train Perplexity | Dev Perplexity | Test Perplexity |
|---|---:|---:|---:|
| Unigram | 976.5437 | 892.2466 | 896.4995 |
| Bigram | 77.0735 | inf | inf |
| Trigram | 7.8730 | inf | inf |

The unigram model had finite perplexity on all three sets because every dev and test token was either in the vocabulary or mapped to `<UNK>`. However, the unsmoothed bigram and trigram models had infinite perplexity on the development and test sets. This likely happened because the dev and test sets contained bigrams or trigrams that never appeared in training. With MLE and no smoothing, unseen n-grams get probability 0, and a single zero-probability token makes the whole sentence probability 0, which causes the infinite perplexity.

The training perplexity decreased a lot from unigram to bigram to trigram. This made sense to us since higher-order models tend to memorize more specific context from the training data. However, without smoothing, that memorization does not generalize well to unseen data.

## Part 2: Additive Smoothing Results

We tested additive smoothing with alpha values of 1.0, 0.1, and 0.01. The results we got on the train and dev data sets are:

| Alpha | Model | Train Perplexity | Dev Perplexity |
|---:|---|---:|---:|
| 1.0 | Unigram | 977.5079 | 894.3902 |
| 1.0 | Bigram | 1442.3087 | 1669.6553 |
| 1.0 | Trigram | 6244.4249 | 9676.6511 |
| 0.1 | Unigram | 976.5549 | 892.3952 |
| 0.1 | Bigram | 407.8449 | 701.7257 |
| 0.1 | Trigram | 1115.6875 | 4899.4883 |
| 0.01 | Unigram | 976.5439 | 892.2607 |
| 0.01 | Bigram | 157.9060 | 442.9904 |
| 0.01 | Trigram | 169.8999 | 2838.5776 |

For all three models, alpha = 0.01 gave the best development perplexity out of the alpha values we tested.

| Model | Best Alpha Based on Dev | Test Perplexity |
|---|---:|---:|
| Unigram | 0.01 | 896.5126 |
| Bigram | 0.01 | 440.8067 |
| Trigram | 0.01 | 2821.3364 |

Additive smoothing fixed the infinite perplexity problem because unseen n-grams no longer received probability 0. However, additive smoothing was not equally effective for every model. The bigram model worked much better than the trigram model with additive smoothing. The trigram model still had very high dev and test perplexity because there are many possible trigram contexts, and additive smoothing spreads probability mass across the entire vocabulary for each context. This can hurt performance a lot when most trigrams are rare.

Alpha = 1.0 performed especially poorly for bigrams and trigrams because it added too much probability mass to unseen events. Alpha = 0.01 was much less aggressive, so it preserved more of the useful training counts while still avoiding zero probabilities.

## Part 3: Linear Interpolation Results

For linear interpolation, we combined the unsmoothed unigram, bigram, and trigram probabilities using a standard weighted sum of the probabilities where \(\lambda_1\), \(\lambda_2\), and \(\lambda_3\) are the weights for the unigram, bigram, and trigram models, respectively.

The lambdas had to sum to 1. We tested five lambda combinations and chose the best one using the development set.

| Lambdas `(lambda1, lambda2, lambda3)` | Train Perplexity | Dev Perplexity |
|---|---:|---:|
| (0.1, 0.3, 0.6) | 11.1515 | 352.2342 |
| (0.2, 0.3, 0.5) | 12.8849 | 306.1609 |
| (0.3, 0.3, 0.4) | 15.3009 | 286.6349 |
| (0.2, 0.2, 0.6) | 11.5326 | 338.8951 |
| (0.4, 0.3, 0.3) | 18.9250 | 279.0537 |

The best lambda setting on the development set was:
(lambda1, lambda2, lambda3) = (0.4, 0.3, 0.3)

Using these hyperparameters, the test perplexity was:

| Model | Test Perplexity |
|---|---:|
| Interpolation with lambdas = (0.4, 0.3, 0.3) | 278.8287 |

This was the best standard full-vocabulary result overall. Interpolation likely worked better than additive smoothing because it allowed the model to use trigram and bigram information when it was available, while still falling back partly on lower-order models when the higher-order context was unreliable. The best lambda setting gave the largest weight to the unigram model, which likely means that having a strong fallback model was important for the model to general to unseen sentences.

## Additional Experiment: Using Half of the Training Data

We also tested what happened when only the first half of the training sentences was used. In this experiment, the vocabulary size decreased to 17537. We saw the following results:

| Setting | Vocabulary Size | Best Interpolation Lambdas | Dev Perplexity | Test Perplexity |
|---|---:|---|---:|---:|
| Full training data, threshold 3 | 26602 | (0.4, 0.3, 0.3) | 279.0537 | 278.8287 |
| Half training data, threshold 3 | 17537 | (0.4, 0.3, 0.3) | 264.4279 | 265.1162 |

In this experiment, using half of the training data actually decreased perplexity on the dev and test sets. We found it really suprising since we thought that in almost all cases, using less training data should make a language model worse on unseen data. However, in our setup, using less training data also changed the vocabulary. With fewer training sentences, more rare words were mapped to `<UNK>`, so the model had a smaller vocabulary and an easier prediction task. Instead of predicting many rare words separately, the model could predict the single `<UNK>` token more often.

Because of this, the lower perplexity does not necessarily mean that the half-data model is a better language model in a real sense. It mostly shows that perplexity can be affected by how the vocabulary and unknown-word handling are defined.

## Additional Experiment: UNK Threshold of 5

We also tested a higher unknown-word threshold of 5, so tokens appearing less than 5 times were converted to `<UNK>`. We got the following results:

| Setting | Vocabulary Size | Best Interpolation Lambdas | Dev Perplexity | Test Perplexity |
|---|---:|---|---:|---:|
| Full training data, threshold 3 | 26602 | (0.4, 0.3, 0.3) | 279.0537 | 278.8287 |
| Full training data, threshold 5 | 18119 | (0.4, 0.3, 0.3) | 235.3662 | 235.1830 |

Using an `<UNK>` threshold of 5 decreased the vocabulary size to 18119, as well as decreased perplexity compared to the standard threshold of 3. We believe this likely happened since more rare words were collapsed into `<UNK>`, which made the vocabulary smaller and made the prediction task easier, which is similar to what happened when we cut the training size in half. Rare words are difficult for n-gram models to predict because they do not appear often enough to get reliable counts. So mapping more of those rare words to `<UNK>` gives the model more examples of `<UNK>` and reduces the number of separate low-frequency tokens it has to model.