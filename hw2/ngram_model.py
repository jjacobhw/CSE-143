from collections import Counter
import math


START_TOKEN = "<START>"
STOP_TOKEN = "<STOP>"
UNK_TOKEN = "<UNK>"

# split the tokens by the spaces between them 
def read_tokenized_sentences(path):
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if tokens:
                sentences.append(tokens)
    return sentences


# creates the vocab using the sentences in the training sentences data,
# excluding tokens that appear less than 
def build_vocabulary(train_sentences, unk_threshold=3):
    token_counts = Counter()
    for sentence in train_sentences:
        for token in sentence:
            token_counts[token] += 1

    vocab = set()
    for token, count in token_counts.items():
        if count >= unk_threshold:
            vocab.add(token)

    vocab.add(UNK_TOKEN)
    vocab.add(STOP_TOKEN)
    if START_TOKEN in vocab:
        vocab.remove(START_TOKEN)

    return vocab, token_counts


def preprocess_sentences(sentences, vocab):
    processed = []
    for sentence in sentences:
        replaced = []
        for token in sentence:
            if token in vocab:
                replaced.append(token)
            else:
                replaced.append(UNK_TOKEN)
        processed.append([START_TOKEN] + replaced + [STOP_TOKEN])
    return processed


class NgramModel:
    def __init__(self, train_sentences_processed, vocab):
        self.train_sentences = train_sentences_processed
        self.vocab = vocab
        self.vocab_size = len(vocab)

        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.trigram_counts = Counter()

        self.bigram_context_counts = Counter()
        self.trigram_context_counts = Counter()

        self.total_unigram_tokens = 0
        self._count_ngrams()

    def _count_ngrams(self):
        for sentence in self.train_sentences:
            # Predict all tokens except <START>. Include <STOP> in M and counts.
            for i in range(1, len(sentence)):
                token = sentence[i]
                self.unigram_counts[token] += 1
                self.total_unigram_tokens += 1

            for i in range(1, len(sentence)):
                prev_token = sentence[i - 1]
                token = sentence[i]
                self.bigram_counts[(prev_token, token)] += 1
                self.bigram_context_counts[prev_token] += 1

            for i in range(2, len(sentence)):
                prev_prev = sentence[i - 2]
                prev = sentence[i - 1]
                token = sentence[i]
                self.trigram_counts[(prev_prev, prev, token)] += 1
                self.trigram_context_counts[(prev_prev, prev)] += 1


    def mle_unigram_prob(self, token):
        count = self.unigram_counts[token]
        if count == 0:
            return 0.0
        return count / self.total_unigram_tokens


    def mle_bigram_prob(self, prev_token, token):
        context_count = self.bigram_context_counts[prev_token]
        if context_count == 0:
            return 0.0
        return self.bigram_counts[(prev_token, token)] / context_count


    def mle_trigram_prob(self, prev_prev, prev, token):
        context_count = self.trigram_context_counts[(prev_prev, prev)]
        if context_count == 0:
            return 0.0
        return self.trigram_counts[(prev_prev, prev, token)] / context_count

    def additive_unigram_prob(self, token, alpha):
        numerator = self.unigram_counts[token] + alpha
        denominator = self.total_unigram_tokens + alpha * self.vocab_size
        return numerator / denominator

    def additive_bigram_prob(self, prev_token, token, alpha):
        numerator = self.bigram_counts[(prev_token, token)] + alpha
        denominator = self.bigram_context_counts[prev_token] + alpha * self.vocab_size
        return numerator / denominator

    def additive_trigram_prob(self, prev_prev, prev, token, alpha):
        numerator = self.trigram_counts[(prev_prev, prev, token)] + alpha
        denominator = self.trigram_context_counts[(prev_prev, prev)] + alpha * self.vocab_size
        return numerator / denominator

    def perplexity_unigram(self, sentences):
        log_prob_sum = 0.0
        m = 0

        for sentence in sentences:
            # M counts predicted tokens only: include <STOP>, exclude <START>.
            for i in range(1, len(sentence)):
                token = sentence[i]
                p = self.mle_unigram_prob(token)
                if p == 0.0:
                    return math.inf
                log_prob_sum += math.log(p)
                m += 1

        return math.exp(-log_prob_sum / m)

    def perplexity_bigram(self, sentences):
        log_prob_sum = 0.0
        m = 0

        for sentence in sentences:
            for i in range(1, len(sentence)):
                prev_token = sentence[i - 1]
                token = sentence[i]
                p = self.mle_bigram_prob(prev_token, token)
                if p == 0.0:
                    return math.inf
                log_prob_sum += math.log(p)
                m += 1

        return math.exp(-log_prob_sum / m)

    def perplexity_trigram(self, sentences):
        log_prob_sum = 0.0
        m = 0

        for sentence in sentences:
            for i in range(1, len(sentence)):
                token = sentence[i]
                if i == 1:
                    p = self.mle_bigram_prob(sentence[i - 1], token)
                else:
                    p = self.mle_trigram_prob(sentence[i - 2], sentence[i - 1], token)

                if p == 0.0:
                    return math.inf
                log_prob_sum += math.log(p)
                m += 1

        return math.exp(-log_prob_sum / m)

    def perplexity_additive_unigram(self, sentences, alpha):
        log_prob_sum = 0.0
        m = 0
        for sentence in sentences:
            for i in range(1, len(sentence)):
                p = self.additive_unigram_prob(sentence[i], alpha)
                log_prob_sum += math.log(p)
                m += 1
        return math.exp(-log_prob_sum / m)

    def perplexity_additive_bigram(self, sentences, alpha):
        log_prob_sum = 0.0
        m = 0
        for sentence in sentences:
            for i in range(1, len(sentence)):
                p = self.additive_bigram_prob(sentence[i - 1], sentence[i], alpha)
                log_prob_sum += math.log(p)
                m += 1
        return math.exp(-log_prob_sum / m)

    def perplexity_additive_trigram(self, sentences, alpha):
        log_prob_sum = 0.0
        m = 0
        for sentence in sentences:
            for i in range(1, len(sentence)):
                if i == 1:
                    p = self.additive_bigram_prob(sentence[i - 1], sentence[i], alpha)
                else:
                    p = self.additive_trigram_prob(sentence[i - 2], sentence[i - 1], sentence[i], alpha)
                log_prob_sum += math.log(p)
                m += 1
        return math.exp(-log_prob_sum / m)

    def perplexity_interpolation(self, sentences, lambda1, lambda2, lambda3):
        if abs((lambda1 + lambda2 + lambda3) - 1.0) > 1e-9:
            raise ValueError("Lambdas must sum to 1.")

        log_prob_sum = 0.0
        m = 0

        for sentence in sentences:
            for i in range(1, len(sentence)):
                token = sentence[i]

                p1 = self.mle_unigram_prob(token)
                p2 = self.mle_bigram_prob(sentence[i - 1], token)

                # For first token after <START>, use bigram behavior for trigram piece too.
                # This mirrors the assignment's trigram first-token rule.
                if i >= 2:
                    p3 = self.mle_trigram_prob(sentence[i - 2], sentence[i - 1], token)
                else:
                    p3 = self.mle_bigram_prob(sentence[i - 1], token)

                p = lambda1 * p1 + lambda2 * p2 + lambda3 * p3
                if p == 0.0:
                    return math.inf

                log_prob_sum += math.log(p)
                m += 1

        return math.exp(-log_prob_sum / m)


def format_perplexity(value):
    if value == math.inf:
        return "inf"
    return f"{value:.4f}"
