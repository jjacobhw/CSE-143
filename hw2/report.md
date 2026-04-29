# N-gram Language Models: Report

## 1. Introduction

This report describes the implementation and evaluation of unigram, bigram, and trigram n-gram language models on the 1 Billion Word Benchmark dataset. Three estimation approaches are compared: Maximum Likelihood Estimation (MLE), additive (Laplace) smoothing, and linear interpolation. All models are trained on the provided training split and hyperparameters are selected using the development set; final results are reported on the held-out test set.

---

## 2. Model Descriptions

### 2.1 N-gram MLE Models

An n-gram language model estimates the probability of each word given the preceding n−1 words. Sentence probability is computed as the product of these conditional probabilities, with each sentence delimited by a `<START>` token at the left boundary and a `<STOP>` token at the right boundary. Perplexity is the evaluation metric throughout:

$$\text{PPL} = \exp\!\left(-\frac{1}{M}\sum_{i=1}^{M}\log P(w_i)\right)$$

where M is the total number of predicted tokens (including `<STOP>`, excluding `<START>`).

**Unigram.** Each token is assumed independent of context:

$$P(w_i) = \frac{C(w_i)}{N}$$

where $C(w_i)$ is the token's count in training and N is the total token count.

**Bigram.** Each token is conditioned on the immediately preceding token:

$$P(w_i \mid w_{i-1}) = \frac{C(w_{i-1},\, w_i)}{C(w_{i-1})}$$

**Trigram.** Each token is conditioned on the two preceding tokens:

$$P(w_i \mid w_{i-2}, w_{i-1}) = \frac{C(w_{i-2},\, w_{i-1},\, w_i)}{C(w_{i-2},\, w_{i-1})}$$

For the first predicted token in each sentence (position i=1 following `<START>`), the trigram model falls back to bigram probability, since no two-token left context is available.

### 2.2 Additive Smoothing

MLE assigns zero probability to any n-gram not seen in training, yielding infinite perplexity on any development or test sentence containing an unseen n-gram. Additive smoothing addresses this by adding a constant α > 0 to every n-gram count before normalization:

$$P_{\alpha}(w_i \mid \text{ctx}) = \frac{C(\text{ctx},\, w_i) + \alpha}{C(\text{ctx}) + \alpha \lvert V \rvert}$$

where $\lvert V \rvert$ is the vocabulary size. This guarantees a non-zero probability for every possible token, at the cost of redistributing some probability mass from observed to unobserved events.

### 2.3 Linear Interpolation

Linear interpolation combines all three MLE models with non-negative weights λ₁, λ₂, λ₃ that sum to 1:

$$P_{\text{interp}}(w_i) = \lambda_1 \cdot P_{\text{uni}}(w_i) + \lambda_2 \cdot P_{\text{bi}}(w_i \mid w_{i-1}) + \lambda_3 \cdot P_{\text{tri}}(w_i \mid w_{i-2}, w_{i-1})$$

Since the unigram component is always positive, the interpolated probability is never zero as long as the word appears in the training vocabulary (or is mapped to `<UNK>`). Weights are treated as hyperparameters selected on the development set.

---

## 3. Experimental Setup

**Dataset.** The 1 Billion Word Benchmark is used with its standard train/dev/test split. Each file is pre-tokenized; sentences are read by whitespace splitting.

**Preprocessing.** Each sentence is wrapped with a `<START>` token at the beginning and a `<STOP>` token at the end. The vocabulary is built from the training set only: any token appearing fewer than 3 times in training is replaced with the special `<UNK>` symbol. The same vocabulary is then applied to dev and test—tokens not in the training vocabulary are likewise replaced with `<UNK>`. The final vocabulary includes `<UNK>` and `<STOP>` but excludes `<START>`, yielding a vocabulary size of **26,602**.

**Debug verification.** For the two-token sentence "HDTV ." the implementation produces the following perplexities, confirming correctness:

| Model | Computed PPL | Expected |
|-------|-------------|----------|
| Unigram | 658.04 | ~658 |
| Bigram | 63.71 | ~63.7 |
| Trigram | 39.48 | ~39.5 |
| Interpolation (λ=0.1, 0.3, 0.6) | 48.11 | ~48.1 |

---

## 4. Results

### 4.1 MLE Models

Table 1 reports perplexity for all three MLE models across splits.

**Table 1. MLE perplexity.**

| Model | Train PPL | Dev PPL | Test PPL |
|-------|-----------|---------|----------|
| Unigram | 976.54 | 892.25 | 896.50 |
| Bigram | 77.07 | ∞ | ∞ |
| Trigram | 7.87 | ∞ | ∞ |

Figure 1 visualizes the train perplexities on a log scale to make the differences across model orders visible.

![Figure 1: MLE Train Perplexity by Model Order (log scale)](figures/fig1_mle_train_ppl.png)

**Figure 1.** MLE train perplexity on a log scale. Each additional order of n-gram dramatically reduces train perplexity, with trigram nearly memorizing the training data.

The bigram and trigram models produce infinite perplexity on both dev and test. This is because any unseen bigram or trigram encountered in dev/test yields a MLE probability of zero, which makes the log-probability undefined. This data sparsity problem motivates the smoothing approaches in the following sections.

The unigram model avoids this issue because every token in dev/test is either in the training vocabulary or mapped to `<UNK>`, both of which have non-zero training counts. However, its perplexity is high (~892 on dev) because ignoring context produces poor probability estimates.

### 4.2 Additive Smoothing

Three values of α are evaluated: 1.0, 0.1, and 0.01. Table 2 reports training and development perplexity for all combinations.

**Table 2. Additive smoothing perplexity (train / dev).**

| α | Unigram Train | Unigram Dev | Bigram Train | Bigram Dev | Trigram Train | Trigram Dev |
|---|---------------|-------------|--------------|------------|---------------|-------------|
| 1.0 | 977.51 | 894.39 | 1442.31 | 1669.66 | 6244.42 | 9676.65 |
| 0.1 | 976.55 | 892.40 | 407.84 | 701.73 | 1115.69 | 4899.49 |
| **0.01** | **976.54** | **892.26** | **157.91** | **442.99** | **169.90** | **2838.58** |

Figure 2 shows dev perplexity by model and α value.

![Figure 2: Additive Smoothing Dev Perplexity vs α](figures/fig2_additive_dev_ppl.png)

**Figure 2.** Additive smoothing dev perplexity (log scale). Smaller α is better across all model orders.

The best α on the development set is **0.01 for all three models**. The smaller α is preferred because it moves less probability mass from observed to unobserved events—at α=1 the denominator is inflated so aggressively that even well-observed n-grams receive far less than their MLE probability, hurting performance on common patterns. Several observations follow:

- **Unigram is largely unaffected.** Since all vocabulary tokens appear in training, smoothing the unigram denominator by α·|V| has minimal effect.
- **Trigram additive smoothing is worse than bigram.** Trigram contexts are sparser: many (w_{i-2}, w_{i-1}) pairs have zero training count. When α·|V| is added to a zero-count denominator, probability mass is spread uniformly across the entire vocabulary, producing poor estimates. This effect compounds at higher order, so additive smoothing degrades with model order.

**Table 3. Test perplexity with best α (chosen from dev).**

| Model | Best α | Test PPL |
|-------|--------|----------|
| Unigram | 0.01 | 896.51 |
| Bigram | 0.01 | 440.81 |
| Trigram | 0.01 | 2821.34 |

### 4.3 Linear Interpolation

Table 4 reports train and dev perplexity for five sets of interpolation weights. The assignment-specified set (0.1, 0.3, 0.6) is included as the first row.

**Table 4. Interpolation perplexity by lambda set.**

| λ₁ (unigram) | λ₂ (bigram) | λ₃ (trigram) | Train PPL | Dev PPL |
|--------------|------------|--------------|-----------|---------|
| 0.1 | 0.3 | 0.6 | 11.15 | 352.23 |
| 0.2 | 0.3 | 0.5 | 12.88 | 306.16 |
| 0.3 | 0.3 | 0.4 | 15.30 | 286.63 |
| 0.2 | 0.2 | 0.6 | 11.53 | 338.90 |
| **0.4** | **0.3** | **0.3** | **18.93** | **279.05** |

Figure 3 visualizes train versus dev perplexity for each lambda set.

![Figure 3: Interpolation Train vs Dev Perplexity](figures/fig3_interpolation_ppl.png)

**Figure 3.** Interpolation train (blue) and dev (orange) perplexity for each lambda set on a log scale. The best-dev set is marked with *.

The best lambda set on the development set is **λ₁=0.4, λ₂=0.3, λ₃=0.3**. As λ₃ (the trigram weight) increases, train perplexity decreases—the trigram component nearly memorizes the training data—but dev perplexity worsens significantly, as the model over-relies on trigram contexts that are unseen at evaluation time. Placing more weight on the unigram term provides a robust floor that prevents the interpolated probability from collapsing to near zero when higher-order contexts are unseen.

**Chosen hyperparameters: λ₁=0.4, λ₂=0.3, λ₃=0.3 → Test PPL = 278.83**

Interpolation (278.83) substantially outperforms the best additive smoothing model (bigram at 440.81) because it handles sparsity by gracefully falling back to lower-order models rather than uniformly spreading mass across the entire vocabulary.

---

## 5. Ablation Experiments

### 5.1 Effect of Reducing Training Data

To assess sensitivity to training set size, the model is re-trained on the first half of the training sentences (30,765 sentences instead of 61,530). With fewer training sentences, more word types fall below the UNK threshold of 3, reducing the vocabulary from 26,602 to **17,537** tokens.

**Table 5. Half-training data: interpolation dev perplexity.**

| λ₁, λ₂, λ₃ | Full Train Dev PPL | Half Train Dev PPL |
|------------|--------------------|--------------------|
| 0.1, 0.3, 0.6 | 352.23 | 352.97 |
| 0.2, 0.3, 0.5 | 306.16 | 299.54 |
| 0.3, 0.3, 0.4 | 286.63 | 275.72 |
| 0.2, 0.2, 0.6 | 338.90 | 332.81 |
| **0.4, 0.3, 0.3** | **279.05** | **264.43** |

Figure 4 compares dev perplexity across all lambda sets for full versus half training data.

![Figure 4: Half Training Data vs Full Training Data](figures/fig4_halftrain_dev_ppl.png)

**Figure 4.** Interpolation dev perplexity for full training (blue) and half training (orange). Half-train reports lower perplexity across all lambda sets.

**Test perplexity with best lambdas (0.4, 0.3, 0.3): Full = 278.83, Half = 265.12**

The result appears counterintuitive: halving the training data *decreases* measured perplexity. The explanation lies in the vocabulary collapse. With half the data, approximately 9,000 additional word types fall below the UNK threshold and are mapped to `<UNK>`. This means the dev set is also preprocessed with this smaller vocabulary, so many more dev tokens become `<UNK>`. The `<UNK>` token is now very frequent and well-estimated, making its individual predictions "easy" and lowering average perplexity.

In an information-theoretic sense, the half-train model is genuinely worse: it has seen fewer examples and loses the ability to distinguish among thousands of word types. A controlled comparison holding the vocabulary fixed would reveal the expected increase in perplexity. The empirical decrease observed here is an artifact of measuring perplexity on a different (more aggressively UNK-ified) dev set. **Reducing training data would increase perplexity on previously unseen data if the vocabulary and preprocessing were held constant.**

### 5.2 Effect of UNK Threshold

The default preprocessing replaces tokens with count < 3 with `<UNK>`. Here the threshold is raised to 5, so tokens appearing fewer than 5 times in training are replaced. This shrinks the vocabulary from 26,602 to **18,119** tokens.

**Table 6. UNK threshold comparison: test perplexity.**

| Method | Threshold = 3 (default) | Threshold = 5 |
|--------|------------------------|----------------|
| Additive Bigram (α=0.01) | 440.81 | 328.78 |
| Additive Trigram (α=0.01) | 2821.34 | 1784.66 |
| Interpolation (best λ) | 278.83 | 235.18 |

Figure 5 compares test perplexity for the two threshold values across the main smoothing methods.

![Figure 5: UNK Threshold Effect on Test Perplexity](figures/fig5_unk_threshold_test_ppl.png)

**Figure 5.** Test perplexity under UNK threshold=3 (default) and threshold=5. Higher threshold reduces measured perplexity across all methods.

Raising the UNK threshold to 5 consistently lowers test perplexity. Two mechanisms drive this:

1. **Reduced sparsity.** A smaller vocabulary means more n-gram types are observed during training. Bigram and trigram probability estimates are more reliable because the training counts are more concentrated over fewer types.
2. **Easier task.** More test tokens are mapped to `<UNK>`, which is a frequent, well-estimated symbol. This same effect was observed in the half-training ablation: when rare words are absorbed into `<UNK>`, the model is predicting a lower-diversity token stream, which mechanically lowers perplexity.

The trade-off is expressiveness: with a threshold of 5 the model cannot distinguish among any of the ~8,500 word types that would otherwise have been in the vocabulary, losing information that could be valuable in downstream tasks. The perplexity improvement is therefore partly a measurement artifact rather than a genuine gain in language modeling quality.

---

## 6. Summary

Table 7 consolidates all final test perplexities across methods and experimental conditions.

**Table 7. Summary of test perplexity across all models and conditions.**

| Method | UNK Thresh | Train Size | Test PPL |
|--------|-----------|------------|----------|
| MLE Unigram | 3 | Full | 896.50 |
| MLE Bigram | 3 | Full | ∞ |
| MLE Trigram | 3 | Full | ∞ |
| Additive Unigram α=0.01 | 3 | Full | 896.51 |
| Additive Bigram α=0.01 | 3 | Full | 440.81 |
| Additive Trigram α=0.01 | 3 | Full | 2821.34 |
| **Interpolation λ=(0.4,0.3,0.3)** | **3** | **Full** | **278.83** |
| Interpolation λ=(0.4,0.3,0.3) | 3 | Half | 265.12 |
| Interpolation λ=(0.4,0.3,0.3) | 5 | Full | 235.18 |

Linear interpolation with **λ₁=0.4, λ₂=0.3, λ₃=0.3** (selected on the development set) achieves the best test perplexity of **278.83** under standard preprocessing conditions. It outperforms additive smoothing because it degrades gracefully: when a trigram context is unseen, it falls back to the bigram and unigram components rather than distributing mass uniformly over the vocabulary. The surprising superiority of higher λ₁ (unigram weight) reflects the high sparsity of trigrams in this corpus—over-relying on the trigram component causes near-zero probabilities on unseen contexts that overwhelm any gains from context-specificity.
