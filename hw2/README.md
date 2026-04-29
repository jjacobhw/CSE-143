## CSE 143 HW2 - Simple N-gram Language Models

This folder contains a beginner-friendly Python implementation of:
- Unigram, bigram, trigram MLE language models
- Additive smoothing (`alpha = 1, 0.1, 0.01`)
- Linear interpolation of unigram/bigram/trigram MLE probabilities

### Files

- `ngram_model.py`: counting, probabilities, perplexity functions
- `run_experiments.py`: command-line script to run all experiments
- `.gitignore`: ignores local data and generated results

### Data

Expected local data files (not committed):
- `hw2-data/1b_benchmark.train.tokens`
- `hw2-data/1b_benchmark.dev.tokens`
- `hw2-data/1b_benchmark.test.tokens`

### Preprocessing used

- Read each sentence with `line.strip().split()`
- Add `<START>` and `<STOP>` to each sentence
- Build vocabulary from **training only**
- Replace training tokens with count `< 3` by `<UNK>`
- Replace unseen dev/test tokens with `<UNK>`
- Vocabulary includes `<UNK>` and `<STOP>`, excludes `<START>`

### Run

From the `hw2/` directory:

```bash
python3 run_experiments.py
```

Optional flags:

```bash
# save CSV outputs in results/
python3 run_experiments.py --save-csv

# experiment: train on half of training sentences
python3 run_experiments.py --half-train

# experiment: use UNK threshold 5 instead of 3
python3 run_experiments.py --unk-threshold 5
```

### First outputs to verify

1. Vocabulary size line:
   - with default preprocessing, expected size is `26602`
2. Debug sentence (`HDTV .`) perplexities should be close to:
   - unigram: ~658
   - bigram: ~63.7
   - trigram: ~39.5
   - interpolation `(0.1, 0.3, 0.6)`: ~48.1
