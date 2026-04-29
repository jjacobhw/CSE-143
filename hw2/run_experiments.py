import argparse
from pathlib import Path

from ngram_model import (
    NgramModel,
    build_vocabulary,
    preprocess_sentences,
    read_tokenized_sentences,
)


def format_pp(value):
    if value == float("inf"):
        return "inf"
    return f"{value:.4f}"


def main():
    parser = argparse.ArgumentParser(description="Run unigram/bigram/trigram experiments.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="hw2-data",
        help="Directory containing train/dev/test token files.",
    )
    parser.add_argument(
        "--unk-threshold",
        type=int,
        default=3,
        help="Tokens with count < threshold become <UNK>.",
    )
    parser.add_argument(
        "--half-train",
        action="store_true",
        help="Use only first half of training sentences.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_path = data_dir / "1b_benchmark.train.tokens"
    dev_path = data_dir / "1b_benchmark.dev.tokens"
    test_path = data_dir / "1b_benchmark.test.tokens"

    print("Loading tokenized data...")
    train_raw = read_tokenized_sentences(train_path)
    dev_raw = read_tokenized_sentences(dev_path)
    test_raw = read_tokenized_sentences(test_path)

    if args.half_train:
        half = len(train_raw) // 2
        train_raw = train_raw[:half]
        print(f"Using half training data: {half} sentences")

    vocab, _ = build_vocabulary(train_raw, unk_threshold=args.unk_threshold)
    print(f"Vocabulary size (should be 26602 with default preprocessing): {len(vocab)}")

    train = preprocess_sentences(train_raw, vocab)
    dev = preprocess_sentences(dev_raw, vocab)
    test = preprocess_sentences(test_raw, vocab)

    model = NgramModel(train, vocab)

    # Debug sentence check requested by assignment.
    debug_processed = preprocess_sentences([["HDTV", "."]], vocab)

    print()
    print("MLE RESULTS")
    for model_name in ["unigram", "bigram", "trigram"]:
        if model_name == "unigram":
            train_pp = model.perplexity_unigram(train)
            dev_pp = model.perplexity_unigram(dev)
            test_pp = model.perplexity_unigram(test)
        elif model_name == "bigram":
            train_pp = model.perplexity_bigram(train)
            dev_pp = model.perplexity_bigram(dev)
            test_pp = model.perplexity_bigram(test)
        else:
            train_pp = model.perplexity_trigram(train)
            dev_pp = model.perplexity_trigram(dev)
            test_pp = model.perplexity_trigram(test)

        print(f"{model_name} train perplexity: {format_pp(train_pp)}")
        print(f"{model_name} dev perplexity: {format_pp(dev_pp)}")
        print(f"{model_name} test perplexity: {format_pp(test_pp)}")
        print()

    debug_uni = model.perplexity_unigram(debug_processed)
    debug_bi = model.perplexity_bigram(debug_processed)
    debug_tri = model.perplexity_trigram(debug_processed)
    debug_interp = model.perplexity_interpolation(debug_processed, 0.1, 0.3, 0.6)

    print("DEBUG CHECK: HDTV .")
    print(f"unigram perplexity: {format_pp(debug_uni)} expected about 658")
    print(f"bigram perplexity: {format_pp(debug_bi)} expected about 63.7")
    print(f"trigram perplexity: {format_pp(debug_tri)} expected about 39.5")
    print(
        "interpolation perplexity with (0.1, 0.3, 0.6): "
        f"{format_pp(debug_interp)} expected about 48.1"
    )

    additive_alphas = [1.0, 0.1, 0.01]
    best_alpha = {"unigram": None, "bigram": None, "trigram": None}
    best_dev = {"unigram": float("inf"), "bigram": float("inf"), "trigram": float("inf")}

    print()
    print("ADDITIVE SMOOTHING RESULTS")
    for alpha in additive_alphas:
        print(f"alpha = {alpha}")

        uni_train = model.perplexity_additive_unigram(train, alpha)
        uni_dev = model.perplexity_additive_unigram(dev, alpha)
        bi_train = model.perplexity_additive_bigram(train, alpha)
        bi_dev = model.perplexity_additive_bigram(dev, alpha)
        tri_train = model.perplexity_additive_trigram(train, alpha)
        tri_dev = model.perplexity_additive_trigram(dev, alpha)

        print(f"unigram train perplexity: {format_pp(uni_train)}")
        print(f"unigram dev perplexity: {format_pp(uni_dev)}")
        print(f"bigram train perplexity: {format_pp(bi_train)}")
        print(f"bigram dev perplexity: {format_pp(bi_dev)}")
        print(f"trigram train perplexity: {format_pp(tri_train)}")
        print(f"trigram dev perplexity: {format_pp(tri_dev)}")
        print()

        if uni_dev < best_dev["unigram"]:
            best_dev["unigram"] = uni_dev
            best_alpha["unigram"] = alpha
        if bi_dev < best_dev["bigram"]:
            best_dev["bigram"] = bi_dev
            best_alpha["bigram"] = alpha
        if tri_dev < best_dev["trigram"]:
            best_dev["trigram"] = tri_dev
            best_alpha["trigram"] = alpha

    best_uni_alpha = best_alpha["unigram"]
    best_bi_alpha = best_alpha["bigram"]
    best_tri_alpha = best_alpha["trigram"]

    best_uni_test = model.perplexity_additive_unigram(test, best_uni_alpha)
    best_bi_test = model.perplexity_additive_bigram(test, best_bi_alpha)
    best_tri_test = model.perplexity_additive_trigram(test, best_tri_alpha)

    print(f"Best unigram alpha based on dev: {best_uni_alpha}")
    print(f"unigram test perplexity with best alpha: {format_pp(best_uni_test)}")
    print()
    print(f"Best bigram alpha based on dev: {best_bi_alpha}")
    print(f"bigram test perplexity with best alpha: {format_pp(best_bi_test)}")
    print()
    print(f"Best trigram alpha based on dev: {best_tri_alpha}")
    print(f"trigram test perplexity with best alpha: {format_pp(best_tri_test)}")

    lambda_sets = [
        (0.1, 0.3, 0.6),
        (0.2, 0.3, 0.5),
        (0.3, 0.3, 0.4),
        (0.2, 0.2, 0.6),
        (0.4, 0.3, 0.3),
    ]

    print()
    print("INTERPOLATION RESULTS")
    best_lambdas = None
    best_interp_dev = float("inf")

    for lambdas in lambda_sets:
        l1, l2, l3 = lambdas
        train_pp = model.perplexity_interpolation(train, l1, l2, l3)
        dev_pp = model.perplexity_interpolation(dev, l1, l2, l3)
        print(f"lambdas = {lambdas}")
        print(f"train perplexity: {format_pp(train_pp)}")
        print(f"dev perplexity: {format_pp(dev_pp)}")
        print()

        if dev_pp < best_interp_dev:
            best_interp_dev = dev_pp
            best_lambdas = lambdas

    best_l1, best_l2, best_l3 = best_lambdas
    best_test_pp = model.perplexity_interpolation(test, best_l1, best_l2, best_l3)
    print(f"Best lambdas based on dev: {best_lambdas}")
    print(f"test perplexity with best lambdas: {format_pp(best_test_pp)}")
    print()
    
if __name__ == "__main__":
    main()
