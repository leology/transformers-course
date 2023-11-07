from core import count_ngrams, build_ngram_model
from data import sample1, sample2


def run_tests():
    assert len(count_ngrams([sample1, sample2], n=3)) == 28
    assert len(count_ngrams([sample1, sample2], n=3)["<BOS>", "<BOS>"]) == 2
    assert count_ngrams([sample1, sample2], n=3)["<BOS>", "<BOS>"]["It"] == 1
    assert count_ngrams([sample1, sample2], n=3)["<BOS>", "<BOS>"]["How"] == 1
    assert count_ngrams([sample1, sample2], n=3)["my", "dear"]["Watson"] == 2
    assert len(count_ngrams([sample1, sample2], n=2)) == 24
    assert len(count_ngrams([sample1, sample2], n=2)["<BOS>",]) == 2
    assert count_ngrams([sample1, sample2], n=2)["<BOS>",]["It"] == 1
    assert count_ngrams([sample1, sample2], n=2)["<BOS>",]["How"] == 1
    assert count_ngrams([sample1, sample2], n=2)["dear",]["Watson"] == 2
    assert build_ngram_model([sample1, sample2], n=3)["<BOS>", "<BOS>"]["It"] == 0.5
    assert build_ngram_model([sample1, sample2], n=3)["<BOS>", "<BOS>"]["How"] == 0.5
    assert build_ngram_model([sample1, sample2], n=3)["my", "dear"]["Watson"] == 1.0
    assert build_ngram_model([sample1, sample2], n=2)["<BOS>",]["It"] == 0.5
    assert build_ngram_model([sample1, sample2], n=2)["<BOS>",]["How"] == 0.5
    assert build_ngram_model([sample1, sample2], n=2)["dear",]["Watson"] == 1.0
