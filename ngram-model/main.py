from core import build_ngram_model
from data import sample1, sample2
from tests import run_tests

if __name__ == "__main__":
    run_tests
    model = build_ngram_model([sample1, sample1], 3)
    print(model)
