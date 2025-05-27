import string
import itertools
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def create_alphabet():
    """Creating an alphabet after Goldberg using " " and "#" (28x28)."""
    alphabet = list(string.ascii_lowercase) +[" ", "#"]

    return alphabet

def create_vocabulary(alphabet):
    """Creating all possible bigrams from alphabet."""
    vocab_iterator = itertools.product(alphabet,repeat=2)
    # Transform iterable to list of bigrams
    vocabulary  = ["".join(pair) for pair in vocab_iterator]
    return vocabulary


if __name__ == "__main__":
    alph = create_alphabet()
    print(alph)
    x = create_vocabulary(alph)
    print (len(x))