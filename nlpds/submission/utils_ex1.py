import string
import itertools
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from nlpds.submission.ex1_1 import BiGramFeaturizer, BiGramLanguageClassifier
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


def create_alphabet() -> list[str]:
    """
    Creating an alphabet after Goldberg using " " and "#" (28x28).

    Returns:
        list[str]: A list of lowercase letters 'a' to 'z', a space ' ', and a pound sign '#'.
    """
    alphabet = list(string.ascii_lowercase) + [" ", "#"]
    return alphabet


def create_french_german_alphabet() -> list[str]:
    """
    Adjusted for exercise 1.3. Can also be used for English since subset propertiy.

    Returns:
        list[str]: A list of letters including German umlauts, French accents, and special chars whitespace and '#'.
    """
    return list("abcdefghijklmnopqrstuvwxyzäöüßàâçéèêëîïôûùüÿœæ #")


def create_vocabulary(alphabet: list[str]) -> list[str]:
    """
    Creating all possible bigrams from alpahbet.

    Parameters:
        alphabet (list[str]): A list of single-character strings representing the alphabet.

    Returns:
        list[str]: A list of all possible two-character strings (bigrams) from the input alphabet.
    """
    vocab_iterator = itertools.product(alphabet, repeat=2)
    # Transform iterable to list of bigrams
    vocabulary = ["".join(pair) for pair in vocab_iterator]
    return vocabulary


def evaluate_and_display_results(y_true: list[str], y_pred: list[str], label_names=("deu", "eng"), title="Confusion Matrix") -> None:
    """
    Display a confusion matrix and print a clasification report.

    Parameters:
        y_true (list or array): The true class labels.
        y_pred (list or array): The predicted class labels.
        label_names (tuple[str, ...]): The order of labels used in the confusion matrix.
        title (str): Title for the confusion matrix plot.

    Returns:
        None: This function displays a plot and prints text, it does not return a value.
    """
    # Confusion matrix
    matrix = confusion_matrix(y_true, y_pred, labels=label_names)
    show = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=label_names)
    show.plot(cmap="Blues", values_format="d")
    plt.title(title)
    plt.show()

    # Classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=label_names))


def get_top_bigrams(texts: list[str], featurizer, top_k: int = 100) -> list[str]:
    """
    Find the top-k bigrams for Exercise 1.2

    Parameters:
        texts (list[str]): A list of sentences to extract bigrams from.
        featurizer: An instance of BiGramFeaturizer or similar with a n_grams method.
        top_k (int): Number of top frequent bigrams to return (default 100).

    Returns:
        list[str]: A list of the top_k most frequent bigram strings.
    """
    counter = Counter()
    for sentence in texts:
        bigrams = featurizer.n_grams(sentence)
        counter.update(bigrams)
    # uncoimment following line to see a counter for bigrams (descending)
    # print(counter)
    return [bg for bg, _ in counter.most_common(top_k)]


if __name__ == "__main__":
    alph = create_alphabet()
    print(alph)
    x = create_vocabulary(alph)
    print(len(x))
