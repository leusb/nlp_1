import string
import itertools
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay, classification_report
from nlpds.submission.ex1_1 import BiGramFeaturizer, BiGramLanguageClassifier
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt



def create_alphabet():
    """Creating an alphabet after Goldberg using " " and "#" (28x28)."""
    alphabet = list(string.ascii_lowercase) +[" ", "#"]

    return alphabet

def create_french_german_alphabet():
    """Adjusted for exercise 1.3."""
    
    return list("abcdefghijklmnopqrstuvwxyzäöüßàâçéèêëîïôûùüÿœæ #")

def create_vocabulary(alphabet):
    """Creating all possible bigrams from alphabet."""
    vocab_iterator = itertools.product(alphabet,repeat=2)
    # Transform iterable to list of bigrams
    vocabulary  = ["".join(pair) for pair in vocab_iterator]
    return vocabulary


def evaluate_and_display_results(y_true, y_pred, label_names=("deu", "eng"), title="Confusion Matrix"):
    """
    Display a confusion matrix and print a classification report.

    Parameters:
        y_true (list or array): The true class labels.
        y_pred (list or array): The predicted class labels.
        label_names (tuple): The order of labels (for matrix rows/columns).
        title (str): Title for the confusion matrix plot.
    """
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=label_names)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(title)
    plt.show()

    # Classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=label_names))

def get_top_bigrams(texts: list[str], featurizer, top_k: int = 100) -> list[str]:
    """Find the top 100 Bigrams for Exercise 1.2."""
    counter = Counter()
    for sentence in texts:
        bigrams = featurizer.n_grams(sentence)
        counter.update(bigrams)
    return [bg for bg, _ in counter.most_common(top_k)]


if __name__ == "__main__":
    alph = create_alphabet()
    print(alph)
    x = create_vocabulary(alph)
    print (len(x))