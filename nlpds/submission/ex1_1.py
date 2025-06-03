from typing import Collection, TypeGuard, List
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from numpy.typing import NDArray

from nlpds.abc.ex1_1 import AbstractLanguageClassifier, AbstractNGramFeaturizer, NGram

# Define a type alias for bi-grams (two-character n-grams)
type BiGram = NGram


def is_bi_gram(s: str) -> TypeGuard[BiGram]:
    """
    Type gard for bi-grams.

    Parameters:
        s (str): A string to check.

    Returns:
        bool: True if the string has length exactly 2, indicating a bi-gram.
    """
    return len(s) == 2


class BiGramFeaturizer(AbstractNGramFeaturizer):
    def __init__(self, vocabulary: Collection[BiGram]) -> None:
        super().__init__(vocabulary)
        # Create a dict to keep track of count per bigram
        self._vocab_index: dict[BiGram, int] = {gram: idx for idx, gram in enumerate(vocabulary)}

    def n_grams(self, sentence: str) -> List[BiGram]:
        """
        Generate n-grams from a sentence.
        Returns a list of `NGram`s (a type alias for `str`).
        If the sentence contains no valid n-grams, returns an empty list.

        Parameters:
            sentence (str): The input sentence from which to extract bi-grams.

        Returns:
            List[BiGram]: A list of extracted bi-grams.
        """

        # Cleaning data
        sentence = sentence.lower()  # lower case leters
        sentence = re.sub(r"[^a-z #]", "", sentence)  # removing every char except: "a-z #"

        # Creating bi-grams
        bi_gram_list: List[BiGram] = []
        for i in range(len(sentence) - 1):
            bi_gram_list.append(sentence[i : i + 2])

        return bi_gram_list

    def featurize(self, sentence: str) -> NDArray:
        """
        Generate a n-gram-frequency feature vector for a sentence.

        Parameters:
            sentence (str): The input sentence to featurize.

        Returns:
            NDArray: A 1D NumPy array of shape (vocab_size,) where each entry
                    corresponds to the frequency of the matching bi-gram.
        """

        vector: NDArray = np.zeros(len(self.vocab))  # verwendet Standardfloat (float64)
        for gram in self.n_grams(sentence):
            if gram in self._vocab_index:
                index = self._vocab_index[gram]
                vector[index] += 1.0
        return vector


class BiGramLanguageClassifier(AbstractLanguageClassifier):
    def __init__(self) -> None:
        """
        Initialize the language classifier using logistic regression.

        The classifier is based on a scikit-learn LogisticRegression model.
        The 'sage' solver is chosen since we have a rather big dataset:
        900k en sentences and 900k de sentences
        """
        super().__init__()
        self.model: LogisticRegression = LogisticRegression(solver="saga", verbose=1, max_iter=10)

    def fit(self, features: NDArray, labels: NDArray) -> NDArray:
        """
        Train the classifir on the giiven data (featurized sentences).

        Parameters:
            features (NDArray): A 2D array of shape (n_samples, n_features) representing
                                            the Bigram frequency vectors for each text
            labels (NDArray): A 1D array of shape (n_samples,) containing the true labels
                                      (e.g., 'deu' or 'eng') for each sample.

        Returns:
            NDArra: The predicted labels for the training data.
        """
        # Fit model to the input data
        self.model.fit(features, labels)
        # Return predictions on the same training data
        return self.model.predict(features)

    def predict(self, features: NDArray) -> NDArray:
        """
        Predict the language for the given featurized sentences.

        Parameters:
            features (NDArray): A 2D aray of shape (n_samples, n_features) representing
                                            the Bi-gram frequency vectors for each text.

        Returns:
            NDArray: The predicted labels for each sample.
        """
        return self.model.predict(features)

    def evaluate(self, features: NDArray, labels: NDArray) -> float:
        """
        Evaluate the model's accuracy on the given test data.

        Parameters:
            featues (NDArray): A 2D array of shape (n_samples, n_features) representing
                                            the Bi-gram frequency vectors for the test samples.
            labels (NDArray): A 1D array of shape (n_samples,) containing the true labels.

        Returns:
            float: The accuracy score (fraction of correct predictions).
        """
        # Predict labels for the input features using internal predict method
        predictions = self.predict(features)
        # Compute and return accuracy
        return accuracy_score(labels, predictions)


if __name__ == "__main__":
    # Test functions for BiGramFeaturizer: 
    example_vocab = ['th', 'he', 'er', 're', 'e ', ' l', 'lo']  # sample bi-grams
    featurizer = BiGramFeaturizer(example_vocab)
    sample_sentence = "Hello there"
    print("Testing n_grams method:", featurizer.n_grams(sample_sentence))
    feature_vector = featurizer.featurize(sample_sentence)
    print(f"Bi-gram feature vector for '{sample_sentence}': {feature_vector}")

    # Test functions  BiGramLanguageClassifier: 
    # Create dummy features and labels
    some_featurs = np.vstack([feature_vector, feature_vector * 0])  # second sample has zeros
    some_labels = np.array(['eng', 'deu'])
    classifier = BiGramLanguageClassifier()
    train_preds = classifier.fit(some_featurs, some_labels)
    print(f"Training predictions: {train_preds}")
    test_preds = classifier.predict(some_featurs)
    print(f"Test predictions: {test_preds}")
    accuracy = classifier.evaluate(some_featurs, some_labels)
    print(f"Accuracy on dummy data: {accuracy}")
