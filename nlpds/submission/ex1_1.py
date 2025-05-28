from typing import Collection, TypeGuard
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from numpy.typing import NDArray




from nlpds.abc.ex1_1 import AbstractLanguageClassifier, AbstractNGramFeaturizer, NGram

type BiGram = NGram


def is_bi_gram(s: str) -> TypeGuard[BiGram]:
    """
    Type guard for bi-grams
    """
    return len(s) == 2


class BiGramFeaturizer(AbstractNGramFeaturizer):
    def __init__(self,vocabulary: Collection[BiGram]):
        super().__init__(vocabulary)
        # Create a dict to keep track of count per bigram
        self._vocab_index = {gram: idx for idx, gram in enumerate(vocabulary)}

    
    def n_grams(self, sentence:str)->list[str]:
        """Generate n-grams from a sentence.
        Returns a list of `NGram`s (a type alias for `str`).
        If the sentence contains no valid n-grams, returns an empty list.
        """

        # Cleaning data
        sentence = sentence.lower() # lowert case letters
        sentence = re.sub(r"[^a-z #]", "", sentence) # removing ever char expcet: "a-z #"

        # Creating bi-grams
        bi_gram_list = [] 
        for i in range(len(sentence)-1):
            bi_gram_list.append(sentence[i:i+2])

        return bi_gram_list
    
    def featurize(self, sentence: str) -> NDArray:
        vector = np.zeros(len(self.vocab), dtype=np.float32)
        for gram in self.n_grams(sentence):
            if gram in self._vocab_index:
                index = self._vocab_index[gram]
                vector[index] += 1
        return vector
    # TODO: Document all methods


class BiGramLanguageClassifier(AbstractLanguageClassifier):
    def __init__(self):
        """
        Initialize the language classifier using logistic regression.

        The classifier is based on a scikit-learn LogisticRegression model.
        The 'sage' solver is chosen since we have a rather big dataset:
        900k en sentences and 900k de sentences
        """
        super().__init__()
        self.model = LogisticRegression(solver="saga",verbose=1, max_iter=10)

    def fit(self, features: NDArray, labels: NDArray) -> NDArray:
        """
        Train the logistic regression model on the given feature matrix and labels.

        Parameters:
            features (NDArray): A 2D array of shape (n_samples, n_features) representing
                                the Bi-gram frequency vectors for each text.
            labels (NDArray): A 1D array of shape (n_samples,) containing the true labels
                              (e.g., 'deu' or 'eng') for each sample.

        Returns:
            NDArray: The predicted labels for the training data.
        """
        # Fit  model to the input data
        self.model.fit(features, labels)
        # Return predictions on the same training data
        return self.model.predict(features)

    def predict(self, features: NDArray) -> NDArray:
        """
        Predict the labels for the given feature matrix.

        Parameters:
            features (NDArray): A 2D array of shape (n_samples, n_features) representing
                                the Bi-gram frequency vectors for each text.

        Returns:
            NDArray: The predicted labels for each sample.
        """
        return self.model.predict(features)

    def evaluate(self, features: NDArray, labels: NDArray) -> float:
        """
        Evaluate the model's accuracy on the given test data.

        Parameters:
            features (NDArray): A 2D array of shape (n_samples, n_features) representing
                                the Bi-gram frequency vectors for the test samples.
            labels (NDArray): A 1D array of shape (n_samples,) containing the true labels.

        Returns:
            float: The accuracy score (fraction of correct predictions).
        """
        # Predict labels for the input features:
        # Call the internal predict method (which wraps model.predict)
        predictions = self.predict(features)
        # Compute and return accuracy
        return accuracy_score(labels, predictions)
    # TODO: Document all methods


if __name__ == "__main__":
    pass