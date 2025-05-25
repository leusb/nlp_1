from abc import ABC, abstractmethod
from typing import Collection, Iterable

import numpy as np
from numpy.typing import NDArray

type NGram = str


class AbstractNGramFeaturizer(ABC):
    def __init__(self, vocabulary: Collection[NGram]):
        """Create a n-gram generator from a vocabulary of valid n-grams."""
        self._vocabulary = vocabulary

    @property
    def vocab(self) -> Collection[NGram]:
        """Get the vocabulary of n-grams."""
        return self._vocabulary

    def __len__(self) -> int:
        """Return the number of n-grams in the vocabulary."""
        return len(self._vocabulary)

    @abstractmethod
    def n_grams(self, sentence: str) -> list[NGram]:
        """Generate n-grams from a sentence.
        Returns a list of `NGram`s (a type alias for `str`).
        If the sentence contains no valid n-grams, returns an empty list.
        """
        raise NotImplementedError

    @abstractmethod
    def featurize(self, sentence: str) -> NDArray:
        """Generate a n-gram-frequency feature vector for a sentence."""
        raise NotImplementedError

    def __call__(self, sentences: str | Iterable[str]) -> NDArray:
        """Convert one or multiple sentences into feature arrays.
        Calls `self.featurize` internally.
        """
        if isinstance(sentences, str):
            return self.featurize(sentences)
        else:
            try:
                return np.array([self.featurize(sent) for sent in sentences])
            except TypeError as e:
                if "is not iterable" in str(e):
                    raise ValueError(
                        "Arguments must be a string or a list of strings!"
                    ) from e
                else:
                    raise e


type Accuracy = float


class AbstractLanguageClassifier:
    @abstractmethod
    def fit(self, features: NDArray, labels: NDArray) -> NDArray:
        """Train the classifier on the given data (featurized sentences)."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, features: NDArray) -> NDArray:
        """Predict the language for the given featurized sentences."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, features: NDArray, labels: NDArray) -> Accuracy:
        """Evaluate the performance of the classifier.
        Should return the accuracy of the predictions under the given labels."""
        raise NotImplementedError
