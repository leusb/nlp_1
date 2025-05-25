from typing import Collection, TypeGuard

from nlpds.abc.ex1_1 import AbstractLanguageClassifier, AbstractNGramFeaturizer, NGram

type BiGram = NGram


def is_bi_gram(s: str) -> TypeGuard[BiGram]:
    """
    Type guard for bi-grams
    """
    return len(s) == 2


class BiGramFeaturizer(AbstractNGramFeaturizer):
    def __init__(
        self,
        vocabulary: Collection[BiGram],
        # ...
    ):
        super().__init__(vocabulary)

        # TODO: Implement the constructor
        raise NotImplementedError

    # TODO: Implement all abstract methods from AbstractBiGramFeaturizer
    # TODO: Document all methods


class BiGramLanguageClassifier(AbstractLanguageClassifier):
    def __init__(
        self,
        # ...
    ):
        super().__init__()

        # TODO: Implement the constructor

    # TODO: Implement all abstract methods from AbstractLanguageClassifier
    # TODO: Document all methods
