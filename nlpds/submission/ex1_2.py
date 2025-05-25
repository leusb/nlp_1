from typing import Collection

from nlpds.abc.ex1_1 import NGram
from nlpds.abc.ex1_2 import AbstractNGramEmbedding


class SkipGramEmbedding(AbstractNGramEmbedding):
    def __init__(
        self,
        vocabulary: Collection[NGram],
        embedding_dim: int,
        # ...
    ):
        super().__init__(vocabulary, embedding_dim)

        # TODO: Implement the constructor

    # TODO: Implement all abstract methods from AbstractNGramEmbedding
    # TODO: Document all methods
