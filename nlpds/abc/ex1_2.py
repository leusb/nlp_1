from abc import abstractmethod
from typing import Collection

from torch import Tensor, nn

from .ex1_1 import NGram


class AbstractNGramEmbedding(nn.Module):
    def __init__(self, vocabulary: Collection[NGram], embedding_dim: int):
        self._vocabulary: dict[NGram, int] = {
            ngram: idx
            for idx, ngram in enumerate(
                sorted(
                    set(vocabulary),
                    # sort vocabulary by n-gram length, then alphabetically
                    key=lambda ngram: chr(32 + len(ngram)) + ngram,
                )
            )
        }
        self._vocabulary_inv: dict[int, NGram] = {
            v: k for k, v in self._vocabulary.items()
        }
        self._embedding_dim = embedding_dim

    @property
    def vocab(self) -> Collection[NGram]:
        """Get the vocabulary of n-grams."""
        return self._vocabulary.keys()

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self._embedding_dim

    @abstractmethod
    def embed_target(self, ngrams: Tensor, *args, **kwargs) -> Tensor:
        """Generate target embeddings for the given n-grams."""
        raise NotImplementedError

    @abstractmethod
    def embed_context(self, ngrams: Tensor, *args, **kwargs) -> Tensor:
        """Generate context embeddings for the given n-grams."""
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        target_embs: Tensor,
        context_embs: Tensor,
        negative_embs: Tensor,
    ) -> Tensor:
        """Calculate the negative sampling objective function of the skip-gram model for the given embeddings."""
        raise NotImplementedError
