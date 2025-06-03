from typing import Collection, Optional
from torch import nn, Tensor
from nlpds.abc.ex1_1 import NGram
from nlpds.abc.ex1_2 import AbstractNGramEmbedding


class SkipGramEmbedding(AbstractNGramEmbedding):
    """
    Skip-gram embedding model using n-gram subword information.

    Args:
        vocabulary (Collection[NGram]): Collection of all ngrams in the vocabulry.
        embedding_dim (int): Dimension of each resulting word embedding.
    """

    def __init__(
        self,
        vocabulary: Collection[NGram],
        embedding_dim: int,
    ):
        nn.Module.__init__(self)  # explicitly call base nn.Module constructor

        super().__init__(vocabulary, embedding_dim)

        vocab_size = len(self._vocabulary)

        # Two separate EmbeddingBag layers: one for trget embeddings, one for context embeddings.
        # EmbeddingBag(mode="mean") computes the mean of all n-gram  ectors for a given word.
        self.target_embeddings = nn.EmbeddingBag(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            mode="mean",
            sparse=False
        )
        self.context_embeddings = nn.EmbeddingBag(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            mode="mean",
            sparse=False
        )

    def embed_target(self, ngrams: Tensor, offsets: Optional[Tensor] = None) -> Tensor:
        """
        Generate target embeddings for the given n-grams.

        Args:
            ngrams (Tensor): 1D LongTensor of shape (sum_of_all_target_n_grams,)
            containing  the flatend indices of n-grams for each target word in the batch.
            offsets (Optional[Tensor]): 1D LongTensor of shape (batch_size + 1,)
            where each entry indicates the start index in `ngrams` for a new word.

        Returns:
            Tensor: 2D FloatTensor of shape (batch_size, embedding_dim) containing
                the averaged embeddings for each target word.
        """
        # use EmbeddingBag to get mean of subword embeddings per word
        return self.target_embeddings(ngrams, offsets)

    def embed_context(self, ngrams: Tensor, offsets: Optional[Tensor] = None) -> Tensor:
        """
        Generate context embeddings for the given n-grams.

        Args:
            ngrams (Tensor): 1D LongTensor of shape (sum_of_all_context_n_grams,)
            containing the flattened indices of n-grams for each context word in the batch.
            offsets (Optional[Tensor]): 1D LongTensor of shape (batch_size + 1,)
            where each entry indicates the start index in `ngrams` for a new word.

        Returns:
            Tensor: 2D FloatTnsor of shape (batch_size, embedding_dim) containing
            the averaged embeddings for each context word.
        """
        # context_embeddings layer also uses mean pooling of n-gram vectors
        return self.context_embeddings(ngrams, offsets)

    def forward(
        self,
        target_embs: Tensor,     # (batch_size, embedding_dim)
        context_embs: Tensor,    # (batch_size, embedding_dim)
        negative_embs: Tensor    # (batch_size, num_neg, embedding_dim)
    ) -> Tensor:
        """
        Calculate the negative sampling objective function of the skip-gram model for the given embeddings.

        Arg:
            target_embs (Tensor): FloatTensor of shape (batch_size, embedding_dim)
            containing embeddings of target words.
            context_embs (Tensor): FloatTensor of shape (batch_size, embedding_dim)
            containing embeddings of corresponding context words.
            negative_embs (Tensor): FloatTensor of shape (batch_size, num_neg, embedding_dim)
            containing embeddings of negative-sampled words for each target.

        Returns:
            Tensor: Scalar FloatTensor representing the average negative sampling loss.
        """
        # compute dot product between each target and its true context
        pos_logits = (target_embs * context_embs).sum(dim=1)  # shape: (batch_size,)

        # compute dot product between each target and its negative samples
    
        neg_logits = (target_embs.unsqueeze(1) * negative_embs).sum(dim=2)

        # positive loss term
        pos_loss = nn.functional.logsigmoid(pos_logits)
        # negative loss term
        neg_loss = nn.functional.logsigmoid(-neg_logits).sum(dim=1)

        # final loss: average negative sum over the batch
        loss = -(pos_loss + neg_loss).mean()
        return loss
