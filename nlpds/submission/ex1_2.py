from typing import Collection
from torch import nn, Tensor
from nlpds.abc.ex1_1 import NGram
from nlpds.abc.ex1_2 import AbstractNGramEmbedding


class SkipGramEmbedding(AbstractNGramEmbedding):
    def __init__(
        self,
        vocabulary: Collection[NGram],
        embedding_dim: int,
    ):
        nn.Module.__init__(self)  # ← 🔧 explizit aufrufen!

        super().__init__(vocabulary, embedding_dim)

        vocab_size = len(self._vocabulary)

        # Zwei separate Embedding-Matrizen: target und context
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

    def embed_target(self, ngrams: Tensor, offsets: Tensor = None) -> Tensor:
        """
        Erzeugt Target-Embeddings als Mittelwert ihrer N-Gramm-Embeddings.
        """
        return self.target_embeddings(ngrams, offsets)

    def embed_context(self, ngrams: Tensor, offsets: Tensor = None) -> Tensor:
        """
        Erzeugt Context-Embeddings als Mittelwert ihrer N-Gramm-Embeddings.
        """
        return self.context_embeddings(ngrams, offsets)

    def forward(
        self,
        target_embs: Tensor,     # (batch_size, embedding_dim)
        context_embs: Tensor,    # (batch_size, embedding_dim)
        negative_embs: Tensor    # (batch_size, num_neg, embedding_dim)
    ) -> Tensor:
        """
        Berechnet den Skip-Gram-Loss mit negativer Sampling:
        - Maximiert Dot-Produkt von (target, context)
        - Minimiert Dot-Produkt von (target, negative samples)
        """
        # Positive Logits: target • context
        pos_logits = (target_embs * context_embs).sum(dim=1)  # (batch,)

        # Negative Logits: target • negatives (für jedes negative sample)
        neg_logits = (target_embs.unsqueeze(1) * negative_embs).sum(dim=2)  # (batch, num_neg)

        # Loss: -log(σ(pos)) - sum(log(σ(-neg)))
        pos_loss = nn.functional.logsigmoid(pos_logits)
        neg_loss = nn.functional.logsigmoid(-neg_logits).sum(dim=1)

        return -(pos_loss + neg_loss).mean()
