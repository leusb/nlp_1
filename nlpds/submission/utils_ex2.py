import itertools
import torch
import random
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


def create_ngram_vocabulary(alphabet: list[str], n: int = 3) -> list[str]:
    """
    Generate a vocabulary consisting of all possible n-grams over the given alphabet.

    Args:
        alphabet (list[str]): List of characters to construct n-grams from.
        n (int): Length of each n-gram (default: 3).

    Returns:
        list[str]: List of all possible n-gram strings of length `n`.
    """
    return [''.join(gram) for gram in itertools.product(alphabet, repeat=n)]


def word_to_ngrams(word: str, n: int = 3) -> list[str]:
    """not used."""
    padded_word = f"#{word}#"
    return [padded_word[i:i + n] for i in range(len(padded_word) - n + 1)]


def generate_skipgram_triplets(
    tokens: list[str],
    window_size: int = 2,
    num_negative: int = 5
) -> list[tuple[str, str, list[str]]]:
    """
    Generate skip-gram training data as triplets (target, context, negatives).

    For each position `i` in `tokens`, the function considers words within
    `window_size` to the left and right as context words. For each (target, context)
    pair, `num_negative` negative samples are drawn uniformly from the vocabulary,
    excluding the target and the chosen context.

    Args:
        tokens (list[str]): List of tokens (words) in the corpus or sentence.
        window_size (int): Number of words to consider on each side for context (default: 2).
        num_negative (int): Number of negative samples to generate per (target, context) pair (default: 5).

    Returns:
        list[tuple[str, str, list[str]]]:
            A list of triples where each triple is:
                (target_word, context_word, [negative_word_1, ..., negative_word_k]).
    """
    vocab = list(set(tokens))
    triplets: list[tuple[str, str, list[str]]] = []

    for i, target in enumerate(tokens):
        # Determine context word indices within the window (excluding the target index)
        left_indices = list(range(max(0, i - window_size), i))
        right_indices = list(range(i + 1, min(len(tokens), i + window_size + 1)))
        context_indices = left_indices + right_indices

        for j in context_indices:
            context = tokens[j]
            negatives: list[str] = []
            # Sample negative words until we have `num_negative`, ensuring they are not target or context
            while len(negatives) < num_negative:
                neg = random.choice(vocab)
                if neg != target and neg != context:
                    negatives.append(neg)
            triplets.append((target, context, negatives))

    return triplets


def ngrams_to_ids(ngrams: list[str], vocab_dict: dict[str, int]) -> list[int]:
    """
    Convert a list of n-gram strings to their corresponding integer IDs.

    Args:
        ngrams (list[str]): List of n-gram substrings.
        vocab_dict (dict[str, int]): Mapping from n-gram string to integer ID.

    Returns:
        list[int]: List of integer IDs for n-grams present in `vocab_dict`.
    """
    return [vocab_dict[ng] for ng in ngrams if ng in vocab_dict]


def word_to_ids(word: str, vocab_dict: dict[str, int], n: int = 3) -> torch.Tensor:
    """
    Convert a word into a tensor of n-gram IDs. If no n-gram is found, return a tensor containing [0].

    Args:
        word (str): The input word.
        vocab_dict (dict[str, int]): Mapping from n-gram string to integer ID.
        n (int): Length of each n-gram (default: 3).

    Returns:
        torch.Tensor: 1D LongTensor of n-gram IDs (or tensor([0]) if no n-gram matches).
    """
    ids = ngrams_to_ids(word_to_ngrams(word, n), vocab_dict)
    # If the word has no n-gram in vocab_dict, return [0] as a fallback ID
    return torch.tensor(ids if ids else [0], dtype=torch.long)


def batch_words_to_padded_tensor(
    words: list[str],
    vocab_dict: dict[str, int],
    n: int = 3
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a batch of words into a flattened tensor of n-gram IDs plus an offsets tensor
    for use with EmbeddingBag.

    Args:
        words (list[str]): List of words in the batch.
        vocab_dict (dict[str, int]): Mapping from n-gram string to integer ID.
        n (int): Length of each n-gram (default: 3).

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - flat_tensor (torch.Tensor): 1D LongTensor containing all n-gram IDs for the batch, concatenated.
            - offsets (torch.Tensor): 1D LongTensor of length (batch_size + 1) where each entry marks
              the start index in `flat_tensor` for a new word. The final element equals total number of IDs.
    """
    sequences: list[torch.Tensor] = [word_to_ids(w, vocab_dict, n) for w in words]
    lengths: list[int] = [len(seq) for seq in sequences]
    # Compute offsets: offsets[i] = sum of lengths of sequences before index i
    offsets_list = [0] + list(torch.cumsum(torch.tensor(lengths[:-1]), dim=0).tolist())
    flat_tensor = torch.cat(sequences)
    return flat_tensor, torch.tensor(offsets_list, dtype=torch.long)


def triplets_to_tensors(
    triplets: list[tuple[str, str, list[str]]],
    vocab_dict: dict[str, int],
    n: int = 3
) -> tuple[
    torch.Tensor, torch.Tensor,  # target: flat IDs, offsets
    torch.Tensor, torch.Tensor,  # context: flat IDs, offsets
    torch.Tensor, torch.Tensor,  # negatives: flat IDs, offsets
    int                          # negatives per target
]:
    """
    Convert a list of skip-gram triplets into tensors suitable for model input.

    Args:
        triplets (list[tuple[str, str, list[str]]]): List of (target, context, negatives) triples.
        vocab_dict (dict[str, int]): Mapping from n-gram string to integer ID.
        n (int): Length of each n-gram (default: 3).

    Returns:
        tuple:
            - t_tensor (torch.Tensor): 1D LongTensor of flattened n-gram IDs for all target words.
            - t_offsets (torch.Tensor): 1D LongTensor of length (batch_size + 1) for targets.
            - c_tensor (torch.Tensor): 1D LongTensor of flattened n-gram IDs for all context words.
            - c_offsets (torch.Tensor): 1D LongTensor of length (batch_size + 1) for contexts.
            - n_tensor (torch.Tensor): 1D LongTensor of flattened n-gram IDs for all negative words.
            - n_offsets (torch.Tensor): 1D LongTensor of length (batch_size*num_negative + 1) for negatives.
            - num_neg (int): Number of negative samples per (target, context) pair.
    """
    targets = [t for t, _, _ in triplets]
    contexts = [c for _, c, _ in triplets]
    negatives = [neg for _, _, negs in triplets for neg in negs]

    # Convert target words to flattened IDs and offsets
    t_tensor, t_offsets = batch_words_to_padded_tensor(targets, vocab_dict, n)
    # Convert context words to flattened IDs and offsets
    c_tensor, c_offsets = batch_words_to_padded_tensor(contexts, vocab_dict, n)
    # Convert negative words to flattened IDs and offsets
    n_tensor, n_offsets = batch_words_to_padded_tensor(negatives, vocab_dict, n)

    # Compute number of negatives per target
    num_neg = len(negatives) // len(triplets)
    return t_tensor, t_offsets, c_tensor, c_offsets, n_tensor, n_offsets, num_neg


def get_similar_words(
    query_word: str,
    vocabulary: list[str],
    model,
    ngram_to_id: dict[str, int],
    top_k: int = 5,
    n: int = 3
) -> list[tuple[str, float]]:
    """
    Find the `top_k` words in `vocabulary` that are most similar to `query_word`
    based on cosine similarity of their embeddings.

    Args:
        query_word (str): The word to query similarity for.
        vocabulary (list[str]): List of candidate words to compare against.
        model: Trained SkipGramEmbedding model with methods embed_target.
        ngram_to_id (dict[str, int]): Mapping from n-gram string to integer ID.
        top_k (int): Number of top similar words to return (default: 5).
        n (int): Length of each n-gram used by the model (default: 3).

    Returns:
        list[tuple[str, float]]: List of (word, similarity_score) sorted in descending order of similarity.
    """
    def _word_to_ids_single(word: str) -> torch.Tensor:
        """
        Helper to convert a single word string into a 2D tensor of n-gram IDs.
        Returns shape (1, num_ngrams) so EmbeddingBag can process it as a batch of size 1.
        """
        padded = f"#{word}#"
        ngrams = [padded[i:i + n] for i in range(len(padded) - n + 1)]
        ids = [ngram_to_id[ng] for ng in ngrams if ng in ngram_to_id]
        return torch.tensor(ids if ids else [0], dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        # Convert query word to IDs and compute its embedding
        query_ids = _word_to_ids_single(query_word)
        query_emb = model.embed_target(query_ids, None).squeeze(0)

        similarity_list: list[tuple[str, float]] = []
        for word in vocabulary:
            word_ids = _word_to_ids_single(word)
            # Compute embedding for each candidate word
            word_emb = model.embed_target(word_ids, None).squeeze(0)
            # Cosine similarity between query and candidate embeddings
            sim_score = F.cosine_similarity(query_emb, word_emb, dim=0).item()
            similarity_list.append((word, sim_score))

        # Sort candidates by similarity in descending order and return top_k
        return sorted(similarity_list, key=lambda x: x[1], reverse=True)[:top_k]


if __name__ == "__main__":
    pass