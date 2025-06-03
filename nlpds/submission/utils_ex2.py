import itertools
import torch
import random
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F



def create_ngram_vocabulary(alphabet: list[str], n: int = 3) -> list[str]:
    """Erzeugt ein Vokabular aus allen möglichen N-Grammen im Alphabet."""
    return [''.join(gram) for gram in itertools.product(alphabet, repeat=n)]

def word_to_ngrams(word: str, n: int = 3) -> list[str]:
    """Fügt '#' Padding hinzu und erzeugt N-Gramme für ein Wort."""
    word = f"#{word}#"
    return [word[i:i+n] for i in range(len(word) - n + 1)]


def generate_skipgram_triplets(tokens: list[str], window_size: int = 2, num_negative: int = 5) -> list[tuple[str, str, list[str]]]:
    """
    Erzeugt Skip-Gram-Daten: Triplets (target, context, negatives)
    """
    vocab = list(set(tokens))
    triplets = []
    for i, target in enumerate(tokens):
        context_range = list(range(max(0, i - window_size), i)) + list(range(i + 1, min(len(tokens), i + window_size + 1)))
        for j in context_range:
            context = tokens[j]
            negatives = []
            while len(negatives) < num_negative:
                neg = random.choice(vocab)
                if neg != target and neg != context:
                    negatives.append(neg)
            triplets.append((target, context, negatives))
    return triplets

def ngrams_to_ids(ngrams: list[str], vocab_dict: dict[str, int]) -> list[int]:
    return [vocab_dict[ng] for ng in ngrams if ng in vocab_dict]

def word_to_ids(word: str, vocab_dict: dict[str, int], n: int = 3) -> torch.Tensor:
    ids = ngrams_to_ids(word_to_ngrams(word, n), vocab_dict)
    return torch.tensor(ids if ids else [0], dtype=torch.long)

def batch_words_to_padded_tensor(words: list[str], vocab_dict: dict[str, int], n: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """Gibt padded Tensor (B × T) und Offsets für EmbeddingBag zurück"""
    sequences = [word_to_ids(w, vocab_dict, n) for w in words]
    lengths = [len(seq) for seq in sequences]
    offsets = [0] + list(torch.cumsum(torch.tensor(lengths[:-1]), dim=0))
    flat = torch.cat(sequences)
    return flat, torch.tensor(offsets, dtype=torch.long)

def triplets_to_tensors(
    triplets: list[tuple[str, str, list[str]]],
    vocab_dict: dict[str, int],
    n: int = 3
) -> tuple[
    torch.Tensor, torch.Tensor,  # target
    torch.Tensor, torch.Tensor,  # context
    torch.Tensor, torch.Tensor,  # negatives (flat), offsets
    int                          # negatives per target
]:
    targets = [t for t, _, _ in triplets]
    contexts = [c for _, c, _ in triplets]
    negatives = [neg for _, _, negs in triplets for neg in negs]

    t_tensor, t_offsets = batch_words_to_padded_tensor(targets, vocab_dict, n)
    c_tensor, c_offsets = batch_words_to_padded_tensor(contexts, vocab_dict, n)
    n_tensor, n_offsets = batch_words_to_padded_tensor(negatives, vocab_dict, n)

    num_neg = len(negatives) // len(triplets)

    return t_tensor, t_offsets, c_tensor, c_offsets, n_tensor, n_offsets, num_neg

def get_similar_words(query_word: str, vocabulary: list[str], model, ngram_to_id: dict[str, int], top_k: int = 5, n: int = 3):
    """
    Findet die `top_k` Wörter im Vokabular, die dem `query_word` am ähnlichsten sind
    (nach Cosinus-Ähnlichkeit).
    """
    def word_to_ids(word: str) -> torch.Tensor:
        ngrams = [f"#{word}#"[i:i+n] for i in range(len(f"#{word}#") - n + 1)]
        ids = [ngram_to_id[ng] for ng in ngrams if ng in ngram_to_id]
        return torch.tensor(ids if ids else [0]).unsqueeze(0)

    with torch.no_grad():
        query_ids = word_to_ids(query_word)
        if query_ids.size(1) == 0:
            return []

        query_emb = model.embed_target(query_ids, None).squeeze(0)

        results = []
        for word in vocabulary:
            word_ids = word_to_ids(word)
            if word_ids.size(1) == 0:
                continue
            word_emb = model.embed_target(word_ids, None).squeeze(0)
            sim = F.cosine_similarity(query_emb, word_emb, dim=0).item()
            results.append((word, sim))

        return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
    
    if __name__ =="__main__":
        pass