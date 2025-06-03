from datasets import DatasetDict, concatenate_datasets, load_dataset
from collections import Counter
from typing import List
import torch
import random
from torch import optim


from nlpds.submission.ex1_1 import BiGramFeaturizer, BiGramLanguageClassifier
from nlpds.submission.utils_ex1 import *
from nlpds.submission.ex1_2 import SkipGramEmbedding
from nlpds.submission.utils_ex2 import *


# NOTE
# You may change any part of this file to fit your needs!
# Implement your main-functions here, add some for evaluation, etc.
# You can also add helper functions or classes to this file
# or create in the module, e.g. `nlpds.submission.utils`.
#
# Make sure that it is possible to run your code from this file.
# If there are multiple things to run, put them each behind the `if __name__ == "__main__"` guard.


def load_datasets() -> DatasetDict:
    dataset_deu: DatasetDict = load_dataset(
        "Texttechnologylab/leipzig-corpora-collection",
        "news_2024_1M",
        split="deu",
    ).train_test_split(0.1, shuffle=True)  # type: ignore

    dataset_eng = load_dataset(
        "Texttechnologylab/leipzig-corpora-collection",
        "news_2024_1M",
        split="eng",
    ).train_test_split(0.1, shuffle=True)  # type: ignore

    return DatasetDict(
        {
            "train": concatenate_datasets(
                [
                    dataset_deu["train"],
                    dataset_eng["train"],
                ]
            ),
            "test": concatenate_datasets(
                [
                    dataset_deu["test"],
                    dataset_eng["test"],
                ]
            ),
        }
    )

def load_deu_fra_datasets() -> DatasetDict:
    dataset_deu = load_dataset(
        "Texttechnologylab/leipzig-corpora-collection",
        "news_2024_1M",
        split="deu",
    ).train_test_split(0.1, shuffle=True)

    dataset_fra = load_dataset(
        "Texttechnologylab/leipzig-corpora-collection",
        "news_2024_1M",
        split="fra",
    ).train_test_split(0.1, shuffle=True)

    return DatasetDict({
        "train": concatenate_datasets([dataset_deu["train"], dataset_fra["train"]]),
        "test": concatenate_datasets([dataset_deu["test"], dataset_fra["test"]]),
    })

def load_datasets_multi(languages: List[str]) -> DatasetDict:
    """
    Lädt mehrere Sprachen und erstellt einen kombinierten Trainings- und Test-Datensatz.
    """
    datasets = [
        load_dataset(
            "Texttechnologylab/leipzig-corpora-collection",
            "news_2024_1M",
            split=lang,
        ).train_test_split(0.1, shuffle=True)  # type: ignore
        for lang in languages
    ]

    return DatasetDict({
        "train": concatenate_datasets([ds["train"] for ds in datasets]),
        "test": concatenate_datasets([ds["test"] for ds in datasets]),
    })


def run_ex1_1():
    datasets = load_datasets()

    # 1. Alphabet und Vokabular
    alphabet = create_alphabet()
    vocabulary = create_vocabulary(alphabet)

    # 2. Featurizer erzeugen
    featurizer = BiGramFeaturizer(vocabulary)

    # 3. Trainings- und Testdaten extrahieren
    train_texts = datasets["train"]["text"]
    train_labels = datasets["train"]["lang"]
    test_texts = datasets["test"]["text"]
    test_labels = datasets["test"]["lang"]

    # 4. Texte featurisieren
    train_features = featurizer(train_texts)
    test_features = featurizer(test_texts)

    # 5. Klassifikator trainieren und auswerten
    classifier = BiGramLanguageClassifier()
    classifier.fit(train_features, train_labels)
    accuracy = classifier.evaluate(test_features, test_labels)
    print("Test accuracy:", accuracy)

    # 6. Erweiterte Ergebnisdarstellung
    test_preds = classifier.predict(test_features)
    evaluate_and_display_results(test_labels, test_preds, label_names=("deu", "eng"))

def run_ex1_1_vocab_optimized():
    datasets = load_datasets()

    alphabet = create_alphabet()
    full_vocabulary = create_vocabulary(alphabet)
    full_featurizer = BiGramFeaturizer(full_vocabulary)

    # Trainingsdaten
    train_texts = datasets["train"]["text"]
    train_labels = datasets["train"]["lang"]
    test_texts = datasets["test"]["text"]
    test_labels = datasets["test"]["lang"]

    # Top-Bigrams extrahieren
    top_bigrams = get_top_bigrams(train_texts, full_featurizer, top_k=20)
    print(len(top_bigrams))

    # Featurizer mit reduziertem Vokabular
    reduced_featurizer = BiGramFeaturizer(top_bigrams)
    train_features = reduced_featurizer(train_texts)
    test_features = reduced_featurizer(test_texts)

    # Klassifikation
    classifier = BiGramLanguageClassifier()
    classifier.fit(train_features, train_labels)
    accuracy = classifier.evaluate(test_features, test_labels)
    print("Test accuracy (reduced vocabulary):", accuracy)

    test_preds = classifier.predict(test_features)
    evaluate_and_display_results(test_labels, test_preds, label_names=("deu", "eng"))


def run_ex1_1_minimal_vocab():
    print("Running Exercise 1.2 – minimal vocabulary search")

    # 1. Dataset laden
    datasets = load_datasets()
    train_texts = datasets["train"]["text"]
    train_labels = datasets["train"]["lang"]
    test_texts = datasets["test"]["text"]
    test_labels = datasets["test"]["lang"]

    # 2. Vollständiges Vokabular & Featurizer
    alphabet = create_alphabet()
    full_vocab = create_vocabulary(alphabet)
    featurizer_full = BiGramFeaturizer(full_vocab)

    # 3. Baseline-Accuracy mit vollem Vokabular
    X_train_full = featurizer_full(train_texts)
    X_test_full = featurizer_full(test_texts)
    clf_full = BiGramLanguageClassifier()
    clf_full.fit(X_train_full, train_labels)
    baseline_acc = clf_full.evaluate(X_test_full, test_labels)
    print(f"Baseline accuracy (full vocab): {baseline_acc:.4f}")

    # 4. Bi-Gramme nach Häufigkeit sortieren
    top_sorted_bigrams = get_top_bigrams(train_texts, featurizer_full, top_k=len(full_vocab))

    # 5. Reduktion: von 100 → 90 → 80 … bis Accuracy < 90%
    threshold = 0.9 * baseline_acc
    step = 1
    best_k = None
    for k in range(10, 0, -step):
        reduced_vocab = top_sorted_bigrams[:k]
        featurizer = BiGramFeaturizer(reduced_vocab)

        X_train = featurizer(train_texts)
        X_test = featurizer(test_texts)

        clf = BiGramLanguageClassifier()
        clf.fit(X_train, train_labels)
        acc = clf.evaluate(X_test, test_labels)
        print(f"k = {k:3d} → acc = {acc:.4f}")

        if acc >= threshold:
            best_k = k
        else:
            break  # abbrechen, sobald Accuracy < 90 %

    if best_k is None:
        print("No minimal k found with ≥ 90% accuracy.")
        return

    print(f"\n✅ Best k = {best_k} Bi-Gramme with acc ≥ 90% of baseline.")

    # 6. Finale Evaluation
    final_vocab = top_sorted_bigrams[:best_k]
    featurizer = BiGramFeaturizer(final_vocab)
    X_train = featurizer(train_texts)
    X_test = featurizer(test_texts)

    clf = BiGramLanguageClassifier()
    clf.fit(X_train, train_labels)
    acc = clf.evaluate(X_test, test_labels)
    print(f"Final accuracy (k={best_k}): {acc:.4f}")
    evaluate_and_display_results(test_labels, clf.predict(X_test), label_names=("deu", "eng"))

def run_ex1_3():
    datasets = load_deu_fra_datasets()

    # Angepasstes Alphabet (inkl. Akzente etc.)
    alphabet = create_french_german_alphabet()
    vocabulary = create_vocabulary(alphabet)

    featurizer = BiGramFeaturizer(vocabulary)

    train_texts = datasets["train"]["text"]
    train_labels = datasets["train"]["lang"]
    test_texts = datasets["test"]["text"]
    test_labels = datasets["test"]["lang"]

    train_features = featurizer(train_texts)
    test_features = featurizer(test_texts)

    classifier = BiGramLanguageClassifier()
    classifier.fit(train_features, train_labels)
    accuracy = classifier.evaluate(test_features, test_labels)
    print("Test accuracy (deu vs. fra):", accuracy)

    test_preds = classifier.predict(test_features)
    evaluate_and_display_results(test_labels, test_preds, label_names=("deu", "fra"))

def run_ex1_1_multi():
    print("Running Exercise 1.1 (Bonus) – Multi-class classification (deu, eng, fra)")

    # Lade 3 Sprachen
    datasets = load_datasets_multi(["deu", "eng", "fra"])

    # Vokabular
    alphabet = create_french_german_alphabet()  # deckt auch eng ab
    vocabulary = create_vocabulary(alphabet)
    featurizer = BiGramFeaturizer(vocabulary)

    # Daten vorbereiten
    train_texts = datasets["train"]["text"]
    train_labels = datasets["train"]["lang"]
    test_texts = datasets["test"]["text"]
    test_labels = datasets["test"]["lang"]

    train_features = featurizer(train_texts)
    test_features = featurizer(test_texts)

    # Klassifikation
    classifier = BiGramLanguageClassifier()
    classifier.fit(train_features, train_labels)
    accuracy = classifier.evaluate(test_features, test_labels)
    print("Test accuracy (deu, eng, fra):", accuracy)

    # Ergebnis darstellen
    test_preds = classifier.predict(test_features)
    evaluate_and_display_results(test_labels, test_preds, label_names=("deu", "eng", "fra"))




def run_ex1_2():
    # 1. Daten laden (z. B. nur Deutsch)
    dataset = load_dataset(
        "Texttechnologylab/leipzig-corpora-collection",
        "news_2024_1M",
        split="deu",
    )

    sentences = [s.split() for s in dataset["text"][:1000]]  # ⛔ begrenzt für Test!
    tokens = [tok for s in sentences for tok in s]
    vocab_set = set(tokens)

    # 2. N-Gramm-Vokabular
    alphabet = list("abcdefghijklmnopqrstuvwxyzäöüß #")  # oder create_alphabet()
    ngram_vocab = create_ngram_vocabulary(alphabet, n=3)
    ngram_to_id = {ng: idx for idx, ng in enumerate(ngram_vocab)}

    # 3. Modell
    embedding_dim = 50
    model = SkipGramEmbedding(ngram_vocab, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 4. Trainingsdaten (Triplets)
    triplets = []
    for s in sentences:
        triplets += generate_skipgram_triplets(s, window_size=2, num_negative=3)

    # 5. Training
    batch_size = 32
    model.train()
    for epoch in range(5):
        random.shuffle(triplets)
        for i in range(0, len(triplets), batch_size):
            batch = triplets[i:i+batch_size]
            if len(batch) < batch_size:
                continue

            t_tensor, t_offsets, c_tensor, c_offsets, n_tensor, n_offsets, num_neg = triplets_to_tensors(batch, ngram_to_id, n=3)

            target_emb = model.embed_target(t_tensor, t_offsets)
            context_emb = model.embed_context(c_tensor, c_offsets)
            neg_emb = model.context_embeddings(n_tensor, n_offsets).view(batch_size, num_neg, -1)

            loss = model(target_emb, context_emb, neg_emb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} – Loss: {loss.item():.4f}")
    
    # Modell evaluieren
    model.eval()
    query = "student"

    similar = get_similar_words(
        query_word=query,
        vocabulary=list(vocab_set),
        model=model,
        ngram_to_id=ngram_to_id,
        top_k=10,
        n=3
    )

    print(f"Ähnliche Wörter zu '{query}':")
    for word, score in similar:
        print(f"  {word:12s} → Cosinus: {score:.4f}")


def run_ex1_2_with_ngram_size(n: int = 3):
    print(f"\n### Starte Training mit n = {n} ###")

    # 1. Daten laden (z. B. nur Deutsch)
    dataset = load_dataset(
        "Texttechnologylab/leipzig-corpora-collection",
        "news_2024_1M",
        split="deu",
    )

    sentences = [s.split() for s in dataset["text"][:500]]  # etwas kürzer für Tests
    tokens = [tok for s in sentences for tok in s]
    vocab_set = set(tokens)

    # 2. N-Gramm-Vokabular
    alphabet = list("abcdefghijklmnopqrstuvwxyzäöüß #")
    ngram_vocab = create_ngram_vocabulary(alphabet, n)
    ngram_to_id = {ng: idx for idx, ng in enumerate(ngram_vocab)}

    # 3. Modell
    embedding_dim = 50
    model = SkipGramEmbedding(ngram_vocab, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 4. Trainingsdaten (Triplets)
    triplets = []
    for s in sentences:
        triplets += generate_skipgram_triplets(s, window_size=2, num_negative=3)

    # 5. Training
    batch_size = 32
    model.train()
    for epoch in range(5):
        random.shuffle(triplets)
        for i in range(0, len(triplets), batch_size):
            batch = triplets[i:i+batch_size]
            if len(batch) < batch_size:
                continue

            t_tensor, t_offsets, c_tensor, c_offsets, n_tensor, n_offsets, num_neg = triplets_to_tensors(batch, ngram_to_id, n)

            target_emb = model.embed_target(t_tensor, t_offsets)
            context_emb = model.embed_context(c_tensor, c_offsets)
            neg_emb = model.context_embeddings(n_tensor, n_offsets).view(batch_size, num_neg, -1)

            loss = model(target_emb, context_emb, neg_emb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} – Loss: {loss.item():.4f}")
    
    # Evaluation
    model.eval()
    query = "student"

    similar = get_similar_words(
        query_word=query,
        vocabulary=list(vocab_set),
        model=model,
        ngram_to_id=ngram_to_id,
        top_k=10,
        n=n
    )

    print(f"\nÄhnliche Wörter zu '{query}' (n = {n}):")
    for word, score in similar:
        print(f"  {word:12s} → Cosinus: {score:.4f}")


if __name__ == "__main__":
    # alphabet = create_alphabet()
    # vocabulary = create_vocabulary(alphabet)
    # featurizer = BiGramFeaturizer(vocabulary)
    # print(featurizer.n_grams("This is fine"))


    # print("Running Exercise 1 - Task 1")
    # run_ex1_1()

    # print("Running Exercise 1 - Task 1.2 (Optimized Vocabulary)")
    # run_ex1_1_multi()

    # print("Running Exercise 1 - Task 2")
    # run_ex1_2()
    run_ex1_2_with_ngram_size(4)
