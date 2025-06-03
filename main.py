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
    """For Extra Exercise. Load multple languages."""
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

#############################################################
#Exercise 1
#############################################################
def run_ex1_1(show_extended_results=True):
    """
    Run Exercise 1.1: Train and evaluate a bi-gram language classifier on German vs. Engish.

    Steps:
      1. Load datasets (German/English ).
      2. Build the alphabet and full vocabulary of bi-grams.
      3. Instantiate a BiGramFeaturizer with the full vocabulary.
      4. Extract train/test texts and labels fro the datasets.
      5. Featurize both train and test sentences into bi-gram frequency vectors.
      6. Train aiGramLanguageClassifier on the training features.
      7. Evaluate the classifier on the  test features and print test accuracy.
      8. Perform extended result display (confusion matrix and classification report

    Note:
      This function uses create_alphabet(), create_vocabulary(), load_datasets(,
      BiGramFeaturizer, BiGramLanguageClassifier, and evaluate_and_display_results().
    """
    # load the German/English training and test data
    datasets = load_datasets()

    # 1. build alphabet and full bi-gram vocabulary
    alphabet = create_alphabet()
    vocabulary = create_vocabulary(alphabet)

    # 2. instantiate a bi-gram featurizer with the full vocabulary
    featurizer = BiGramFeaturizer(vocabulary)

    # 3. extract texts and labels for training and testing
    train_texts = datasets["train"]["text"]
    train_labels = datasets["train"]["lang"]
    test_texts = datasets["test"]["text"]
    test_labels = datasets["test"]["lang"]

    # 4. featurize the train/test sentences into bi-gram frequency vectors
    train_features = featurizer(train_texts)
    test_features = featurizer(test_texts)

    # 5. train the bi-gram language classifier and evaluate on test set
    classifier = BiGramLanguageClassifier()
    classifier.fit(train_features, train_labels)
    accuracy = classifier.evaluate(test_features, test_labels)
    print("Test accuracy:", accuracy)

     # 6. extended result display: confusion matrix and classification report. Toggle via parameter
    if show_extended_results:
        test_preds = classifier.predict(test_features)
        evaluate_and_display_results(test_labels, test_preds, label_names=("deu", "eng"))


def run_ex1_1_vocab_optimized(show_extended_results=True):
    """
    Run Exercise 1.1 (vocab-optimized): Train and evaluate using only top-k bi-grams.

    Steps:
      1. Load datasets (German/English).
      2. Build the full alphabet and vocabulary
      3. Featurize training texts with full vocabulary to obtain counts.
      4. Extract top 20 most frequent bi-grams from the training set.
      5. Instantiate a reduced featurizer with only the top-20 bi-grams 
      6. Featurize train/test texts with thereduced featurizer  .
      7. Train BiGramLanguageClassifier on reduced features.
      8. Evaluate on test set and print accuracy with reduced vocabulary.
      9. Display cnfusion matrix and classification report.

    Note:
      This function uses get_top_bigrams() to select the most frequent bi-grams.
      A smaller vocabulary may speed up training but can reduce accuracy.
    """
    # load the German/English datasets
    datasets = load_datasets()

    # build the alphabet and full bi-gram vocabulary
    alphabet = create_alphabet()
    full_vocabulary = create_vocabulary(alphabet)
    full_featurizer = BiGramFeaturizer(full_vocabulary)

    # extract train/test texts and labels
    train_texts = datasets["train"]["text"]
    train_labels = datasets["train"]["lang"]
    test_texts = datasets["test"]["text"]
    test_labels = datasets["test"]["lang"]

    # extract top-20 most frequent bi-grams from training texts
    top_bigrams = get_top_bigrams(train_texts, full_featurizer, top_k=20)
    print(len(top_bigrams))  # testing

    # instantiate a reduced featurizer with only the top-20 bi-grams
    reduced_featurizer = BiGramFeaturizer(top_bigrams)
    train_features = reduced_featurizer(train_texts)
    test_features = reduced_featurizer(test_texts)

    # train the classifier on reduced features and evaluate
    classifier = BiGramLanguageClassifier()
    classifier.fit(train_features, train_labels)
    accuracy = classifier.evaluate(test_features, test_labels)
    print("Test accuracy (reduced vocabulary):", accuracy)

    # extended result display: confusion matrix and classification report. Toggle via parameter
    if show_extended_results:
        test_preds = classifier.predict(test_features)
        evaluate_and_display_results(test_labels, test_preds, label_names=("deu", "eng"))


def run_ex1_1_minimal_vocab(show_extended_results=True, start_val = 10):
    """
    Run Exercise 1.2: Find the minimal bi-gram vocabulary size for ≥90% baseline accuracy.

    Steps:
      1. Load German/English datasets.
      2. Build full bi-gram vocabulary and featurizer.
      3. Compute baseline accuracy using the full vocabulary.
      4. Sort all bi-grams by frequency in descending order.
      5. Iteratively reduce vocablary size from 10 down to 1:
          a. Featurize with topk bi-grams.
          b. Train classifier and evaluate on test set.
          c. Stop when accuracy drops below 90% of the baseline.
      6. If found, report he best k and re-  evaluate with that vocabulary size.
    """
    print("Running Exercise 1.2:minimal vocabulary seearch") 

    # 1. load the German/English training and test sets
    datasets = load_datasets()
    train_texts = datasets["train"]["text"]
    train_labels = datasets["train"]["lang"]
    test_texts = datasets["test"]["text"]
    test_labels = datasets["test"]["lang"]

    # 2. build full bi-gram vocabulary and featurizer
    alphabet = create_alphabet()
    full_vocab = create_vocabulary(alphabet)
    featurizer_full = BiGramFeaturizer(full_vocab)

    # 3. compute baseline accuracy with full vocabulary
    print("Calculating Baseline accuracy for compariuson.")
    X_train_full = featurizer_full(train_texts)
    X_test_full = featurizer_full(test_texts)
    clf_full = BiGramLanguageClassifier()
    clf_full.fit(X_train_full, train_labels)
    baseline_acc = clf_full.evaluate(X_test_full, test_labels)
    print(f"Baseline accuracy (full vocab): {baseline_acc:.4f}")

    # 4. get all bi-grams sorted by frequency (descending)
    top_sorted_bigrams = get_top_bigrams(train_texts, featurizer_full, top_k=len(full_vocab))

    # 5. iterative vocabulary reduction: from k=10 → k=1
    threshold = 0.9 * baseline_acc
    best_k = None
    for k in range(start_val, 0, -1):
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
            break  # stop when accuracy falls below 90% of baseline

    if best_k is None:
        print("No minimal k found with ≥ 90% acc.")
        return

    print(f"Best k = {best_k} bi-grams with acc ≥ 90% of Baseline")

    # 6. final evaluation with best_k bi-grams
    final_vocab = top_sorted_bigrams[:best_k]
    featurizer = BiGramFeaturizer(final_vocab)
    X_train = featurizer(train_texts)
    X_test = featurizer(test_texts)

    classifier = BiGramLanguageClassifier()
    classifier.fit(X_train, train_labels)
    acc = classifier.evaluate(X_test, test_labels)
    print(f"Final accuracy (k={best_k}): {acc:.4f}")

    if show_extended_results:
        evaluate_and_display_results(test_labels, classifier.predict(X_test), label_names=("deu", "eng"))


def run_ex1_3(show_extended_results=True):
    """
    Run Exercise 1.3: Train and evaluate a bi-gram classifier on German vs. French.

    Steps:
      1. Load German/French datasets.
      2. Build an alphabet that includes accents and special characters.
      3. Create full bi-gram vocabulary frm the extendedalphabet. 
      4. Instantiate a BiGramFeaturizer with the extended vocabulary.
      5. Extract  train/test texts and labels (German/French).
      6. Featurize the sentences and train BiGramLanguageClassifier.
      7. Evaluate on test set andprint accuracy for deu vs. fa.
      8. Display confusion matrix and classification report

    Note:
      This function uses create_french_german_alphabet() to include accented letters.
    """
    # load German/French training and test sets
    datasets = load_deu_fra_datasets()

    # 1. build extended alphabet (including accents and umlauts)
    alphabet = create_french_german_alphabet()
    vocabulary = create_vocabulary(alphabet)

    # 2. instantiate a bi-gram featurizer with extended vocabulary
    featurizer = BiGramFeaturizer(vocabulary)

    # 3. extract train/test texts and labels
    train_texts = datasets["train"]["text"]
    train_labels = datasets["train"]["lang"]
    test_texts = datasets["test"]["text"]
    test_labels = datasets["test"]["lang"]

    # 4. featurize the train/test sentences
    train_features = featurizer(train_texts)
    test_features = featurizer(test_texts)

    # 5. train classifier and evaluate on test set
    classifier = BiGramLanguageClassifier()
    classifier.fit(train_features, train_labels)
    accuracy = classifier.evaluate(test_features, test_labels)
    print("Test accuracy (deu vs. fra):", accuracy)

    # 6. extended result display
    if show_extended_results:
        test_preds = classifier.predict(test_features)
        evaluate_and_display_results(test_labels, test_preds, label_names=("deu", "fra"))


def run_ex1_extra(show_extended_results=True):
    """
    Run Exercise 1.1 extra: Multi-class classification on German, English, and French.

    Steps:
      1. Load multi-language dataset containing Grman, English, and French.
      2. Build an extended alphabet to cover all three languages.
      3. Create ful bi-gram vocabulary from that alphabet. 
      4. Instantiate a BiGramFeaturizer with full vocbulary.
      5. Extract traintest texts and labels (deu, eng, fra).
      6. Featurize sentences and train BiGranmLnguageClassifier for 3 classes.
      7.  Evaluat on test set an print multi-class accuracy.
      8. Display confusion matrix and clasifcation report for all three l abels.

    Note:
      This is an optional extra exercise requiring multi-class logistic regression.
    """
    print("Running Exercise 1.1 (Extra: Multi-class classification (deu, eng, fra)")

    # 1. load datasets for three languages
    datasets = load_datasets_multi(["deu", "eng", "fra"])

    # 2. build extended alphabet that covers German, English, and French
    alphabet = create_french_german_alphabet()  # already includes basic English letters
    vocabulary = create_vocabulary(alphabet)
    featurizer = BiGramFeaturizer(vocabulary)

    # 3. extract train/test texts and labels for deu, eng, fra
    train_texts = datasets["train"]["text"]
    train_labels = datasets["train"]["lang"]
    test_texts = datasets["test"]["text"]
    test_labels = datasets["test"]["lang"]

    # 4. featurize sentences into bi-gram frequency vectors
    train_features = featurizer(train_texts)
    test_features = featurizer(test_texts)

    # 5. train a multi-class classifier and evaluate on test set
    classifier = BiGramLanguageClassifier()
    classifier.fit(train_features, train_labels)
    accuracy = classifier.evaluate(test_features, test_labels)
    print("Test accuracy (deu, eng, fra):", accuracy)

    # 6. extended result display for multi-class classification
    if show_extended_results:
        test_preds = classifier.predict(test_features)
        evaluate_and_display_results(test_labels, test_preds, label_names=("deu", "eng", "fra"))


#############################################################
#Exercise 2
#############################################################
def run_ex1_2(gram_size=2):
    """
    Run Excerise 1.2: Train skip-gram embedding with n-grams of size 3 on a small German corpus.

    Steps:
      1. Load a subset of the Leipzig news dataset (only German).
      2. Build a trigram vocabulary from a predefined alphabet.
      3. Initialize SkipGramEmbedding model and Adam optimzer.
      4. Generate skip-gram triplets (target, context, negatives) from sentences.
      5. Train the model for 10 epochs on batches of size 32.
      6. Evaluate by finding the top-10 most simliar words to quqries strinsg
    """
    # load data (german only for quick tests)
    dataset = load_dataset(
        "Texttechnologylab/leipzig-corpora-collection",
        "news_2024_1M",
        split="deu",
    )
    sentences = [s.split() for s in dataset["text"][:1000]]  # limit to 1000 sentences for speed
    tokens = [tok for s in sentences for tok in s]
    vocab_set = set(tokens)

    # build x-gram vocabulary using a small alphabet
    alphabet = list("abcdefghijklmnopqrstuvwxyzäöüß #")
    ngram_vocab = create_ngram_vocabulary(alphabet, gram_size)
    ngram_to_id = {ng: idx for idx, ng in enumerate(ngram_vocab)}

    # initialize model and optimizer
    embedding_dim = 50
    model = SkipGramEmbedding(ngram_vocab, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # prepare training triplets
    triplets = []
    for s in sentences:
        triplets += generate_skipgram_triplets(s, window_size=2, num_negative=3)

    # training loop
    batch_size = 32
    model.train()
    for epoch in range(10):
        random.shuffle(triplets)
        for i in range(0, len(triplets), batch_size):
            batch = triplets[i:i + batch_size]
            if len(batch) < batch_size:
                continue  # skip incomplete batch

            t_tensor, t_offsets, c_tensor, c_offsets, n_tensor, n_offsets, num_neg = triplets_to_tensors(
                batch, ngram_to_id, n=gram_size
            )

            # embed and compute loss
            target_emb = model.embed_target(t_tensor, t_offsets)
            context_emb = model.embed_context(c_tensor, c_offsets)
            neg_emb = model.context_embeddings(n_tensor, n_offsets).view(batch_size, num_neg, -1)

            loss = model(target_emb, context_emb, neg_emb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1} - Loss: {loss.item():.4f}")

    # evaluation: find top-10 similar words to "student"
    model.eval()
    queries = ["student", "Haus", "Auto", "Freund", "Mädchen", "Arbeit"]
    for query in queries:
        similar = get_similar_words(
            query_word=query,
            vocabulary=list(vocab_set),
            model=model,
            ngram_to_id=ngram_to_id,
            top_k=10,
            n=gram_size
        )
        print(f"\nWords simliar to '{query}':")
        for word, score in similar:
            print(f"  {word:12s} → Cosine: {score:.4f}")


def run_ex1_2_with_ngram_size(n: int = 4):
    """
    Run Excerise 1.2 with variable n-gram size on a small German corpus.

    Steps:
      1. Load a subset of the Leipzig news dataset (only German). 
      2. Build an n-gram vokabulry fom a predefined alphabet using parameter n.
      3. Iniitalize SkipGramEmbedding model and Adam optimizr.
      4. Generate skip-gram triplets (target, context, negatives) from sentences.
      5. Traing the model for 5 epochs on batches of size 32
      6. Evalutation: fid the top-10 most simliar German words to a set of query strings.
    """
    print(f"\n### Starting traing with n = {n} ###")

    # 1. data loading (German only for quick tests)  
    dataset = load_dataset(
        "Texttechnologylab/leipzig-corpora-collection",
        "news_2024_1M",
        split="deu",
    )
    sentences = [s.split() for s in dataset["text"][:500]]  # limit to 500 sentences for tests
    tokens = [tok for s in sentences for tok in s]
    vocab_set = set(tokens)

    # 2. n-gram vocabulary creation
    alphabet = list("abcdefghijklmnopqrstuvwxyzäöüß #")
    ngram_vocab = create_ngram_vocabulary(alphabet, n)
    ngram_to_id = {ng: idx for idx, ng in enumerate(ngram_vocab)}

    # 3. model initialization
    embedding_dim = 50
    model = SkipGramEmbedding(ngram_vocab, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 4. prepare skip-gram triplets d for traing
    triplets = []
    for s in sentences:
        triplets += generate_skipgram_triplets(s, window_size=2, num_negative=3)

    # 5. traing loop
    batch_size = 32
    model.train()
    for epoch in range(5):
        random.shuffle(triplets)
        for i in range(0, len(triplets), batch_size):
            batch = triplets[i:i + batch_size]
            if len(batch) < batch_size:
                continue  # skip if batch is too small

            # convert to tensors using n as gram size
            t_tensor, t_offsets, c_tensor, c_offsets, n_tensor, n_offsets, num_neg = triplets_to_tensors(
                batch, ngram_to_id, n
            )

            # embedings and compute loass
            target_emb = model.embed_target(t_tensor, t_offsets)
            context_emb = model.embed_context(c_tensor, c_offsets)
            neg_emb = model.context_embeddings(n_tensor, n_offsets).view(batch_size, num_neg, -1)

            loss = model(target_emb, context_emb, neg_emb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1} – Loss: {loss.item():.4f}")

    # 6. evalutation: find top-10 siliar German words for multiple queries
    model.eval()
    queries = ["student", "Haus", "Auto", "Freund", "Mädchen", "Arbeit"]
    for query in queries:
        similar = get_similar_words(
            query_word=query,
            vocabulary=list(vocab_set),
            model=model,
            ngram_to_id=ngram_to_id,
            top_k=10,
            n=n
        )
        print(f"\nWords simliar to '{query}' (n = {n}):")
        for word, score in similar:
            print(f"  {word:12s} → Cosine: {score:.4f}")



if __name__ == "__main__":
    # alphabet = create_alphabet()
    # vocabulary = create_vocabulary(alphabet)
    # featurizer = BiGramFeaturizer(vocabulary)
    # print(featurizer.n_grams("This is fine"))


    print("Running Exercise 1 - Task 1")
    run_ex1_1()

    print("Running Exercise 1 - Task 2: Optimized")
    run_ex1_1_vocab_optimized()

    print("Running Exercise 1 - Task 2: Minimized")
    run_ex1_1_minimal_vocab()

    print("Running Exercise 1 - Task 3: ")
    run_ex1_3()

    print("Running Exercise 1 - Extra:")
    run_ex1_extra()

    #############################################################
    #Exercise 2
    #############################################################
    print("Running Exercise 2 - Task 1 Bigrams")
    run_ex1_2()
    print("Running Exercise 2 - Task 1 Trigrams")
    run_ex1_2(3)

    print("Running Exercise 2 - Task 2")
    run_ex1_2_with_ngram_size(4)
