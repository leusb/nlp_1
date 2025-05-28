from datasets import DatasetDict, concatenate_datasets, load_dataset
from collections import Counter

from nlpds.submission.ex1_1 import BiGramFeaturizer, BiGramLanguageClassifier
from nlpds.submission.utils import create_alphabet, create_vocabulary, evaluate_and_display_results, get_top_bigrams, create_french_german_alphabet



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



def run_ex1_2():
    dataset = load_dataset(
        "Texttechnologylab/leipzig-corpora-collection",
        "news_2024_1M",
        split=...,  # TODO
    )
    # TODO


if __name__ == "__main__":
    # alphabet = create_alphabet()
    # vocabulary = create_vocabulary(alphabet)
    # featurizer = BiGramFeaturizer(vocabulary)
    # print(featurizer.n_grams("This is fine"))


    # print("Running Exercise 1 - Task 1")
    # run_ex1_1()

    # print("Running Exercise 1 - Task 1.2 (Optimized Vocabulary)")
    run_ex1_3()

    # print("Running Exercise 1 - Task 2")
    # run_ex1_2()
