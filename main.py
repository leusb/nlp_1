from datasets import DatasetDict, concatenate_datasets, load_dataset
from nlpds.submission.ex1_1 import BiGramFeaturizer, BiGramLanguageClassifier
from nlpds.submission.utils import create_alphabet, create_vocabulary



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


    print("Running Exercise 1 - Task 1")
    run_ex1_1()

    # print("Running Exercise 1 - Task 2")
    # run_ex1_2()
