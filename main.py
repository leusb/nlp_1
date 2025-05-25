from datasets import DatasetDict, concatenate_datasets, load_dataset

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

    # TODO:
    # - instantiate the featurizer
    # - featurize the dataset
    # - train the classifier
    # - evaluate the classifier


def run_ex1_2():
    dataset = load_dataset(
        "Texttechnologylab/leipzig-corpora-collection",
        "news_2024_1M",
        split=...,  # TODO
    )
    # TODO


if __name__ == "__main__":
    print("Running Exercise 1 - Task 1")
    run_ex1_1()

    print("Running Exercise 1 - Task 2")
    run_ex1_2()
