#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tqdm import tqdm
from typing import List, Tuple, Callable, Any
from collections import Counter
from .utils import ArgparseFormatter, dir_path, file_path, get_formatted_logger
from sklearn.datasets import fetch_20newsgroups
import argparse
import typing
import json
import os
import re

import nltk

nltk.download("punkt")
from nltk.tokenize import sent_tokenize, word_tokenize


def read_data_from_dataloader(
    loader: Callable[..., Any], **kwargs
) -> Tuple[List[str], List[str]]:
    data_bunch = loader(**kwargs)

    data = data_bunch.get("data")
    labels = [data_bunch.target_names[target] for target in data_bunch.get("target")]

    assert len(data) == len(labels)

    return data, labels


def read_data_from_path(
    data_path: str, labels_path: str
) -> Tuple[List[str], List[str]]:
    """Read data and labels from files to memory"""
    # read data
    with open(data_path, "r") as input_file_stream:
        data = [line.strip() for line in input_file_stream]

    # read labels
    with open(labels_path, "r") as input_file_stream:
        labels = [line.strip() for line in input_file_stream]

    # ensure data sanity
    assert len(data) == len(labels)

    # return all
    return data, labels


def get_indices_by_category(labels: List[str]) -> Tuple[List[str], List[List[int]]]:
    """Compute indices by category"""
    # get unique list of sorted labels
    unique_labels = sorted(set(labels))

    # find indices matching each category
    indices_by_category = [
        [index for index, label in enumerate(labels) if label == unique_label]
        for unique_label in unique_labels
    ]

    # return both unique labels and indices
    return unique_labels, indices_by_category


def get_clean_doc(doc: str) -> str:
    """Clean a given document"""
    return re.sub(r"\n", " ", re.sub(r"\[[^\[\]]*\]|[^\w\s]|_|\d", "", doc)).lower()


def get_ngram_stats(
    doc: str, ngrams: int, ngram_method: str, ngram_token: str
) -> typing.Counter:
    """Gather n-gram statistics per document"""

    # initialize counter
    counter: typing.Counter = Counter()

    if ngram_method == "normal":
        # clean doc
        doc = get_clean_doc(doc)
        # tokenize document by words
        words = word_tokenize(doc)

        if ngram_token == "word":
            # iterate over words
            for index in range(len(words) - 1):
                counter += Counter([" ".join(words[index : index + ngrams])])

        elif ngram_token == "char":
            # iterate over characters in words
            for word in words:
                counter += Counter(
                    [word[i : i + ngrams] for i in range(len(word) - ngrams + 1)]
                )

        elif ngram_token == "char_wb":
            # pad each word with a space
            doc = [f" {word} " for word in words]

            # iterate over characters in words
            for word in doc:
                counter += Counter(
                    [word[i : i + ngrams] for i in range(len(word) - ngrams + 1)]
                )

    elif ngram_method == "sentence":
        if ngram_token == "word":
            # tokenize document into sentences
            doc = sent_tokenize(doc)

            # iterate over the characters within sentence structures
            for sentence in doc:
                sentence = get_clean_doc(sentence)
                words = word_tokenize(sentence)

                for index in range(len(words) - ngrams + 1):
                    counter += Counter([" ".join(words[index : index + ngrams])])

        elif ngram_token == "char":
            # clean doc
            doc = get_clean_doc(doc)
            # tokenize document by words
            doc = word_tokenize(doc)

            # iterate over characters in words
            for word in doc:
                counter += Counter(
                    [word[i : i + ngrams] for i in range(len(word) - ngrams + 1)]
                )

        elif ngram_token == "char_wb":
            # tokenize into sentences
            doc = sent_tokenize(doc)

            # iterate over the characters within sentence structures
            for sentence in doc:
                sentence = get_clean_doc(sentence)

                words = word_tokenize(sentence)

                for index, word in enumerate(words):
                    if index == 0:
                        word = f"{word} "
                    elif index == len(words) - 1:
                        word = f" {word}"
                    else:
                        word = f" {word} "

                    counter += Counter(
                        [word[i : i + ngrams] for i in range(len(word) - ngrams + 1)]
                    )

    # return final counter
    return counter


def get_normalized_profile(
    raw_profile: List[Tuple[str, int]]
) -> List[Tuple[str, float]]:
    """Compute normalized ngram profile"""
    # compute total count
    total = sum([element[1] for element in raw_profile])

    # normalize and return new profile
    return [(element[0], element[1] / total) for element in raw_profile]


def main(args: argparse.Namespace) -> None:
    """Main workflow to compute category profiles"""
    # read in data and labels to memory
    LOGGER.info("Reading data from disk")
    data, labels = read_data_from_dataloader(
        fetch_20newsgroups, subset="train", remove=("headers", "footers", "quotes")
    )

    # get unique labels and indices
    LOGGER.info("Computing category indices")
    unique_labels, indices_by_category = get_indices_by_category(labels)

    # create model and fill with metadata
    model: dict = {}
    model["config"] = {}
    model["profiles"] = {}
    model["config"]["ngrams"] = args.ngrams
    model["config"]["ngram_cutoff"] = args.ngram_cutoff
    model["config"]["ngram_method"] = args.ngram_method
    model["config"]["ngram_token"] = args.ngram_token

    # iterate over all categories and update dictionary
    LOGGER.info("Computing all category profiles")
    for unique_label, indices in tqdm(list(zip(unique_labels, indices_by_category))):
        # create a local counter per-category
        local_counter: typing.Counter = Counter()

        # get all category-specific data
        data_subset = [data[index] for index in indices]

        # compute n-gram statistics and update counter
        for doc in data_subset:
            local_counter += get_ngram_stats(
                doc, args.ngrams, args.ngram_method, args.ngram_token
            )

        # truncate counter
        local_counter = local_counter.most_common(args.ngram_cutoff)

        # normalize output from counter's most_common function
        local_counter = get_normalized_profile(local_counter)

        # add category profile to model
        model["profiles"][unique_label] = dict(local_counter)

    # create model and and path
    model_name = "model_%s_%s_%s_%s.json" % (
        args.ngrams,
        args.ngram_cutoff,
        args.ngram_method,
        args.ngram_token,
    )
    model_path = os.path.join(args.models_directory, model_name)

    # dump final model
    LOGGER.info("Dumping final model: %s" % model_path)
    with open(model_path, "w", encoding="utf8") as output_file_stream:
        json.dump(model, output_file_stream, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=ArgparseFormatter)
    parser.add_argument(
        "--train-data",
        type=file_path,
        default="./data/wili-2018/x_train.txt",
        help="Path to training data",
    )
    parser.add_argument(
        "--train-labels",
        type=file_path,
        default="./data/wili-2018/y_train.txt",
        help="Path to training labels",
    )
    parser.add_argument(
        "--models-directory",
        type=dir_path,
        default="./models",
        help="Directory to dump models and logs",
    )
    parser.add_argument(
        "--ngrams",
        type=int,
        default=3,
        help="N-grams to use for category profiles",
    )
    parser.add_argument(
        "--ngram-cutoff",
        type=int,
        default=300,
        help="Maximum character n-grams per category profile",
    )
    parser.add_argument(
        "--ngram-method",
        type=str,
        default="normal",
        choices=["normal", "sentence"],
        help="Define how the n-grams are built up",
    )
    parser.add_argument(
        "--ngram-token",
        type=str,
        default="char",
        choices=["word", "char", "char_wb"],
        help="Define the token considered to build n-gram profile",
    )
    parser.add_argument(
        "--logging-level",
        help="Set logging level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        type=str,
    )
    LOGGER = get_formatted_logger(parser.parse_known_args()[0].logging_level)
    main(parser.parse_args())
