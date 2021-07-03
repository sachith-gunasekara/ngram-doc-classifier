#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tqdm import tqdm
from typing import List, Tuple
from collections import Counter
from .utils import ArgparseFormatter, dir_path, file_path, get_formatted_logger
import argparse
import typing
import json
import os
import re


def read_data(data_path: str, labels_path: str) -> Tuple[List[str], List[str]]:
    """ Read data and labels from files to memory """
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


def get_indices_by_language(
        labels: List[str]) -> Tuple[List[str], List[List[int]]]:
    """ Compute indices by language """
    # get unique list of sorted labels
    unique_labels = sorted(set(labels))

    # find indices matching each language
    indices_by_language = [[
        index for index, label in enumerate(labels) if label == unique_label
    ] for unique_label in unique_labels]

    # return both unique labels and indices
    return unique_labels, indices_by_language


def get_clean_doc(doc: str) -> str:
    """ Clean a given document """
    return re.sub(r'[!@#$()%^*?:;~.,"\'`0-9]', ' ', doc).lower()


def get_ngram_stats(doc: str, n: int) -> typing.Counter:
    """ Gather n-gram statistics per document """
    # split a document on spaces
    doc = doc.split()

    # initialize counter
    counter: typing.Counter = Counter()

    # iterate over characters in words
    for word in doc:
        counter += Counter([word[i:i + n] for i in range(len(word) - n + 1)])

    # return final counter
    return counter


def get_normalized_profile(
        raw_profile: List[Tuple[str, int]]) -> List[Tuple[str, float]]:
    """ Compute normalized language profile """
    # compute total count
    total = sum([element[1] for element in raw_profile])

    # normalize and return new profile
    return [(element[0], element[1] / total) for element in raw_profile]


def main(args: argparse.Namespace) -> None:
    """ Main workflow to compute language profiles """
    # read in data and labels to memory
    LOGGER.info("Reading data from disk")
    data, labels = read_data(args.train_data, args.train_labels)

    # get unique labels and indices
    LOGGER.info("Computing language indices")
    unique_labels, indices_by_language = get_indices_by_language(labels)

    # create model and fill with metadata
    model: dict = {}
    model["config"] = {}
    model["profiles"] = {}
    model["config"]["ngrams"] = args.ngrams
    model["config"]["ngram_cutoff"] = args.ngram_cutoff

    # iterate over all languages and update dictionary
    LOGGER.info("Computing all language profiles")
    for unique_label, indices in tqdm(
            list(zip(unique_labels, indices_by_language))):
        # create a local counter per-language
        local_counter: typing.Counter = Counter()

        # get all language-specific data
        data_subset = [get_clean_doc(data[index]) for index in indices]

        # compute n-gram statistics and update counter
        for doc in data_subset:
            local_counter += get_ngram_stats(doc, args.ngrams)

        # truncate counter
        local_counter = local_counter.most_common(args.ngram_cutoff)

        # normalize output from counter's most_common function
        local_counter = get_normalized_profile(local_counter)

        # add language profile to model
        model["profiles"][unique_label] = dict(local_counter)

    # create model and and path
    model_name = "model_%s_%s.json" % (args.ngrams, args.ngram_cutoff)
    model_path = os.path.join(args.models_directory, model_name)

    # dump final model
    LOGGER.info("Dumping final model: %s" % model_path)
    with open(model_path, "w", encoding='utf8') as output_file_stream:
        json.dump(model, output_file_stream, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=ArgparseFormatter)
    parser.add_argument("--train-data",
                        type=file_path,
                        default="./data/wili-2018/x_train.txt",
                        help="Path to training data")
    parser.add_argument("--train-labels",
                        type=file_path,
                        default="./data/wili-2018/y_train.txt",
                        help="Path to training labels")
    parser.add_argument("--models-directory",
                        type=dir_path,
                        default="./models",
                        help="Directory to dump models")
    parser.add_argument("--ngrams",
                        type=int,
                        default=2,
                        help="N-grams to use for language profiles")
    parser.add_argument("--ngram-cutoff",
                        type=int,
                        default=300,
                        help="Maximum n-grams per language profile")
    parser.add_argument(
        "--logging-level",
        help="Set logging level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        type=str)
    LOGGER = get_formatted_logger(parser.parse_known_args()[0].logging_level)
    main(parser.parse_args())
