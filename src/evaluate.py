#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from math import sqrt
import numpy as np
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm
from typing import List, Tuple
from sklearn.metrics import classification_report
from .utils import ArgparseFormatter, file_path, dir_path, get_formatted_logger
from .train import (
    read_data_from_path,
    read_data_from_dataloader,
    get_clean_doc,
    get_ngram_stats,
)
import argparse
import typing
import json
import os
from sklearn.datasets import fetch_20newsgroups


def euclidean_distance(doc_vector: typing.List, cat_vector: typing.List):
    """
    Compute Euclidean distance between a document and category profile
    """
    doc_vector = np.array(doc_vector)
    cat_vector = np.array(cat_vector)

    distance = norm(doc_vector - cat_vector)

    return distance


def cosine_similarity(doc_vector: typing.List, cat_vector: typing.List):
    """
    Compute Cosine similarity between a document and category profile
    """
    doc_vector = np.array(doc_vector)
    cat_vector = np.array(cat_vector)

    similarity = dot(doc_vector, cat_vector) / (norm(doc_vector) * norm(cat_vector))

    return similarity


def get_diff_norms(counter: typing.Counter, model: dict) -> List[Tuple[str, float]]:
    """Compute all distance norms for a given document counter"""
    # define list for storage
    diff_norms = []

    # loop across all categories for comparison
    for cat in model["profiles"].keys():
        doc_vector = [
            float(counter[key]) if key in counter else 0.0
            for key in model["profiles"][cat].keys()
        ]
        doc_sum = sum(doc_vector)
        if doc_sum:
            doc_vector = [doc_score / doc_sum for doc_score in doc_vector]
            cat_vector = list(model["profiles"][cat].values())
            distance = euclidean_distance(doc_vector, cat_vector)
            diff_norms.append((cat, distance))
    # return final list
    return diff_norms


def main(args: argparse.Namespace) -> None:
    """Main workflow to evaluate categories detection models"""
    # read in data and labels to memory
    LOGGER.info("Reading data")
    data, labels = read_data_from_dataloader(
        fetch_20newsgroups, subset="test", remove=("headers", "footers", "quotes")
    )
    # data, labels = read_data_from_path(args.test_data, args.test_labels)

    # read model into memory
    LOGGER.info("Reading model: %s" % args.model)
    with open(args.model, "r") as input_file_stream:
        model = json.load(input_file_stream)

    # extract model-specific parameters
    ngrams_start = model["config"]["ngrams_start"]
    ngrams_end = model["config"]["ngrams_end"]
    ngram_method = model["config"]["ngram_method"]
    ngram_token = model["config"]["ngram_token"]
    predictions = []

    # iterate over all categories and update dictionary
    LOGGER.info("Detecting categories sequentially, this might take some time")
    for doc in tqdm(data):
        # compute n-gram statistics and update counter
        counter = get_ngram_stats(
            doc, ngrams_start, ngrams_end, ngram_method, ngram_token
        )

        # compute closest category
        diff_norms = get_diff_norms(counter, model)

        if diff_norms != []:
            predictions.append(sorted(diff_norms, key=lambda x: x[1])[0][0])
        else:
            predictions.append("Unknown")

    # produce classification report
    report = classification_report(labels, predictions, output_dict=True)
    report_path = os.path.join(
        args.models_directory,
        "reports",
        "euclidean",
        "classification_report_%s_to_%s_%s_%s_%s.json"
        % (
            ngrams_start,
            ngrams_end,
            model["config"]["ngram_cutoff"],
            ngram_method,
            ngram_token,
        ),
    )

    # dump classification report
    LOGGER.info("Dumping classification report: %s" % report_path)
    with open(report_path, "w") as output_file_stream:
        json.dump(report, output_file_stream)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=ArgparseFormatter)
    parser.add_argument(
        "--model",
        type=file_path,
        default="./models/euclidean/model_3_300_normal_char.json",
        help="Path to model JSON file",
    )
    parser.add_argument(
        "--test-data",
        type=file_path,
        default="./data/wili-2018/x_test.txt",
        help="Path to test data",
    )
    parser.add_argument(
        "--test-labels",
        type=file_path,
        default="./data/wili-2018/y_test.txt",
        help="Path to test labels",
    )
    parser.add_argument(
        "--models-directory",
        type=dir_path,
        default="./models",
        help="Directory to dump models and logs",
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
