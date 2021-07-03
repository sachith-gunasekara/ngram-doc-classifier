#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from math import sqrt
from tqdm import tqdm
from typing import List, Tuple
from .utils import ArgparseFormatter, file_path, dir_path, get_formatted_logger
from .train import read_data, get_clean_doc, get_ngram_stats
from sklearn.metrics import classification_report
import argparse
import typing
import json
import os


def get_diff_norms(counter: typing.Counter,
                   model: dict) -> List[Tuple[str, float]]:
    """ Compute all distance norms for a given document counter """
    # define list for storage
    diff_norms = []

    # loop across all languages for comparison
    for lang in model["profiles"].keys():
        doc_vector = [
            float(counter[key]) if key in counter else 0.
            for key in model["profiles"][lang].keys()
        ]
        doc_sum = sum(doc_vector)
        if doc_sum:
            doc_vector = [doc_score / doc_sum for doc_score in doc_vector]
            lang_vector = list(model["profiles"][lang].values())
            distance = sqrt(
                sum([(doc_score - lang_score)**2
                     for doc_score, lang_score in zip(doc_vector, lang_vector)
                     ]))
            diff_norms.append((lang, distance))

    # return final list
    return diff_norms


def main(args: argparse.Namespace) -> None:
    """ Main workflow to compute language profiles """
    # read in data and labels to memory
    LOGGER.info("Reading data from disk")
    data, labels = read_data(args.test_data, args.test_labels)

    # read model into memory
    LOGGER.info("Reading model: %s" % args.model)
    with open(args.model, "r") as input_file_stream:
        model = json.load(input_file_stream)

    # extract model-specific parameters
    ngrams = model["config"]["ngrams"]
    predictions = []

    # iterate over all languages and update dictionary
    LOGGER.info("Detecting languages sequentially")
    for doc in tqdm(data):
        # clean document
        doc = get_clean_doc(doc)

        # compute n-gram statistics and update counter
        counter = get_ngram_stats(doc, ngrams)

        # compute closest language
        diff_norms = get_diff_norms(counter, model)
        if diff_norms != []:
            predictions.append(sorted(diff_norms, key=lambda x: x[1])[0][0])
        else:
            predictions.append("UNK")

    # produce classification report
    report = classification_report(labels, predictions, output_dict=True)

    # dump classification report
    report_path = os.path.join(
        args.models_directory, "classification_report_%s_%s.json" %
        (ngrams, model["config"]["ngram_cutoff"]))
    LOGGER.info("Dumping classification report: %s" % report_path)
    with open(report_path, "w") as output_file_stream:
        json.dump(report, output_file_stream)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=ArgparseFormatter)
    parser.add_argument("--model",
                        type=file_path,
                        default="./models/model_2_300.json",
                        help="Path to model JSON file")
    parser.add_argument("--test-data",
                        type=file_path,
                        default="./data/wili-2018/x_test.txt",
                        help="Path to test data")
    parser.add_argument("--test-labels",
                        type=file_path,
                        default="./data/wili-2018/y_test.txt",
                        help="Path to test labels")
    parser.add_argument("--models-directory",
                        type=dir_path,
                        default="./models",
                        help="Directory to dump models")
    parser.add_argument(
        "--logging-level",
        help="Set logging level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        type=str)
    LOGGER = get_formatted_logger(parser.parse_known_args()[0].logging_level)
    main(parser.parse_args())
