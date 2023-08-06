#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tqdm import tqdm
from .utils import ArgparseFormatter, file_path, get_formatted_logger
from .train import get_clean_doc, get_ngram_stats
from .evaluate import get_diff_norms
import argparse
import json


def main(args: argparse.Namespace) -> None:
    """Main workflow to detect categories"""
    # read in data and labels to memory
    LOGGER.info("Reading data from disk")
    with open(args.predict_data, "r") as input_file_stream:
        data = [line.strip() for line in input_file_stream]

    # read model into memory
    LOGGER.info("Reading model: %s" % args.model)
    with open(args.model, "r") as input_file_stream:
        model = json.load(input_file_stream)

    # extract model-specific parameters
    ngrams_start = model["config"]["ngrams_start"]
    ngrams_end = model["config"]["ngrams_end"]
    predictions = []

    # iterate over all categories and update dictionary
    LOGGER.info("Detecting categories sequentially")
    for doc in tqdm(data):
        # clean document
        doc = get_clean_doc(doc)

        # compute n-gram statistics and update counter
        counter = get_ngram_stats(doc, ngrams_start, ngrams_end)

        # compute closest category
        diff_norms = get_diff_norms(counter, model)
        if diff_norms != []:
            predictions.append(sorted(diff_norms, key=lambda x: x[1])[0][0])
        else:
            predictions.append("Unknown")

    # print final results
    for doc, prediction in zip(data, predictions):
        print("%s" % prediction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=ArgparseFormatter)
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--predict-data", type=file_path, required=True, help="Path to prediction data"
    )
    parser.add_argument(
        "--model",
        type=file_path,
        default="./models/model_3_300.json",
        help="Path to model JSON file",
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
