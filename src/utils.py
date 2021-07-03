#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os

FORMAT = (
    '%(asctime)s | %(levelname)s | %(filename)s | %(funcName)s | %(message)s')


def dir_path(path: str) -> str:
    """
    Argparse type helper to ensure directory exists
    """
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError("%s is not a valid directory" % path)


def file_path(path: str) -> str:
    """
    Argparse type helper to ensure file exists
    """
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError("%s is not a valid file" % path)


def get_formatted_logger(level: str) -> logging.Logger:
    """
    Create a sane logger
    """
    # get root logger
    logger = logging.getLogger()

    # define logger levels
    levels = {
        'critical': logging.CRITICAL,
        'error': logging.ERROR,
        'warning': logging.WARNING,
        'info': logging.INFO,
        'debug': logging.DEBUG
    }

    # set logger level
    logger.setLevel(levels[level.lower()])

    # create formatter
    formatter = logging.Formatter(FORMAT)

    # set output stream to stdout
    stderr_handler = logging.StreamHandler()
    stderr_handler.setLevel(logging.DEBUG)
    stderr_handler.setFormatter(formatter)

    # add stream to logger
    logger.addHandler(stderr_handler)

    # return final logger
    return logger


class Metavar_Circum_Symbols(argparse.HelpFormatter):
    """
    Help message formatter which uses the argument 'type' as the default
    metavar value (instead of the argument 'dest')

    Only the name of this class is considered a public API. All the methods
    provided by the class are considered an implementation detail.
    """
    def _get_default_metavar_for_optional(self,
                                          action: argparse.Action) -> str:
        """
        Function to return option metavariable type with circum-symbols
        """
        return "<" + action.type.__name__ + ">"  # type: ignore

    def _get_default_metavar_for_positional(self,
                                            action: argparse.Action) -> str:
        """
        Function to return positional metavariable type with circum-symbols
        """
        return "<" + action.type.__name__ + ">"  # type: ignore


class ArgparseFormatter(argparse.ArgumentDefaultsHelpFormatter,
                        Metavar_Circum_Symbols):
    """
    Class to combine argument parsers in order to display meta-variables
    and defaults for arguments
    """
    pass
