import os
import sys
import warnings
from sys import stderr

warnings.filterwarnings(
    "ignore",
    # message="estimator",
    category=FutureWarning,
)

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import BaseEstimator

DIRNAME = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, DIRNAME)

from lib.pu import (
    BASE_MODELS,
    create_classifier,
)

DEFAULT_RANDOM_STATE = None
DEFAULT_NPROCS = 1

DEFAULT_TRAINING_FILENAME = os.path.join(
    DIRNAME,
    "training_data.csv",
)
_OUTDIR = os.path.join(DIRNAME, "outputs")
DEFAULT_OUTPUT_FILENAME = os.path.join(
    _OUTDIR,
    "pu_classifier.joblib",
)
DEFAULT_BASE_CLASSIFIER = "randomforest"
DEFAULT_NUM_NEGATIVES = 0
DEFAULT_WEIGHTS_COLUMN = None
CONST_WEIGHTS_COLUMN = "Cu (Mt)"


def main(
    training_filename=DEFAULT_TRAINING_FILENAME,
    output_filename=DEFAULT_OUTPUT_FILENAME,
    random_state=DEFAULT_RANDOM_STATE,
    base_classifier=DEFAULT_BASE_CLASSIFIER,
    weights_column=DEFAULT_WEIGHTS_COLUMN,
    nprocs=DEFAULT_NPROCS,
    verbose=False,
    preservation=False,
    remove_cumulative=False,
    num_negatives=DEFAULT_NUM_NEGATIVES,
):
    args = check_args(
        training_filename=training_filename,
        output_filename=output_filename,
        base_classifier=base_classifier,
        weights_column=weights_column,
        random_state=random_state,
        preservation=preservation,
        remove_cumulative=remove_cumulative,
        nprocs=nprocs,
        verbose=verbose,
        num_negatives=num_negatives,
    )
    training_filename = args["training_filename"]
    output_filename = args["output_filename"]
    base_classifier = args["base_classifier"]
    weights_column = args["weights_column"]
    random_state = args["random_state"]
    preservation = args["preservation"]
    remove_cumulative = args["remove_cumulative"]
    nprocs = args["nprocs"]
    verbose = args["verbose"]
    num_negatives = args["num_negatives"]

    random_state = np.random.default_rng(random_state)

    if verbose:
        print(
            "Creating classifier from training data: " + training_filename,
            file=stderr,
        )
        print(
            "Base estimator: {}".format(
                type(base_classifier).__name__
                if isinstance(
                    base_classifier,
                    BaseEstimator,
                )
                else base_classifier
            )
        )
        if weights_column is not None:
            print(
                "Weighting samples by column: " + weights_column,
                file=stderr,
            )
        if preservation:
            print(
                "Including preservation-related quantities",
                file=stderr,
            )
        if num_negatives > 0:
            print(
                "Including {} known negative observations".format(num_negatives),
                file=stderr,
            )

    output_dir = os.path.dirname(output_filename)
    if not os.path.isdir(output_dir):
        if verbose:
            print(
                f"Output directory does not exist; creating now: {output_dir}",
                file=stderr,
            )
        os.makedirs(output_dir, exist_ok=True)

    training_data = pd.read_csv(training_filename)
    # if "set" in training_data.columns:
    #     training_data = training_data[training_data["set"] == "train"]
    classifier, x, y = create_classifier(
        training_data=training_data,
        random_state=random_state,
        base_classifier=base_classifier,
        threads=nprocs,
        return_arrays=True,
        weights_column=weights_column,
        preservation=preservation,
        remove_cumulative=remove_cumulative,
        num_negatives=num_negatives,
    )
    if output_filename is not None:
        if verbose:
            print(
                "Saving classifier to file: " + output_filename,
                file=stderr,
            )
        dump(classifier, output_filename)
    return classifier, x, y


def get_args():
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Create P-U classifier from given training data",
    )
    parser.add_argument(
        "-i",
        "--input-file",
        help="training data input filename; default: `{}`".format(
            os.path.relpath(DEFAULT_TRAINING_FILENAME)
        ),
        default=DEFAULT_TRAINING_FILENAME,
        dest="training_filename",
        metavar="INPUT_FILE",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        help="model output filename; default: `{}`".format(
            os.path.relpath(DEFAULT_OUTPUT_FILENAME)
        ),
        default=DEFAULT_OUTPUT_FILENAME,
        dest="output_filename",
        metavar="OUTPUT_FILE",
    )
    parser.add_argument(
        "-w",
        "--weights",
        help="weight samples by a column; default: '{}'".format(
            CONST_WEIGHTS_COLUMN
        ),
        nargs="?",
        default=DEFAULT_WEIGHTS_COLUMN,
        const=CONST_WEIGHTS_COLUMN,
        dest="weights_column",
        metavar="WEIGHTS_COLUMN",
    )
    parser.add_argument(
        "-n",
        "--nprocs",
        help="number of processes to use; default: {}".format(DEFAULT_NPROCS),
        type=int,
        default=DEFAULT_NPROCS,
        dest="nprocs",
    )
    parser.add_argument(
        "--base-estimator",
        help="base estimator to use; default: 'randomforest'",
        choices=set(BASE_MODELS.keys()),
        default="randomforest",
        metavar="MODEL",
        dest="base_classifier",
    )
    parser.add_argument(
        "-p",
        "--preservation",
        help="include preservation-related quantities in classifier",
        action="store_true",
        dest="preservation",
    )
    parser.add_argument(
        "--no-cumulative",
        help="do not include cumulative subducted quantities in classifier",
        action="store_true",
        dest="remove_cumulative",
    )
    parser.add_argument(
        "-r",
        "--random-state",
        help="seed for RNG; default: {}".format(DEFAULT_RANDOM_STATE),
        type=int,
        dest="random_state",
        default=DEFAULT_RANDOM_STATE,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="print more information to stderr",
        action="store_true",
        dest="verbose",
    )
    # parser.add_argument(
    #     "-g",
    #     "--num-negatives",
    #     help="number of negative observations to include as unlabelled; default: {}".format(
    #         DEFAULT_NUM_NEGATIVES
    #     ),
    #     type=int,
    #     default=DEFAULT_NUM_NEGATIVES,
    #     dest="num_negatives",
    # )
    return vars(parser.parse_args())


def check_args(**kwargs):
    training_filename = os.path.abspath(
        kwargs.pop("training_filename", DEFAULT_TRAINING_FILENAME)
    )
    output_filename = os.path.abspath(
        kwargs.pop("output_filename", DEFAULT_OUTPUT_FILENAME)
    )
    base_classifier = kwargs.pop("base_classifier", DEFAULT_BASE_CLASSIFIER)
    weights_column = kwargs.pop("weights_column", DEFAULT_WEIGHTS_COLUMN)
    nprocs = int(kwargs.pop("nprocs", DEFAULT_NPROCS))
    preservation = bool(kwargs.pop("preservation", False))
    remove_cumulative = bool(kwargs.pop("remove_cumulative", False))
    random_state = int(kwargs.pop("random_state", DEFAULT_RANDOM_STATE))
    verbose = bool(kwargs.pop("verbose", False))
    num_negatives = int(kwargs.pop("num_negatives", DEFAULT_NUM_NEGATIVES))

    for key in kwargs.keys():
        raise TypeError(
            "check_args got an unexpected keyword argument: " + key
        )

    if not os.path.isfile(training_filename):
        raise FileNotFoundError("Input file not found: " + training_filename)
    if num_negatives < 0:
        raise ValueError("`num_negatives` must be greater than or equal to zero.")

    return {
        "training_filename": training_filename,
        "output_filename": output_filename,
        "base_classifier": base_classifier,
        "weights_column": weights_column,
        "nprocs": nprocs,
        "preservation": preservation,
        "remove_cumulative": remove_cumulative,
        "random_state": random_state,
        "verbose": verbose,
        "num_negatives": num_negatives,
    }


if __name__ == "__main__":
    main(**get_args())
