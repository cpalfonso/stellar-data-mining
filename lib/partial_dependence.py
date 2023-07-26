"""Functions to generate partial dependence plots for ML models."""
import os
import time
from datetime import timedelta
from string import ascii_uppercase
from sys import stderr

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import PartialDependenceDisplay

from lib.format_feature_names import format_feature_name
from lib.pu import (
    CORRELATED_COLUMNS,
    COLUMNS_TO_DROP,
    PRESERVATION_COLUMNS,
    CUMULATIVE_COLUMNS,
)

DEFAULT_CLASSIFIER_FILENAME = "pu_classifier.joblib"
DEFAULT_DATA_FILENAME = "training_data.csv"
DEFAULT_OUTPUT_BASENAME = "partial_dependence"
DEFAULT_SAVE_FILENAME = "partial_dependence.joblib"

UNUSED_COLS = CORRELATED_COLUMNS | COLUMNS_TO_DROP | PRESERVATION_COLUMNS
FIGURE_SIZE = (11, 16)
NROWS = 3
NCOLS = 2
N_FEATURES = NROWS * NCOLS


def make_plot(
    output_basename=DEFAULT_OUTPUT_BASENAME,
    classifier_filename=DEFAULT_CLASSIFIER_FILENAME,
    data_filename=DEFAULT_DATA_FILENAME,
    save_filename=None,
    load_filename=None,
    n_jobs=1,
    verbose=False,
    cumulative_cols=True,
):
    """Create a partial dependence plot, loading from file if possible.

    Parameters
    ----------
    output_basename : str, default: 'partial_dependence'
        Base name of output image files.
    classifier_filename : str or scikit-learn estimator, default: 'pu_classifier.joblib'
        Scikit-learn classifier for partial dependence.
    data_filename : str or pandas.DataFrame, default: 'training_data.csv'
        Training dataset.
    save_filename : str, optional
        If provided, save partial dependence results to this joblib file.
    load_filename : str, optional
        If provided, load partial dependence results from this joblib file.
    verbose : bool, default: False
        Print log to stderr.
    n_jobs : int, default: 1
        Number of processes to use.
    cumulative_cols : bool, default: True
        Include cumulative subducted quantities in dataset.
    """
    if load_filename is not None:
        if not os.path.isfile(load_filename):
            raise FileNotFoundError(f"Input file not found: {load_filename}")
    if save_filename is not None:
        _check_dir(save_filename, name="save", verbose=verbose)
    _check_dir(output_basename, name="output", verbose=verbose)

    fig, axs = plt.subplots(NROWS, NCOLS, figsize=FIGURE_SIZE)

    if load_filename is not None:
        if verbose:
            print(
                "Loading PartialDependenceDisplay results from file:",
                str(load_filename),
                sep=" ",
                file=stderr,
            )
        disp = joblib.load(load_filename)
        if not isinstance(disp, PartialDependenceDisplay):
            raise TypeError(
                "Invalid type: {}".format(type(disp))
            )
        disp.plot(ax=np.ravel(axs))
    else:
        disp_kwargs = get_display_kw(
            classifier=classifier_filename,
            data=data_filename,
            n_jobs=n_jobs,
            cumulative_cols=cumulative_cols,
        )
        disp = PartialDependenceDisplay.from_estimator(
            ax=np.ravel(axs),
            verbose=int(verbose),
            **disp_kwargs,
        )
    adjust_plot(axs)
    _save_plot(fig, output_basename, verbose=verbose)

    if save_filename is not None:
        if verbose:
            print(
                " - Saving results to file: " + save_filename,
                file=stderr,
            )
        joblib.dump(disp, save_filename)


def _check_dir(filename, name="output", verbose=False):
    directory = os.path.abspath(os.path.dirname(filename))
    if not os.path.isdir(directory):
        if verbose:
            print(
                "{} directory does not exist;".format(name.capitalize()),
                "creating now:",
                directory,
                sep=" ",
                file=stderr,
            )
        os.makedirs(directory, exist_ok=True)


def get_display_kw(
    classifier,
    data,
    n_jobs=1,
    cumulative_cols=True,
):
    """Determine the appropriate keyword arguments for
    PartialDependenceDisplay.from_estimator.

    Parameters
    ----------
    classifier : str or scikit-learn estimator
        Scikit-learn classifier for partial dependence.
    data : str or pandas.DataFrame
        Training dataset.
    n_jobs : int, default: 1
        Number of processes to use.
    cumulative_cols : bool, default: True
        Include cumulative subducted quantities in dataset.

    Returns
    -------
    kwargs : dict
        Keyword argument dictionary with the following keys:
        - estimator
        - X
        - features
        - feature_names
        - n_jobs
    """
    if not isinstance(classifier, BaseEstimator):
        classifier = joblib.load(classifier)
    if isinstance(data, str):
        data = pd.read_csv(data)
    else:
        data = pd.DataFrame(data)

    if cumulative_cols:
        unused_cols = UNUSED_COLS.copy()
    else:
        unused_cols = UNUSED_COLS | CUMULATIVE_COLUMNS
    feature_names = sorted(
        set(data.columns).difference(unused_cols | {"label"})
    )
    feature_names = np.array(feature_names, dtype="object")

    if hasattr(classifier, "feature_importances_"):
        importance_array = np.array(classifier.feature_importances_)
    elif hasattr(classifier, "estimators_"):
        importance_array = np.array(
            [
                i.feature_importances_
                for i in classifier.estimators_
            ]
        ).mean(axis=0)
    else:
        raise TypeError(f"Invalid classifier type: {type(classifier)}")

    importances = pd.Series(importance_array, index=feature_names)
    importances_sorted = importances.sort_values(ascending=False)

    data_subset = (
        data
            .sort_index(axis="columns")
            .drop(columns=list(unused_cols | {"label"}), errors="ignore")
            .dropna()
    )
    x_train = np.array(data_subset)

    return {
        "estimator": classifier,
        "X": x_train,
        "features": list(importances_sorted.index[:N_FEATURES]),
        "feature_names": feature_names,
        "n_jobs": n_jobs,
    }


def adjust_plot(axs):
    """Tidy up plot (labels, gridlines, etc.)"""
    for ax, label in zip(np.ravel(axs), ascii_uppercase):
        if ax is not None:
            ax.set_xlabel(format_feature_name(ax.get_xlabel()))
            ax.tick_params(labelsize=8.25)
            ax.grid(color="grey", alpha=0.3, linestyle="dashed")
            ax.text(
                0.022, 0.97,
                label,
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax.transAxes,
                fontsize=18,
                bbox=dict(
                    facecolor="white",
                    edgecolor="lightgrey",
                    linewidth=0.5,
                    boxstyle="Round, pad=0.15, rounding_size=0.15",
                )
            )

    for i in np.arange(np.shape(axs)[0]):
        for j in np.arange(np.shape(axs)[1]):
            if j == 0:
                axs[i, j].set_ylabel(
                    "Partial dependence\n(probability)"
                )
            else:
                axs[i, j].set_ylabel("")


def _save_plot(fig, output_basename, verbose=False):
    for ext in (".png", ".pdf"):
        if verbose:
            print(
                " - Writing output file: " + output_basename + ext,
                file=stderr,
            )
        fig.savefig(output_basename + ext, dpi=350, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Create partial dependence plots (PNG and PDF format).",
    )
    parser.add_argument(
        "-i",
        "--input-file",
        help="classifier filename (default: {})".format(
            DEFAULT_CLASSIFIER_FILENAME
        ),
        default=DEFAULT_CLASSIFIER_FILENAME,
        dest="classifier_filename",
    )
    parser.add_argument(
        "-d",
        "--data-file",
        help="training data filename (default: {})".format(
            DEFAULT_DATA_FILENAME
        ),
        default=DEFAULT_DATA_FILENAME,
        dest="data_filename",
    )
    parser.add_argument(
        "-o",
        "--output-name",
        help="output image basename (default: {})".format(
            DEFAULT_OUTPUT_BASENAME
        ),
        default=DEFAULT_OUTPUT_BASENAME,
        dest="output_basename",
    )
    parser.add_argument(
        "-s",
        "--save-results",
        help=(
            "write PartialDependenceDisplay object to file "
            + "(default: {})".format(DEFAULT_SAVE_FILENAME)
        ),
        nargs="?",
        default=None,
        const=DEFAULT_SAVE_FILENAME,
        dest="save_filename",
    )
    parser.add_argument(
        "-l",
        "--load-results",
        help=(
            "load PartialDependenceDisplay object from file "
            + "(default: {})".format(DEFAULT_SAVE_FILENAME)
        ),
        nargs="?",
        default=None,
        const=DEFAULT_SAVE_FILENAME,
        dest="load_filename",
    )
    parser.add_argument(
        "-n",
        "--n_jobs",
        help="number of jobs to use (default: 1)",
        type=int,
        default=1,
        dest="n_jobs",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="print logs to stderr",
        action="store_true",
        dest="verbose",
    )
    parser.add_argument(
        "-c",
        "--no-cumulative",
        help="ignore cumulative subducted quantities",
        action="store_false",
        dest="cumulative_cols",
    )
    args = parser.parse_args()

    start_time = time.time()

    make_plot(
        output_basename=args.output_basename,
        classifier_filename=args.classifier_filename,
        data_filename=args.data_filename,
        save_filename=args.save_filename,
        load_filename=args.load_filename,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
        cumulative_cols=args.cumulative_cols,
    )

    if args.verbose:
        end_time = time.time()
        duration = timedelta(seconds=end_time - start_time)
        print(
            f"Duration: {duration}",
            file=stderr,
        )
