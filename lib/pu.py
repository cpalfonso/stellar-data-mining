import os
from sys import stderr

import numpy as np
import pandas as pd
import xarray as xr
from joblib import (
    Parallel,
    delayed,
    dump,
)
from pulearn import BaggingPuClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from xgboost import XGBClassifier

COLUMNS_TO_DROP = {
    "present_lon",
    "present_lat",
    "region",
    "Cu (Mt)",
    "age (Ma)",
    "plate_id",
    "lon",
    "lat",
    "set",
    "source",
    "overriding_plate_id",
    "subducting_plate_ID",
    "trench_plate_ID",
    "arc_segment_length (degrees)",
    "trench_normal_angle (degrees)",
    "distance_from_trench_start (degrees)",
    "crustal_thickness_n",
    "magnetic_anomaly_n",
}
CORRELATED_COLUMNS = {
    "arc_trench_distance (km)",
    "convergence_rate (cm/yr)",
    "subducting_plate_absolute_velocity (cm/yr)",
    "trench_velocity (cm/yr)",
    "crustal_thickness_max (m)",
    "crustal_thickness_range (m)",
    "crustal_thickness_median (m)",
    "crustal_thickness_min (m)",
    "magnetic_anomaly_max (nT)",
    "magnetic_anomaly_range (nT)",
    "magnetic_anomaly_median (nT)",
    "water_thickness (m)",
    "subducted_water_volume (m)",
    "slab_dip (degrees)",
}
PRESERVATION_COLUMNS = {
    "total_precipitation (km)",
    "total_convergence (km)",
}
CUMULATIVE_COLUMNS = {
    "subducted_plate_volume (m)",
    "subducted_sediment_volume (m)",
    "subducted_water_volume (m)",
    "subducted_carbonates_volume (m)",
}

BASE_MODELS = {
    "randomforest": RandomForestClassifier(
        n_jobs=1,
        n_estimators=100,
    ),
    "gradientboosting": HistGradientBoostingClassifier(
        max_iter=200,
    ),
    "adaboost": AdaBoostClassifier(
        n_estimators=200,
    ),
    "xgboost": XGBClassifier(
        n_estimators=200,
    ),
}

DEFAULT_RANDOM_STATE = None
DEFAULT_NPROCS = 1

DIRNAME = os.path.abspath(os.path.dirname(__file__))
DEFAULT_TRAINING_FILENAME = os.path.join(
    DIRNAME,
    "..",
    "training_data.csv",
)
_OUTDIR = os.path.join(DIRNAME, "..", "outputs")
DEFAULT_OUTPUT_FILENAME = os.path.join(
    _OUTDIR,
    "pu_classifier.joblib",
)
DEFAULT_BASE_CLASSIFIER = "randomforest"
DEFAULT_WEIGHTS_COLUMN = None
CONST_WEIGHTS_COLUMN = "Cu (Mt)"


def create_classifier(
    training_data,
    base_classifier="randomforest",
    random_state=None,
    threads=1,
    return_arrays=False,
    label="label",
    fit=True,
    pu_kwargs=None,
    preservation=False,
    remove_correlated=True,
    remove_cumulative=False,
):
    random_state = np.random.default_rng(random_state)

    if not isinstance(base_classifier, BaseEstimator):
        if base_classifier not in BASE_MODELS.keys():
            raise ValueError("Invalid base_classifier: {}".format(base_classifier))
        base_classifier = BASE_MODELS[base_classifier]
    if pu_kwargs is None:
        pu_kwargs = {}

    x, y = get_xy(
        training_data,
        label=label,
        remove_correlated=remove_correlated,
        remove_cumulative=remove_cumulative,
        remove_preservation=(not preservation),
    )

    n_estimators = pu_kwargs.pop("n_estimators", 250)
    max_samples = pu_kwargs.pop("max_samples", 1.0)

    bc = BaggingPuClassifier(
        base_classifier,
        n_jobs=threads,
        random_state=np.random.RandomState(random_state.bit_generator),
        n_estimators=n_estimators,
        max_samples=max_samples,
        **pu_kwargs,
    )
    if fit:
        bc.fit(x, y)
    if return_arrays:
        return bc, x, y
    return bc


def get_xy(
    data,
    label="label",
    remove_correlated=True,
    remove_cumulative=False,
    remove_preservation=True,
):
    if not isinstance(data, pd.DataFrame):
        try:
            data = pd.read_csv(data)
        except Exception:
            data = pd.DataFrame(data)

    to_drop = COLUMNS_TO_DROP.copy()
    if remove_correlated:
        to_drop = to_drop.union(CORRELATED_COLUMNS)
    if remove_cumulative:
        to_drop = to_drop.union(CUMULATIVE_COLUMNS)
    if remove_preservation:
        to_drop = to_drop.union(PRESERVATION_COLUMNS)

    data = data.sort_index(axis="columns")
    data = data.drop(columns=list(to_drop), errors="ignore")
    data = data.dropna()
    x = np.array(data.drop(columns=label))
    y = np.array(
        (data[label] == "positive").astype("int")
    )
    return x, y


def downsample_unlabelled(data, n=None, random_state=None, label="label"):
    if not isinstance(random_state, np.random.Generator):
        random_state = np.random.default_rng(random_state)
    if n is None:
        n = (data[label] == "positive").sum()
    labelled = data[data[label].isin({"positive", "negative"})]
    unlabelled = data[data[label].isin({"unlabeled", "unlabelled"})]
    idx = random_state.choice(
        unlabelled.index,
        size=n,
        replace=False,
    )
    unlabelled = unlabelled.loc[idx, :]
    return pd.concat((labelled, unlabelled), ignore_index=True)


def calculate_feature_importances(classifier):
    try:
        return classifier.feature_importances_
    except AttributeError:
        pass
    importances = [i.feature_importances_ for i in classifier.estimators_]
    importances = np.vstack(importances)
    importances = np.mean(importances, axis=0)
    return importances


def calculate_probabilities(point_data, classifier):
    point_data = point_data.copy()

    features = []
    for column in sorted(point_data.columns.values):
        if column in (
            "lon",
            "lat",
            "present_lon",
            "present_lat",
            "age (Ma)",
        ):
            continue
        arr = np.array(point_data[column]).reshape((-1, 1))
        features.append(arr)
    x = np.hstack(features)

    try:
        probs = classifier.predict_proba(x)[:, 1].flatten()
    except ValueError as err:
        for i in (
            set(point_data.columns.values).difference(
                {
                    "lon",
                    "lat",
                    "present_lon",
                    "present_lat",
                    "age (Ma)",
                }
            )
        ):
            print(i)
        raise err

    lons = np.array(point_data["lon"])
    lats = np.array(point_data["lat"])
    ages = np.array(point_data["age (Ma)"])
    out = pd.DataFrame(
        {
            "lon": lons,
            "lat": lats,
            "age (Ma)": ages,
            "probability": probs,
        }
    )
    return out


def create_probability_grids(
    probabilities,
    output_dir,
    resolution=None,
    extent=None,
    threads=1,
    verbose=False,
):
    data = probabilities.copy()
    times = data["age (Ma)"].unique()

    if threads == 1:
        for time in times:
            _create_grid_time(
                time=time,
                data_lons=np.array(data[data["age (Ma)"] == time]["lon"]),
                data_lats=np.array(data[data["age (Ma)"] == time]["lat"]),
                data_values=np.array(data[data["age (Ma)"] == time]["probability"]),
                resolution=resolution,
                output_dir=output_dir,
                extent=extent,
                verbose=verbose,
            )
    else:
        with Parallel(threads, verbose=10 * int(verbose)) as p:
            p(
                delayed(_create_grid_time)(
                    time=time,
                    data_lons=np.array(data[data["age (Ma)"] == time]["lon"]),
                    data_lats=np.array(data[data["age (Ma)"] == time]["lat"]),
                    data_values=np.array(data[data["age (Ma)"] == time]["probability"]),
                    resolution=resolution,
                    output_dir=output_dir,
                    extent=extent,
                    verbose=verbose,
                )
                for time in times
            )


def _create_grid_time(
    time,
    data_lons,
    data_lats,
    data_values,
    resolution,
    output_dir,
    extent=None,
    verbose=False,
):
    time = int(np.around(time))
    output_filename = os.path.join(
        output_dir, "probability_grid_{}Ma.nc".format(time)
    )

    if extent is None:
        xmin = np.nanmin(data_lons)
        xmax = np.nanmax(data_lons)
        ymin = np.nanmin(data_lats)
        ymax = np.nanmax(data_lats)
    else:
        xmin, xmax, ymin, ymax = extent

    if resolution is None:
        resx = np.nanmin(np.gradient(np.sort(np.unique(data_lons))))
        resy = np.nanmin(np.gradient(np.sort(np.unique(data_lats))))
    else:
        resx = resolution
        resy = resolution

    grid_lons = np.arange(xmin, xmax + resx, resx)
    grid_lats = np.arange(ymin, ymax + resy, resy)
    grid_mlons, grid_mlats = np.meshgrid(grid_lons, grid_lats)

    arr = np.full((grid_lats.size, grid_lons.size), np.nan, dtype=float)
    for data_lon, data_lat, data_value in zip(
        data_lons, data_lats, data_values
    ):
        mask = np.logical_and(grid_mlons == data_lon, grid_mlats == data_lat)
        arr[mask] = data_value

    dset = xr.Dataset(
        data_vars={
            "z": (("lat", "lon"), arr),
        },
        coords={
            "lon": grid_lons,
            "lat": grid_lats,
            # "time": time,
        },
    )
    if verbose:
        print(
            "\t- Writing output file: " + os.path.basename(output_filename),
            file=stderr,
        )
    dset.to_netcdf(
        output_filename,
        encoding={
            "z": {
                "zlib": True,
                "dtype": "float32",
            }
        },
    )
    return dset


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
):
    args = _check_args(
        training_filename=training_filename,
        output_filename=output_filename,
        base_classifier=base_classifier,
        weights_column=weights_column,
        random_state=random_state,
        preservation=preservation,
        remove_cumulative=remove_cumulative,
        nprocs=nprocs,
        verbose=verbose,
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

    output_dir = os.path.dirname(output_filename)
    if not os.path.isdir(output_dir):
        if verbose:
            print(
                f"Output directory does not exist; creating now: {output_dir}",
                file=stderr,
            )
        os.makedirs(output_dir, exist_ok=True)

    training_data = pd.read_csv(training_filename)
    classifier, x, y = create_classifier(
        training_data=training_data,
        random_state=random_state,
        base_classifier=base_classifier,
        threads=nprocs,
        return_arrays=True,
        weights_column=weights_column,
        preservation=preservation,
        remove_cumulative=remove_cumulative,
    )
    if output_filename is not None:
        if verbose:
            print(
                "Saving classifier to file: " + output_filename,
                file=stderr,
            )
        dump(classifier, output_filename)
    return classifier, x, y


def _get_args():
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
    return vars(parser.parse_args())


def _check_args(**kwargs):
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

    for key in kwargs.keys():
        raise TypeError(
            "check_args got an unexpected keyword argument: " + key
        )

    if not os.path.isfile(training_filename):
        raise FileNotFoundError(f"Input file not found: {training_filename}")

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
    }


if __name__ == "__main__":
    main(**_get_args())