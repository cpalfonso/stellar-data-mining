"""Functions for training ML models and preparing training data."""
import os
import warnings
from multiprocessing import cpu_count
from sys import stderr

import geopandas as gpd
import numpy as np
import pandas as pd
import pygplates
import rioxarray
import xarray as xr
from joblib import (
    Parallel,
    delayed,
    dump,
)
from pulearn import BaggingPuClassifier
from shapely.geometry import MultiPoint
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from xgboost import XGBClassifier

from .misc import reconstruct_by_topologies

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
    "magnetic_anomaly_max (nT)",
    "magnetic_anomaly_range (nT)",
    "magnetic_anomaly_median (nT)",
    "magnetic_anomaly_mean (nT)",
    "magnetic_anomaly_min (nT)",
    "magnetic_anomaly_std (nT)",
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
    "erosion (m)",
}
CUMULATIVE_COLUMNS = {
    "subducted_plate_volume (m)",
    "subducted_sediment_volume (m)",
    "subducted_water_volume (m)",
    "subducted_carbonates_volume (m)",
}

UNUSED_COLUMNS = COLUMNS_TO_DROP | CORRELATED_COLUMNS | PRESERVATION_COLUMNS

BASE_MODELS = {
    "randomforest": RandomForestClassifier(
        n_jobs=1,
        n_estimators=50,
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
PU_PARAMS = {
    "n_estimators": 100,
    "max_samples": 1.0,
}

DEFAULT_RANDOM_STATE = None
DEFAULT_NPROCS = 1

DIRNAME = os.path.abspath(os.path.dirname(__file__))
DEFAULT_TRAINING_FILENAME = os.path.join(
    DIRNAME,
    "..",
    "prepared_data",
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
    remove_correlated=True,
    remove_cumulative=False,
):
    """Create and (optionally) train a PU classifier.

    Parameters
    ----------
    training_data : str or pandas.DataFrame
        Data frame containing training data.
    base_classifier : str or scikit-learn estimator, default: 'randomforest'
        Base classifier to use for BaggingPuClassifier. Valid string options
        are: 'randomforest', 'gradientboosting', 'adaboost', and 'xgboost'.
    random_state : int, optional
        Seed for random number generator.
    threads : int, default: 1
        Number of processes to use.
    return_arrays : bool, default: False
        Return arrays used to train model (X and y).
    label : str, default: 'label'
        Column name containing labels ('y').
    fit : bool, default: True
        Train the classifier on the provided data.
    pu_kwargs : dict, optional
        Additional keyword arguments to pass to BaggingPuClassifier.
    remove_correlated : bool, default: True
        Remove correlated columns from training data.
    remove_cumulative : bool, default: False
        Remove cumulative subducted quantities columns from training data.

    Returns
    -------
    classifier : BaggingPuClassifier
        The PU classifier (trained if `fit == True`).
    x, y : numpy.ndarray (if `return_arrays == True`)
        x and y are ndarrays of shape (n, m) and (n,), respectively, where
        n is the number of samples and m is the number of features.
    """
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
    )

    n_estimators = pu_kwargs.pop("n_estimators", PU_PARAMS["n_estimators"])
    max_samples = pu_kwargs.pop("max_samples", PU_PARAMS["max_samples"])

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
):
    """Extract training X and y arrays from data frame.

    Parameters
    ----------
    data : str or pandas.DataFrame
        Data frame containing training data.
    label : str, default: 'label'
        Column name containing labels ('y').
    remove_correlated : bool, default: True
        Remove correlated columns from training data.
    remove_cumulative : bool, default: False
        Remove cumulative subducted quantities columns from training data.

    Returns
    -------
    x, y : numpy.ndarray
        x and y are ndarrays of shape (n, m) and (n,), respectively, where
        n is the number of samples and m is the number of features.
    """
    if isinstance(data, str):
        data = pd.read_csv(data)
    else:
        data = pd.DataFrame(data)

    to_drop = COLUMNS_TO_DROP.copy()
    if remove_correlated:
        to_drop = to_drop.union(CORRELATED_COLUMNS)
    if remove_cumulative:
        to_drop = to_drop.union(CUMULATIVE_COLUMNS)
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
    """Downsample unlabelled data to ensure an equal number of labelled and
    unlabelled samples.
    """
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
    """Extract feature importances from a classifier (including ensembles)."""
    try:
        return classifier.feature_importances_
    except AttributeError:
        pass
    importances = [i.feature_importances_ for i in classifier.estimators_]
    importances = np.vstack(importances)
    importances = np.mean(importances, axis=0)
    return importances


def generate_grid_points(
    times,
    resolution,
    polygons_dir,
    topological_features=None,
    rotation_model=None,
    n_jobs=1,
    verbose=False,
):
    """Generate a global grid of points for creating prospectivity maps.

    Parameters
    ----------
    times : sequence of float
        Timesteps at which to generate grid points.
    resolution : float
        Resolution of grid points (degrees).
    polygons_dir : str
        Directory containing subduction zone study area polygon shapefiles.
    topological_features : FeatureCollection
        Topological features for plate reconstruction.
    rotation_model : RotationModel
        Rotation model for plate reconstruction.
    n_jobs : int, default: 1
        Number of processes to use.
    verbose : bool, default: False
        Print log to stderr.

    Returns
    -------
    pandas.DataFrame
        Data frame containing the regular grid points.
    """
    n_jobs = int(n_jobs)
    if n_jobs == 0:
        raise ValueError("n_jobs must not be zero")
    elif n_jobs < 0:
        n_jobs = cpu_count() + n_jobs + 1

    # Earlier times take longer to reconstruct, so ensure they are
    # evenly split between processes
    times = np.array(times)
    times_split = [
        times[i::n_jobs]
        for i in range(n_jobs)
    ]

    with Parallel(n_jobs, verbose=int(verbose)) as parallel:
        out = parallel(
            delayed(_grid_points_subset)(
                times=t,
                resolution=resolution,
                polygons_dir=polygons_dir,
                topological_features=topological_features,
                rotation_model=rotation_model,
                verbose=verbose,
            )
            for t in times_split
        )
    out = pd.concat(out, ignore_index=True)
    out = out.drop(columns="index", errors="ignore")
    return out


def _grid_points_subset(
    times,
    resolution,
    polygons_dir,
    topological_features=None,
    rotation_model=None,
    verbose=False,
):
    if (
        topological_features is not None
        and not isinstance(
            topological_features,
            pygplates.FeatureCollection,
        )
    ):
        topological_features = pygplates.FeatureCollection(
            pygplates.FeaturesFunctionArgument(
                topological_features
            ).get_features()
        )
    if (
        rotation_model is not None
        and not isinstance(
            rotation_model,
            pygplates.RotationModel,
        )
    ):
        rotation_model = pygplates.RotationModel(rotation_model)

    out = [
        _grid_points_time(
            time=t,
            resolution=resolution,
            polygons_dir=polygons_dir,
            topological_features=topological_features,
            rotation_model=rotation_model,
            verbose=verbose,
        )
        for t in times
    ]
    out = pd.concat(out, ignore_index=True)
    return out


def _grid_points_time(
    time,
    resolution,
    polygons_dir,
    topological_features=None,
    rotation_model=None,
    verbose=False,
):
    if (
        topological_features is not None
        and not isinstance(
            topological_features,
            pygplates.FeatureCollection,
        )
    ):
        topological_features = pygplates.FeatureCollection(
            pygplates.FeaturesFunctionArgument(
                topological_features
            ).get_features()
        )
    if (
        rotation_model is not None
        and not isinstance(
            rotation_model,
            pygplates.RotationModel,
        )
    ):
        rotation_model = pygplates.RotationModel(rotation_model)

    polygons_filename = os.path.join(
        polygons_dir, f"study_area_{time:0.0f}Ma.shp"
    )
    gdf = gpd.read_file(polygons_filename)
    polygons = gdf.geometry
    minx, miny, maxx, maxy = polygons.total_bounds

    minx = (int(minx / resolution) - 1) * resolution
    maxx = (int(maxx / resolution) + 1) * resolution
    miny = (int(miny / resolution) - 1) * resolution
    maxy = (int(maxy / resolution) + 1) * resolution

    lons = np.arange(minx, maxx + resolution, resolution)
    lats = np.arange(miny, maxy + resolution, resolution)

    mlons, mlats = np.meshgrid(lons, lats)
    mlons = mlons.reshape((-1, 1))
    mlats = mlats.reshape((-1, 1))
    coords = np.column_stack((mlons, mlats))
    mp = MultiPoint(coords)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        intersection = polygons.unary_union.intersection(mp)
    intersection_coords = np.row_stack([i.coords for i in intersection.geoms])
    plons = intersection_coords[:, 0]
    plats = intersection_coords[:, 1]

    if time == 0.0:
        present_lons = np.array(plons)
        present_lats = np.array(plats)
    elif topological_features is None or rotation_model is None:
        present_lats = np.full_like(plats, np.nan)
        present_lons = np.full_like(plons, np.nan)
    else:
        present_day_coords = reconstruct_by_topologies(
            topological_features,
            rotation_model,
            np.fliplr(intersection_coords),
            start_time=float(time),
            end_time=0.0,
            time_step=1.0,
        )
        present_lats = present_day_coords[:, 0]
        present_lons = present_day_coords[:, 1]

    out = pd.DataFrame(
        {
            "lon": plons,
            "lat": plats,
            "present_lon": present_lons,
            "present_lat": present_lats,
            "age (Ma)": time,
        }
    )
    return out


def calculate_probabilities(
    point_data,
    classifier,
    remove_correlated=True,
    remove_cumulative=False,
    label="label",
):
    """Calculate probabilities from grid point data and classifier.

    Parameters
    ----------
    point_data : str or pandas.DataFrame
        Data frame containing grid point data.
    classifier : scikit-learn estimator
        Trained classifier.
    remove_correlated : bool, default: True
        Remove correlated columns from training data.
    remove_cumulative : bool, default: False
        Remove cumulative subducted quantities columns from training data.
    label : str, default: 'label'

    Returns
    -------
    pandas.DataFrame
        Data frame with the following columns:
        - 'lon'
        - 'lat'
        - 'age (Ma)'
        - 'probability'
    """
    if isinstance(point_data, str):
        point_data = pd.read_csv(point_data)
    else:
        point_data = pd.DataFrame(point_data)

    to_drop = COLUMNS_TO_DROP.copy()
    to_drop.add(label)
    if remove_correlated:
        to_drop = to_drop.union(CORRELATED_COLUMNS)
    if remove_cumulative:
        to_drop = to_drop.union(CUMULATIVE_COLUMNS)
    to_drop = to_drop.union(PRESERVATION_COLUMNS)
    to_drop = to_drop.difference(
        {
            "lon",
            "lat",
            "age (Ma)",
        }
    )

    point_data = point_data.sort_index(axis="columns")
    point_data = point_data.drop(columns=list(to_drop), errors="ignore")
    point_data = point_data.dropna()
    lons = np.array(point_data["lon"])
    lats = np.array(point_data["lat"])
    ages = np.array(point_data["age (Ma)"])
    x = np.array(
        point_data.drop(columns=["lon", "lat", "age (Ma)"], errors="ignore")
    )
    probs = classifier.predict_proba(x)[:, 1].flatten()
    out = pd.DataFrame(
        {
            "lon": lons,
            "lat": lats,
            "age (Ma)": ages,
            "probability": probs,
        }
    )
    return out


def create_grids(
    data,
    output_dir,
    resolution=None,
    extent=None,
    threads=1,
    verbose=False,
    column="probability",
    filename_format="probability_grid_{}Ma.nc",
):
    """Create raster grids from grid point data.

    Parameters
    ----------
    data : str or pandas.DataFrame
        Data frame containing the following columns:
        - 'age (Ma)'
        - 'lon'
        - 'lat'
        - {column} (by default, 'probability')

    output_dir : str
        Write netCDF output files to this directory.
    resolution : float, optional
        The resolution of the raster grids. By default, this will be
        determined from the point data.
    extent : tuple of float, optional
        The extent of the raster grids. By default, this will be determined
        from the point data.
    threads : int, default: 1
        Number of processes to use.
    verbose : bool, default: False
        Print log to stderr.
    column : str, default: 'probability'
        The column containing the values to grid.
    """
    if isinstance(data, str):
        data = pd.read_csv(data)
    else:
        data = pd.DataFrame(data)

    times = data["age (Ma)"].unique()

    if threads == 1:
        for time in times:
            _create_grid_time(
                time=time,
                data_lons=np.array(data[data["age (Ma)"] == time]["lon"]),
                data_lats=np.array(data[data["age (Ma)"] == time]["lat"]),
                data_values=np.array(data[data["age (Ma)"] == time][column]),
                resolution=resolution,
                output_dir=output_dir,
                extent=extent,
                verbose=verbose,
                filename_format=filename_format,
            )
    else:
        with Parallel(threads, verbose=10 * int(verbose)) as p:
            p(
                delayed(_create_grid_time)(
                    time=time,
                    data_lons=np.array(data[data["age (Ma)"] == time]["lon"]),
                    data_lats=np.array(data[data["age (Ma)"] == time]["lat"]),
                    data_values=np.array(data[data["age (Ma)"] == time][column]),
                    resolution=resolution,
                    output_dir=output_dir,
                    extent=extent,
                    verbose=verbose,
                    filename_format=filename_format,
                )
                for time in times
            )


create_probability_grids = create_grids


def _create_grid_time(
    time,
    data_lons,
    data_lats,
    data_values,
    resolution,
    output_dir,
    extent=None,
    verbose=False,
    filename_format="probability_grid_{}Ma.nc",
):
    time = int(np.around(time))
    output_filename = os.path.join(
        output_dir, filename_format.format(time)
    )

    if extent is None:
        xmin = np.nanmin(data_lons)
        xmax = np.nanmax(data_lons)
        ymin = np.nanmin(data_lats)
        ymax = np.nanmax(data_lats)
    elif extent == "global":
        xmin, xmax, ymin, ymax = -180, 180, -90, 90
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
    dset.rio.write_crs(4326, inplace=True)
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
