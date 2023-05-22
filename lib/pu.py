import os
from sys import stderr

import numpy as np
import pandas as pd
import xarray as xr
from joblib import (
    Parallel,
    delayed,
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


def create_classifier(
    training_data,
    base_classifier="randomforest",
    random_state=None,
    threads=1,
    return_arrays=False,
    label="label",
    fit=True,
    pu_kwargs={},
    weights_column=None,
    downsample=True,
    preservation=False,
    remove_correlated=True,
    remove_cumulative=False,
    num_negatives=0,
):
    random_state = np.random.default_rng(random_state)

    if not isinstance(base_classifier, BaseEstimator):
        if base_classifier not in BASE_MODELS.keys():
            raise ValueError("Invalid base_classifier: {}".format(base_classifier))
        base_classifier = BASE_MODELS[base_classifier]

    out = prepare_training_arrays(
        training_data=training_data,
        label=label,
        random_state=random_state,
        weights_column=weights_column,
        downsample=downsample,
        preservation=preservation,
        remove_correlated=remove_correlated,
        remove_cumulative=remove_cumulative,
        negatives=num_negatives,
    )
    if weights_column is None:
        x, y = out
        weights = None
    else:
        x, y, weights = out

    if "n_estimators" not in pu_kwargs:
        pu_kwargs["n_estimators"] = 250
    if "max_samples" not in pu_kwargs:
        pu_kwargs["max_samples"] = 1.0

    bc = BaggingPuClassifier(
        base_classifier,
        n_jobs=threads,
        random_state=np.random.RandomState(random_state.bit_generator),
        **pu_kwargs,
    )
    if fit:
        bc.fit(x, y, sample_weight=weights)
    if return_arrays:
        return bc, x, y
    return bc


def prepare_training_arrays(
    training_data,
    label="label",
    random_state=None,
    return_dataframe=False,
    weights_column=None,
    downsample=True,
    preservation=False,
    remove_correlated=True,
    remove_cumulative=False,
    negatives=0,
):
    random_state = np.random.default_rng(random_state)

    try:
        training_data = pd.read_csv(training_data, low_memory=False)
    except Exception:
        training_data = pd.DataFrame(training_data)

    negatives = int(negatives)
    if negatives < 0:
        raise ValueError("`negatives` must be greater than or equal to zero.")
    if negatives > 0:
        negative_data = training_data[training_data["label"] == "negative"]
        negative_indices = random_state.choice(negative_data.shape[0], size=negatives)
        negative_data = negative_data.iloc[negative_indices, :]
    else:
        negative_data = None

    if "set" in training_data.columns:
        training_data = training_data[training_data["set"] == "train"]
    training_data = training_data[
        training_data["label"].isin(
            {
                "positive",
                "unlabeled",
                "unlabelled",
            }
        )
    ]
    if negative_data is not None:
        training_data = pd.concat((training_data, negative_data))
    if weights_column is not None:
        weights = training_data[weights_column].copy()
        valid_weights = weights[~weights.isna()]
        num_to_fill = weights.isna().sum()
        weights[weights.isna()] = random_state.choice(
            valid_weights,
            size=num_to_fill,
        )
        training_data["sample_weight"] = weights
    else:
        weights = None

    training_data = training_data.drop(columns=COLUMNS_TO_DROP, errors="ignore")
    if remove_correlated:
        training_data = training_data.drop(columns=CORRELATED_COLUMNS, errors="ignore")
    if remove_cumulative:
        training_data = training_data.drop(columns=CUMULATIVE_COLUMNS, errors="ignore")

    if not preservation:
        training_data = training_data.drop(columns=PRESERVATION_COLUMNS, errors="ignore")

    training_data = training_data.dropna()
    if downsample:
        training_data = undersample(
            training_data,
            label,
            random_state=random_state,
            skip_columns="negative" if negatives > 0 else None,
        )

    training_data = training_data.sort_index(axis="columns")

    y = np.array(
        training_data[label].apply(
            lambda x: 1 if x == "positive" else 0
        )
    )

    features = []
    for column in sorted(training_data.columns.values):
        if column == "label" or (column == "sample_weight"):
            continue
        arr = np.array(training_data[column]).reshape((-1, 1))
        features.append(arr)
    x = np.hstack(features)

    out = [x, y]
    if weights is not None:
        weights = np.array(training_data["sample_weight"])
    if return_dataframe:
        out.append(training_data)
    if weights is not None:
        out.append(weights)
    return tuple(out)


def undersample(data, label="label", random_state=None, skip_columns=None):
    if isinstance(random_state, np.random.RandomState):
        random_state = np.random.default_rng(seed=random_state._bit_generator)
    elif not isinstance(random_state, np.random.Generator):
        random_state = np.random.default_rng(seed=random_state)

    if skip_columns is not None:
        if isinstance(skip_columns, str):
            skip_columns = [skip_columns]
        skipped_inds = data[label].isin(skip_columns)
        skipped_data = data[skipped_inds]
        data = data[~skipped_inds]
    else:
        skipped_data = None

    gb = data.groupby(label)
    sizes = gb.size()
    n = sizes.min()
    out = []
    for label_value, label_data in gb:
        if sizes.loc[label_value] == n:
            tmp = label_data
        else:
            tmp_indices = random_state.choice(label_data.shape[0], n)
            tmp = label_data.iloc[tmp_indices, :]
        out.append(tmp)
    if skipped_data is not None:
        out.append(skipped_data)
    return pd.concat(out)


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
