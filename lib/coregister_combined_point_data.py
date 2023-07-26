"""Functions to join deposit and unlabelled point data to subduction
zone kinematics data.
"""
import os
from sys import stderr

import numpy as np
import pandas as pd
from gplately import EARTH_RADIUS
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors


def run_coregister_combined_point_data(
    point_data,
    subduction_data,
    output_filename=None,
    n_jobs=1,
    verbose=False,
):
    """Join point data to subduction zone data.

    Parameters
    ----------
    point_data : str or DataFrame
        Point dataset.
    subduction_data : str or DataFrame
        Subduction zone dataset.
    output_filename : str, optional
        If provided, write the joined data to a CSV file.
    n_jobs : int
        Number of processes to use.
    verbose : bool, default: False
        Print log to stderr.

    Returns
    -------
    DataFrame
        The joined dataset.
    """
    if isinstance(point_data, str):
        if verbose:
            print(
                "Loading point data from file: " + point_data,
                file=stderr,
            )
        point_data = pd.read_csv(point_data)
    else:
        point_data = pd.DataFrame(point_data)

    if isinstance(subduction_data, str):
        if verbose:
            print(
                "Loading subduction data from file: " + subduction_data,
                file=stderr,
            )
        subduction_data = pd.read_csv(subduction_data)
    else:
        subduction_data = pd.DataFrame(subduction_data)

    times = point_data["age (Ma)"].unique()

    with Parallel(n_jobs, verbose=int(verbose)) as parallel:
        out = parallel(
            delayed(coregister_combined_point_data)(
                time=time,
                points=point_data[point_data["age (Ma)"] == time],
                szs=subduction_data[
                    subduction_data["age (Ma)"] == int(np.around(time))
                ],
            )
            for time in times
        )

    out = pd.DataFrame(pd.concat(out, ignore_index=True))

    out = out.drop(columns="index", errors="ignore")
    if "label" in out.columns:
        sort_by = ["label", "age (Ma)"]
    else:
        sort_by = "age (Ma)"
    out = out.sort_values(by=sort_by, ignore_index=True)
    if output_filename is not None:
        output_dir = os.path.dirname(os.path.abspath(output_filename))
        if not os.path.isdir(output_dir):
            if verbose:
                print(
                    "Output directory does not exist; creating now: "
                    + output_dir,
                    file=stderr,
                )
            os.makedirs(output_dir, exist_ok=True)
        if verbose:
            print(
                "Writing output to file: "
                + os.path.basename(output_filename),
                file=stderr,
            )
        out.to_csv(output_filename, index=False)
    return out


def coregister_combined_point_data(time, points, szs):
    """Coregister datasets at a give time.

    Parameters
    ----------
    time : float
    points : DataFrame
        Point dataset.
    szs : DataFrame
        Subduction zone dataset.
    """
    points = points.copy()
    szs = szs.copy().reset_index()

    points = points[points["age (Ma)"] == time]
    szs = szs[szs["age (Ma)"] == int(np.around(time))]

    columns_to_add = set(szs.columns.values) - set(points.columns.values)
    for column in columns_to_add:
        points[column] = np.nan

    lon_points = np.array(points["lon"]).reshape((-1, 1))
    lat_points = np.array(points["lat"]).reshape((-1, 1))
    coords_points = np.deg2rad(np.hstack((lat_points, lon_points)))

    lon_data = np.array(szs["lon"]).reshape((-1, 1))
    lat_data = np.array(szs["lat"]).reshape((-1, 1))
    coords_data = np.deg2rad(np.hstack((lat_data, lon_data)))

    neigh = NearestNeighbors(metric="haversine", n_jobs=1)
    neigh.fit(coords_data)

    distances, indices = neigh.kneighbors(
        coords_points, n_neighbors=1, return_distance=True
    )
    # distances = np.rad2deg(distances).flatten()
    distances = distances.flatten() * EARTH_RADIUS
    indices = indices.flatten()

    for column in columns_to_add:
        for i_points, i_szs in zip(points.index, indices):
            points.at[i_points, column] = szs.at[i_szs, column]
    points["distance_to_trench (km)"] = distances

    return points
