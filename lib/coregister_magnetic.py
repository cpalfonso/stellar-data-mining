"""Functions to join deposit and unlabelled point data to
present-day magnetic anomaly raster data.
"""
from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.neighbors import NearestNeighbors

from .misc import (
    _PathLike,
    _PathOrDataFrame,
)

# Default distance threshold is 6 arc-minutes
DEFAULT_DISTANCE_THRESHOLD = 6 / 60  # degrees


def coregister_magnetic(
    data: _PathOrDataFrame,
    filename: Union[_PathLike, xr.Dataset],
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    n_jobs: Optional[int] = None,
) -> pd.DataFrame:
    """Join point data to present-day magnetic anomaly raster.

    Parameters
    ----------
    data : str or DataFrame
        Point dataset.
    filename : str or Dataset
        Magnetic anomaly raster data or filename, in xarray-compatible
        format.
    distance_threshold : float, default: 0.1
        Search radius (in degrees of arc) for assigning raster
        data to points.
    n_jobs : int, optional
        Number of processes to use.

    Returns
    -------
    DataFrame
        The joined dataset.
    """
    if isinstance(data, str):
        data = pd.read_csv(data)
    else:
        data = pd.DataFrame(data)

    def load_dset(dset):
        mag = np.array(dset["z"])
        try:
            grid_lons = np.array(dset["lon"])
        except KeyError:
            grid_lons = np.array(dset["x"])
        try:
            grid_lats = np.array(dset["lat"])
        except KeyError:
            grid_lats = np.array(dset["y"])
        return mag, grid_lons, grid_lats

    if isinstance(filename, xr.Dataset):
        mag, grid_lons, grid_lats = load_dset(filename)
    else:
        with xr.open_dataset(filename) as dset:
            mag, grid_lons, grid_lats = load_dset(dset)

    mag = np.ravel(mag)

    mlons, mlats = np.meshgrid(grid_lons, grid_lats)
    mlons = np.deg2rad(mlons).reshape((-1, 1))
    mlats = np.deg2rad(mlats).reshape((-1, 1))
    mcoords = np.hstack((mlats, mlons))

    neigh = NearestNeighbors(metric="haversine", n_jobs=n_jobs)
    neigh.fit(mcoords)

    point_lons = np.deg2rad(np.array(data["present_lon"]))
    point_lats = np.deg2rad(np.array(data["present_lat"]))
    point_coords = np.hstack(
        (
            point_lats.reshape((-1, 1)),
            point_lons.reshape((-1, 1)),
        )
    )
    _, indices = neigh.radius_neighbors(
        point_coords,
        radius=np.deg2rad(distance_threshold),
        return_distance=True,
        sort_results=True,
    )

    columns = {
        "magnetic_anomaly_mean (nT)": np.nanmean,
        "magnetic_anomaly_min (nT)": np.nanmin,
        "magnetic_anomaly_max (nT)": np.nanmax,
        "magnetic_anomaly_median (nT)": np.nanmedian,
        "magnetic_anomaly_std (nT)": np.nanstd,
    }
    arrays = {i: np.full(data.shape[0], np.nan) for i in columns.keys()}
    columns["magnetic_anomaly_n"] = lambda x: np.size(x) - np.count_nonzero(np.isnan(x))
    arrays["magnetic_anomaly_n"] = np.full(data.shape[0], 0)

    for i in range(data.shape[0]):
        indices_point = indices[i]
        if indices_point.size == 0:
            continue
        anomaly_vals = mag[indices_point]
        for column in columns.keys():
            func = columns[column]
            value = func(anomaly_vals)
            arrays[column][i] = value
    for column in columns.keys():
        data[column] = arrays[column]
    data["magnetic_anomaly_range (nT)"] = (
        data["magnetic_anomaly_max (nT)"]
        - data["magnetic_anomaly_min (nT)"]
    )

    return data
