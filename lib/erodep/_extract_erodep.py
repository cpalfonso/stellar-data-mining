import glob
import os
import warnings
from functools import lru_cache
from sys import stderr
from typing import (
    Callable,
    Optional,
    Tuple,
    Type,
    Union,
)

import numpy as np
import pandas as pd
import xarray as xr
from gplately.tools import lonlat2xyz
from numpy.typing import (
    ArrayLike,
    NDArray,
)
from sklearn.neighbors import (
    KNeighborsRegressor,
    NearestNeighbors,
    RadiusNeighborsRegressor,
)

from ..misc import (
    _PathLike,
    _PathOrDataFrame,
    load_data,
)

__all__ = [
    "time_from_filename",
    "filename_from_time",
    "extract_erodep",
    "extract_lat_lon",
    "create_regressor",
    "interpolate_values",
    "calculate_erodep",
]

_DIRNAME = os.path.abspath(os.path.dirname(__file__))
_DEFAULT_DIR = os.path.abspath(os.path.join(_DIRNAME, "..", "..", "erodep_maps"))


def time_from_filename(filename: _PathLike) -> float:
    basename = os.path.basename(filename)
    return float(
        basename.split("erodep")[-1].split("Ma.nc")[0]
    )


def filename_from_time(time: float, dir: Optional[_PathLike] = None) -> str:
    basename = f"erodep{time:0.0f}Ma.nc"
    if dir is None:
        return basename
    return os.path.join(dir, f"erodep{time:0.0f}Ma.nc")


@lru_cache(maxsize=2)
def extract_erodep(time: float, dir: str = _DEFAULT_DIR) -> NDArray[np.floating]:
    time = np.around(time)
    if time < 0.0:
        raise ValueError(f"Invalid time: {time}")
    timestep_total = _erodep_timestep(time=time, dir=dir)
    if time == 0.0:
        return timestep_total
    return timestep_total + extract_erodep(time - 1.0, dir=dir)


@lru_cache(maxsize=5)
def _erodep_timestep(time, dir=_DEFAULT_DIR, dt=1.0):
    if time < 0.0:
        raise ValueError(f"Invalid time: {time}")
    if time == 0.0:
        lats, lons = extract_lat_lon(dir=dir)
        shape = (lats.size, lons.size)
        return np.zeros(shape, dtype=np.float_)

    erodep_rate = _erorate_timestep(time=time, dir=dir)
    timestep_total = erodep_rate * dt
    return timestep_total


@lru_cache(maxsize=5)
def _erorate_timestep(time, dir=_DEFAULT_DIR):
    if time < 0.0:
        raise ValueError(f"Invalid time: {time}")

    filenames = _get_erodep_filenames(dir=dir)
    erodep_times = {time_from_filename(_) for _ in filenames}
    erodep_times = np.array(sorted(erodep_times))

    if time in erodep_times:
        filename = filename_from_time(time, dir=dir)
        with xr.open_dataset(filename) as dset:
            erodep_rate = np.array(dset["erate"])
    else:
        t0 = np.max(erodep_times[erodep_times < time])
        t1 = np.min(erodep_times[erodep_times > time])

        dt0 = time - t0
        dt1 = t1 - time

        f0 = filename_from_time(t0, dir=dir)
        with xr.open_dataset(f0) as dset0:
            erorate0 = np.array(dset0["erate"])
        f1 = filename_from_time(t1, dir=dir)
        with xr.open_dataset(f1) as dset1:
            erorate1 = np.array(dset1["erate"])
        erodep_rate = (
            erorate0 * dt1  # (dt0 + dt1) - dt0
            + erorate1 * dt0
        ) / (dt0 + dt1)  # weighted mean

    erodep_rate *= 1.0e6 * 1.0e-3  # mm/yr to m/Myr
    return erodep_rate


@lru_cache(maxsize=1)
def extract_lat_lon(dir: str = _DEFAULT_DIR) -> Tuple[
    NDArray[np.floating],
    NDArray[np.floating],
]:
    filenames = _get_erodep_filenames(dir=dir)
    filename = filenames[0]
    with xr.open_dataset(filename) as dset:
        lons = np.array(dset["longitude"])
        lats = np.array(dset["latitude"])
    return lats, lons


@lru_cache(maxsize=1)
def _get_erodep_filenames(dir=_DEFAULT_DIR):
    return glob.glob(os.path.join(dir, "erodep*Ma.nc"))


def create_regressor(
    data: ArrayLike,
    lons: Optional[ArrayLike] = None,
    lats: Optional[ArrayLike] = None,
    type: Union[
        Type[RadiusNeighborsRegressor],
        Type[KNeighborsRegressor],
    ] = RadiusNeighborsRegressor,
    **kwargs
) -> Union[RadiusNeighborsRegressor, KNeighborsRegressor]:
    radius = kwargs.pop("radius", np.deg2rad(0.5))
    n_neighbors = kwargs.pop("n_neighbors", 1)
    metric = kwargs.pop("metric", "haversine")
    if type is RadiusNeighborsRegressor:
        neigh = type(radius=radius, metric=metric, **kwargs)
    elif type is KNeighborsRegressor:
        neigh = type(n_neighbors=n_neighbors, metric=metric, **kwargs)
    else:
        raise TypeError(f"Invalid type: {type}")

    if lons is None:
        lons = np.linspace(-np.pi, np.pi, np.shape(data)[0])
    if lats is None:
        lats = np.linspace(-0.5 * np.pi, 0.5 * np.pi, np.shape(data)[1])
    mlons, mlats = np.meshgrid(lons, lats)
    x = np.column_stack(
        (
            np.ravel(mlats),
            np.ravel(mlons),
        )
    )
    # x = np.deg2rad(x)
    y = np.ravel(data)

    valid_points = ~np.isnan(y)
    x = x[valid_points, :]
    y = y[valid_points]

    neigh.fit(x, y)
    return neigh


def interpolate_values(
    lons: ArrayLike,
    lats: ArrayLike,
    grid_values: ArrayLike,
    radius: float,
    neigh: Optional[NearestNeighbors] = None,
    grid_lons: Optional[ArrayLike] = None,
    grid_lats: Optional[ArrayLike] = None,
    weighted: bool = False,
    n_jobs: int = 1,
    metric: Union[str, Callable] = "haversine",
) -> Tuple[NDArray, NearestNeighbors]:
    if metric != "haversine":
        # Convert radius to Cartesian distance
        radius = (2 * (1 - np.cos(radius))) ** 0.5

    if neigh is None:
        if grid_lons is None or grid_lats is None:
            raise TypeError(
                "Must provide either `neigh` or `grid_lons` and `grid_lats`"
            )
        neigh = NearestNeighbors(
            radius=radius,
            metric=metric,
            n_jobs=n_jobs,
        )
        if np.shape(grid_lons) != np.shape(grid_lats):
            grid_lons, grid_lats = np.meshgrid(grid_lons, grid_lats)
        if metric == "haversine":
            x_train = np.deg2rad(
                np.column_stack(
                    (
                        np.ravel(grid_lats),
                        np.ravel(grid_lons),
                    )
                )
            )
        else:
            x_train = np.column_stack(
                lonlat2xyz(
                    np.ravel(grid_lons),
                    np.ravel(grid_lats),
                    degrees=True,
                )
            )
        neigh.fit(x_train)

    if metric == "haversine":
        x = np.deg2rad(
            np.column_stack(
                (
                    np.ravel(lats),
                    np.ravel(lons),
                )
            )
        )
    else:
        x = np.column_stack(
            lonlat2xyz(
                np.ravel(lons),
                np.ravel(lats),
                degrees=True,
            )
        )

    distances, indices = neigh.radius_neighbors(
        x,
        radius=radius,
        return_distance=True,
    )

    grid_values = np.ravel(grid_values)
    out = np.full(np.size(lons), np.nan)
    for i, (dists, inds) in enumerate(zip(distances, indices)):
        vals_point = grid_values[inds]
        tmp_filter = ~np.isnan(vals_point)
        vals_point = vals_point[tmp_filter]

        if weighted:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                weights_point = 1.0 / dists
                weights_point = weights_point[tmp_filter]
                weights_point[np.isinf(weights_point)] = 1.0e6
                out[i] = np.sum(vals_point * weights_point) / np.sum(weights_point)
        else:
            out[i] = np.nanmean(vals_point)

    return out, neigh


def calculate_erodep(
    data: _PathOrDataFrame,
    input_dir: _PathLike,
    weighted: bool = True,
    radius=np.deg2rad(2.5),
    n_jobs: int = 1,
    metric: str = "haversine",
    column_name: str = "erosion (m)",
    clip: bool = True,
    verbose: bool = False,
):
    data = load_data(data, verbose=verbose)

    erodep_col = pd.Series(
        np.full(data.shape[0], np.nan),
        index=data.index,
        name=column_name,
    )
    neigh = None
    if verbose:
        print("", end="", flush=True)
    for age, data_age in data.groupby("age (Ma)"):
        if verbose:
            print(f"\rWorking on {age:0.0f} Ma", end="", flush=True)
        try:
            if age == 0:
                point_erodep = np.full(data_age.shape[0], 0.0)
            else:
                total_erodep = extract_erodep(
                    time=age,
                    dir=input_dir,
                )
                lats, lons = extract_lat_lon(dir=input_dir)
                point_erodep, neigh = interpolate_values(
                    lons=data_age["present_lon"],
                    lats=data_age["present_lat"],
                    grid_values=total_erodep,
                    radius=radius,
                    neigh=neigh,
                    grid_lons=lons,
                    grid_lats=lats,
                    weighted=weighted,
                    n_jobs=n_jobs,
                    metric=metric,
                )
            for i, val in zip(data_age.index, point_erodep):
                erodep_col.at[i] = val
        except Exception as err:
            print("\n", end="", flush=True)
            raise err
    data = data.join(erodep_col)
    if clip:
        data[column_name] = np.clip(-1 * data[column_name], 0.0, np.inf)
    return data
