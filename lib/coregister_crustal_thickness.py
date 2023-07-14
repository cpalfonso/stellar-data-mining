import os

from joblib import delayed, Parallel
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import xarray as xr

DEFAULT_DISTANCE_THRESHOLD = 3.0  # degrees


def run_coregister_crustal_thickness(
    point_data,
    input_dir,
    distance_threshold=DEFAULT_DISTANCE_THRESHOLD,
    n_jobs=1,
    verbose=False,
):
    with Parallel(n_jobs, verbose=int(verbose)) as parallel:
        out = parallel(
            delayed(coregister_crustal_thickness)(
                time=t,
                input_dir=input_dir,
                df=d,
                distance_threshold=distance_threshold,
            )
            for t, d in point_data.groupby("age (Ma)")
        )

    out = pd.DataFrame(pd.concat(out, ignore_index=True))
    if "label" in out.columns:
        sort_by = ["label", "age (Ma)"]
    else:
        sort_by = "age (Ma)"
    out = out.sort_values(by=sort_by, ignore_index=True)
    return out


def coregister_crustal_thickness(
    time,
    input_dir,
    df,
    distance_threshold=DEFAULT_DISTANCE_THRESHOLD,
):
    df = df.copy()
    df = df[df["age (Ma)"] == time]

    input_filename = os.path.join(
        input_dir, "crustal_thickness_{}Ma.nc".format(time)
    )
    with xr.open_dataset(input_filename) as dset:
        thickness = np.array(dset["z"])
        try:
            grid_lons = np.array(dset["lon"])
        except KeyError:
            grid_lons = np.array(dset["x"])
        try:
            grid_lats = np.array(dset["lat"])
        except KeyError:
            grid_lats = np.array(dset["y"])

    mlons, mlats = np.meshgrid(grid_lons, grid_lats)
    mlons = np.deg2rad(mlons[~np.isnan(thickness)])
    mlats = np.deg2rad(mlats[~np.isnan(thickness)])
    thickness = thickness[~np.isnan(thickness)]

    mcoords = np.hstack(
        (
            mlats.reshape((-1, 1)),
            mlons.reshape((-1, 1)),
        )
    )
    neigh = NearestNeighbors(metric="haversine")
    neigh.fit(mcoords)

    point_lons = np.deg2rad(np.array(df["lon"]))
    point_lats = np.deg2rad(np.array(df["lat"]))
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
        "crustal_thickness_mean (m)": np.nanmean,
        "crustal_thickness_min (m)": np.nanmin,
        "crustal_thickness_max (m)": np.nanmax,
        "crustal_thickness_median (m)": np.nanmedian,
        "crustal_thickness_std (m)": np.nanstd,
        "crustal_thickness_n": np.size,
    }
    arrays = {
        i: np.full(df.shape[0], np.nan) for i in columns.keys()
        if i != "crustal_thickness_n"
    }
    arrays["crustal_thickness_n"] = np.full(df.shape[0], 0, dtype="int32")
    # for column in columns:
        # df[column] = np.full(df.shape[0], np.nan)

    for i in range(df.shape[0]):
        indices_point = indices[i]
        if indices_point.size == 0:
            continue
        data = thickness[indices_point]
        for column in columns.keys():
            func = columns[column]
            value = func(data)
            arrays[column][i] = value

    for column in columns.keys():
        df[column] = arrays[column]
    df["crustal_thickness_range (m)"] = (
        df["crustal_thickness_max (m)"]
        - df["crustal_thickness_min (m)"]
    )

    return df
