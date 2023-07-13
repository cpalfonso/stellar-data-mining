import concurrent.futures
import os
from sys import stderr

import geopandas as gpd
import numpy as np
import pandas as pd
import pygplates
from gplately.tools import xyz2lonlat
from joblib import Parallel, delayed
from shapely.geometry import Point

from .misc import reconstruct_by_topologies

# North/South America bounds
DEFAULT_LAT_SPLIT = 12.0
DEFAULT_LON_MIN = -180.0
DEFAULT_LON_MAX = -30.0

DEFAULT_UNLABELED_POINTS_FILENAME = None


def generate_unlabelled_points(
    times,
    input_dir,
    num,
    threads=1,
    output_filename=DEFAULT_UNLABELED_POINTS_FILENAME,
    seed=None,
    topological_features=None,
    rotation_model=None,
    verbose=False,
):
    seq = np.random.SeedSequence(entropy=seed)
    rngs = [np.random.default_rng(i) for i in seq.spawn(threads)]
    times_split = np.array_split(times, threads)

    with Parallel(threads, verbose=int(verbose)) as p:
        results = p(
            delayed(_multiple_timesteps)(
                times=t,
                input_dir=input_dir,
                topological_features=topological_features,
                rotation_model=rotation_model,
                num=num,
                rng=rng,
            )
            for t, rng in zip(times_split, rngs)
        )
    results_flattened = []
    for i in results:
        results_flattened.extend(i)
    results = results_flattened
    del results_flattened

    results = pd.concat(results, ignore_index=True).sort_values(by="age (Ma)")
    results["source"] = "random"
    results["label"] = "unlabelled"
    def func(row):
        if row["source"] != "random":
            try:
                return row["region"]
            except KeyError:
                return np.nan
        lon = row["present_lon"]
        lat = row["present_lat"]
        if not (lon >= DEFAULT_LON_MIN and lon <= DEFAULT_LON_MAX):
            region = "other"
        elif lat >= DEFAULT_LAT_SPLIT:
            region = "NAm"
        else:
            region = "SAm"
        return region
    results["region"] = results.apply(func, axis="columns")

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
                "Writing output to file: " + str(output_filename),
                file=stderr,
            )
        results.to_csv(output_filename, index=False)
    return results


def _multiple_timesteps(
        times,
        input_dir,
        num,
        rng,
        topological_features,
        rotation_model,
):
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(seed=rng)

    if not isinstance(topological_features, pygplates.FeatureCollection):
        topological_features = pygplates.FeatureCollection(
            pygplates.FeaturesFunctionArgument(
                topological_features
            ).get_features()
        )
    if not isinstance(rotation_model, pygplates.RotationModel):
        rotation_model = pygplates.RotationModel(rotation_model)

    out = []
    for time in times:
        out.append(
            _generate_points_timestep(
                time=time,
                input_dir=input_dir,
                topological_features=topological_features,
                rotation_model=rotation_model,
                num=num,
                rng=rng,
            )
        )
    return out


def _generate_points_timestep(
    time,
    input_dir,
    topological_features,
    rotation_model,
    num,
    rng,
):
    input_filename = os.path.join(
        input_dir, "study_area_{}Ma.shp".format(time)
    )
    gdf = gpd.read_file(input_filename)

    points = np.full((num, 2), np.nan)
    to_fill = np.where(np.any(np.isnan(points), axis=1))[0]
    num_to_fill = to_fill.size
    while num_to_fill > 0:
        generated_points = generate_points(
            n=num_to_fill,
            output_format="degrees",
            order="lonlat",
            threads=1,
            rng=rng,
        )
        generated_points[
            ~_points_in_polygons(generated_points, gdf["geometry"])
        ] = np.nan
        points[to_fill] = generated_points

        to_fill = np.where(np.any(np.isnan(points), axis=1))[0]
        num_to_fill = to_fill.size

    if topological_features is not None and rotation_model is not None:
        if time == 0.0:
            present_day_coords = np.fliplr(points)
        else:
            present_day_coords = reconstruct_by_topologies(
                topological_features,
                rotation_model,
                np.fliplr(points),
                start_time=float(time),
                end_time=0.0,
                time_step=1.0,
            )
    else:
        present_day_coords = np.full_like(points, np.nan)

    out = pd.DataFrame(
        {
            "lon": points[:, 0],
            "lat": points[:, 1],
            "present_lon": present_day_coords[:, 1],
            "present_lat": present_day_coords[:, 0],
            "age (Ma)": time,
        }
    )
    return out


def generate_points(
    n=1, output_format="radians", order="lonlat", threads=1, rng=None
):
    """Generate uniformly-distributed points on the unit sphere."""
    valid_output_formats = {
        "radians",
        "degrees",
        "xyz",
    }
    valid_orders = {
        "lonlat",
        "latlon",
    }

    seed = None

    output_format = str(output_format).lower()
    if output_format not in valid_output_formats:
        raise ValueError("Invalid `output_format`: " + output_format)
    order = str(order).lower()
    if order not in valid_orders:
        raise ValueError("Invalid `order`: " + order)

    if threads == 1:
        if rng is None:
            rng = np.random.default_rng(seed=seed)
        if not isinstance(rng, np.random.Generator):
            raise TypeError("Invalid `rng` type: " + str(type(rng)))
        xyz = _generate_points(n=n, rng=rng)
    else:
        if rng is None:
            rng = np.random.SeedSequence(seed)
        if not isinstance(rng, np.random.SeedSequence):
            raise TypeError("Invalid `rng` type: " + str(type(rng)))
        xyz = _generate_points_threaded(n=n, threads=threads, seq=rng)

    if output_format == "xyz":
        return xyz
    lon, lat = xyz2lonlat(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], degrees=False)
    lon = np.array(lon)
    lat = np.array(lat)
    out = np.hstack((lon.reshape((-1, 1)), lat.reshape((-1, 1))))
    if order == "latlon":
        out = np.fliplr(out)
    if output_format == "degrees":
        out = np.rad2deg(out)
    return out


def _generate_points(n=1, rng=None):
    seed = None

    if rng is None:
        rng = np.random.default_rng(seed=seed)
    xyz = np.zeros((n, 3))
    zero_rows = np.where(np.all(np.isclose(xyz, 0.0), axis=1))[0]
    num_rows = zero_rows.size
    while num_rows > 0:
        tmp = rng.standard_normal(size=(num_rows, 3))
        xyz[zero_rows] = tmp
        zero_rows = np.where(np.all(np.isclose(xyz, 0.0), axis=1))[0]
        num_rows = zero_rows.size

    xyz /= np.sqrt((xyz ** 2).sum(axis=1)).reshape((-1, 1))
    return xyz


def _generate_points_threaded(n=1, threads=2, seq=None):
    seed = None

    if seq is None:
        seq = np.random.SeedSequence(seed)
    generators = [np.random.default_rng(i) for i in seq.spawn(threads)]
    xyz = np.zeros((n, 3))

    executor = concurrent.futures.ThreadPoolExecutor(threads)
    step = np.ceil(n / threads).astype(np.int_)

    def _fill(random_state, out, first, last):
        zero_rows = np.where(np.all(np.isclose(out[first:last], 0.0), axis=1))[
            0
        ]
        num_rows = zero_rows.size
        while num_rows > 0:
            random_state.standard_normal(
                size=(num_rows, 3), out=out[first:last]
            )
            zero_rows = np.where(
                np.all(np.isclose(out[first:last], 0.0), axis=1)
            )[0]
            num_rows = zero_rows.size
        out[first:last] /= np.sqrt((out[first:last] ** 2).sum(axis=1)).reshape(
            (-1, 1)
        )

    futures = {}
    for i in range(threads):
        args = (_fill, generators[i], xyz, i * step, (i + 1) * step)
        futures[executor.submit(*args)] = i
    concurrent.futures.wait(futures)

    executor.shutdown(False)
    return xyz


def _points_in_polygons(points, polygons):
    polygons_sorted = sorted(polygons, key=lambda x: x.area, reverse=True)
    out = np.zeros(points.shape[0], dtype=bool)
    for i in range(points.shape[0]):
        p = Point(points[i, 0], points[i, 1])
        for polygon in polygons_sorted:
            if polygon.contains(p):
                out[i] = True
                break
        else:
            out[i] = False
    return out
