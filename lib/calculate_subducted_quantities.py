import os
from sys import stderr

from gplately import EARTH_RADIUS
import numpy as np
import pandas as pd
import xarray as xr

_DEFAULT_RESOLUTION = 0.5  # degrees


def run_calculate_subducted_quantities(
    subduction_data,
    quantities,
    output_dirs,
    resolution=_DEFAULT_RESOLUTION,
    min_time=None,
    max_time=None,
    nprocs=1,
    verbose=False,
    transforms=None,
):
    if transforms is None:
        transforms = {}
    if isinstance(quantities, str):
        quantities = [quantities]
    if isinstance(output_dirs, str) and len(quantities) > 1:
        output_dirs = [os.path.join(output_dirs, i.split()[0]) for i in quantities]
    if len(quantities) != len(output_dirs):
        raise ValueError(
            "Length mismatch: n. quantities = {}".format(len(quantities))
            + ", n. output dirs = {}".format(len(output_dirs))
        )

    try:
        subduction_data = pd.read_csv(subduction_data)
    except Exception as e:
        if isinstance(subduction_data, str):
            raise FileNotFoundError(
                "Could not read filename: " + subduction_data
            ) from e
        if not isinstance(subduction_data, pd.DataFrame):
            raise TypeError(
                "Invalid `subduction_data`: {}".format(subduction_data)
            ) from e
    
    for i in quantities:
        if i not in subduction_data.columns.values:
            message = (
                "Column name not found: {}".format(i)
                + "\n"
                + "Valid options: {}".format(subduction_data.columns.values)
            )
            raise ValueError(message)

    for i in transforms:
        t = transforms[i]
        if not callable(t):
            message = (
                "Invalid transform for quantity {}".format(i)
                + ": {}".format(t)
            )
            raise TypeError(message)

    for i in output_dirs:
        if not os.path.isdir(i):
            if verbose:
                print(
                    "Output directory does not exist; creating: " + i,
                    file=stderr,
                )
            os.makedirs(i, exist_ok=True)

    subduction_data = subduction_data.dropna(
        axis="index",
        subset=[
            "age (Ma)",
            "arc_segment_length (degrees)",
            "convergence_rate_orthogonal (cm/yr)",
            "lon",
            "lat",
            *quantities,
        ],
    )

    if min_time is not None and max_time is not None:
        times = range(max_time, min_time - 1, -1)
    else:
        if min_time is not None:
            subduction_data = subduction_data[
                subduction_data["age Ma)"] >= min_time
            ]
        if max_time is not None:
            subduction_data = subduction_data[
                subduction_data["age Ma)"] <= max_time
            ]
        times = sorted(subduction_data["age (Ma)"].unique(), reverse=True)

    if verbose:
        print("Extracting data...", file=stderr)

    if nprocs == 1:
        timestep_results = [
            extract_timestep(
            time,
            (subduction_data[subduction_data["age (Ma)"] == time]).copy(),
            quantities,
            resolution,
            transforms,
            )
            for time in times
        ]
    else:
        from joblib import Parallel, delayed

        p = Parallel(nprocs, verbose=10 * int(verbose))
        timestep_results = p(
            delayed(extract_timestep)(
                time,
                (subduction_data[subduction_data["age (Ma)"] == time]).copy(),
                quantities,
                resolution,
                transforms,
            )
            for time in times
        )

    densities = {}
    cumulative_densities = {}
    for quantity in quantities:
        densities[quantity] = np.dstack([i[quantity] for i in timestep_results])
        cumulative_densities[quantity] = np.cumsum(densities[quantity], axis=-1)

    if verbose:
        print("Writing output files...", file=stderr)
    for i, time in enumerate(times):
        for quantity, output_dir in zip(quantities, output_dirs):
            density = densities[quantity][..., i]
            cumulative_density = cumulative_densities[quantity][..., i]

            for which, data in zip(
                ("", "cumulative_"),
                (density, cumulative_density),
            ):
                output_filename = os.path.join(
                    output_dir,
                    "{}density_{}Ma.nc".format(which, time),
                )
                if verbose:
                    print(
                        "\t- writing output file: " + output_filename,
                        file=stderr,
                    )
                _write_file(data, output_filename, resolution=resolution)

    return cumulative_densities


def extract_timestep(
    time,
    subduction_data,
    quantities,
    resolution=_DEFAULT_RESOLUTION,
    transforms={},
):
    subduction_data = subduction_data[subduction_data["age (Ma)"] == time]

    # Distance in m, time in Myr
    segment_lengths = (
        np.deg2rad(np.array(subduction_data["arc_segment_length (degrees)"]))
        * EARTH_RADIUS
        * 1.0e3
    )
    subduction_rates = (
        np.array(subduction_data["convergence_rate_orthogonal (cm/yr)"])
        * 0.01
        * 1.0e6
    )
    subduction_rates = np.clip(subduction_rates, 0.0, np.inf)

    lons = np.array(subduction_data["lon"])
    lats = np.array(subduction_data["lat"])

    xedges = np.arange(-180.0, 180.0 + resolution, resolution)
    glons = (0.5 * (np.roll(xedges, 1) + xedges))[1:]
    yedges = np.arange(-90.0, 90.0 + resolution, resolution)
    glats = (0.5 * (np.roll(yedges, 1) + yedges))[1:]

    _, mlats = np.meshgrid(glons, glats)
    lon_lengths = _longitude_length(mlats, delta=resolution)
    lat_lengths = np.full_like(mlats, _latitude_length(delta=resolution))
    cell_areas = lon_lengths * lat_lengths

    out = {}
    for quantity in quantities:
        values = np.array(subduction_data[quantity])
        if quantity in transforms:
            values = transforms[quantity](values)
        weights = values * segment_lengths * subduction_rates
        density, _, _ = np.histogram2d(
            x=lons,
            y=lats,
            bins=(xedges, yedges),
            weights=weights,
        )
        density = density.T
        density = density / cell_areas

        out[quantity] = density
    return out


def _write_file(data, filename, resolution=_DEFAULT_RESOLUTION):
    xedges = np.arange(-180.0, 180.0 + resolution, resolution)
    lons = (0.5 * (np.roll(xedges, 1) + xedges))[1:]
    yedges = np.arange(-90.0, 90.0 + resolution, resolution)
    lats = (0.5 * (np.roll(yedges, 1) + yedges))[1:]

    dset = xr.Dataset(
        data_vars={
            "z": (("lat", "lon"), data),
        },
        coords={
            "lon": lons,
            "lat": lats,
        }
    )
    encoding = {"z": {"dtype": "float32", "zlib": True}}
    dset.to_netcdf(filename, encoding=encoding)
    return dset


def _longitude_length(latitude, delta=1.0, radius=EARTH_RADIUS * 1000.0, degrees=True):
    if degrees:
        latitude = np.deg2rad(latitude)
        length = np.deg2rad(1.0) * radius * np.cos(latitude)
    else:
        length = radius * np.cos(latitude)
    return delta * length


def _latitude_length(delta=1.0, radius=EARTH_RADIUS * 1000.0, degrees=True):
    if degrees:
        delta = np.deg2rad(delta)
    return radius * delta
