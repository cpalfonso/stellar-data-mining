"""Miscellaneous useful functions."""
import os
import re
import warnings
from sys import stderr
from typing import (
    Iterable,
    Union,
)

import numpy as np
import pandas as pd
import pygplates
from gplately import EARTH_RADIUS
from gplately.reconstruction import reconstruct_points
from gplately.tools import plate_isotherm_depth
from pandas.errors import PerformanceWarning
from ptt.utils.points_in_polygons import find_polygons

# For backwards compatibility
from .water import (
    calculate_water_thickness,
    DEFAULT_SURFACE_POROSITY as AVERAGE_OCEAN_FLOOR_SEDIMENT_SURFACE_POROSITY,
    DEFAULT_POROSITY_DECAY as AVERAGE_OCEAN_FLOOR_SEDIMENT_POROSITY_DECAY,
)

__all__ = [
    "SLAB_DIP_COEFF_A",
    "SLAB_DIP_COEFF_B",
    "ARC_DEPTH",
    "PLATE_THICKNESS_ISOTHERM",
    "calculate_slab_flux",
    "reconstruct_by_topologies",
    "format_feature_name",
    "load_data",
    "filter_topological_features",
]

# Slab dip parameters
SLAB_DIP_COEFF_A = 0.00639289
SLAB_DIP_COEFF_B = 8.00315437
ARC_DEPTH = 125.0  # km
PLATE_THICKNESS_ISOTHERM = 1150.0  # degrees Celsius

# Type hints
_PathLike = Union[os.PathLike, str]
_PathOrDataFrame = Union[_PathLike, pd.DataFrame]
_FeatureCollectionInput = Union[
    pygplates.FeatureCollection,
    str,
    pygplates.Feature,
    Iterable[pygplates.Feature],
    Iterable[
        Union[
            pygplates.FeatureCollection,
            str,
            pygplates.Feature,
            Iterable[pygplates.Feature],
        ]
    ],
]
_RotationModelInput = Union[
    pygplates.RotationModel,
    _FeatureCollectionInput,
]


def calculate_slab_flux(df, inplace=False):
    """Calculate slab flux from convergence rate and plate thickness.

    Parameters
    ----------
    df : str or pandas.DataFrame
        Input data frame or CSV file, containing the following columns:
        - 'convergence_rate_orthogonal (cm/yr)', and
        - 'plate_thickness (m)' or
        - 'seafloor_age (Ma)'

    inplace : bool, default: False
        If `data` is a data frame, modify it in-place.

    Returns
    -------
    pandas.DataFrame
        Input data frame, copied if `inplace == False`, with additional
        column: 'slab_flux (m^2/yr)'.
    """
    if isinstance(df, str):
        df = pd.read_csv(df)
    if not inplace:
        df = df.copy()

    rates = np.array(df["convergence_rate_orthogonal (cm/yr)"]) * 0.01
    if "plate_thickness (m)" in df.columns.values:
        thicknesses = np.array(df["plate_thickness (m)"])
    else:
        ages = np.array(df["seafloor_age (Ma)"])
        thicknesses = plate_isotherm_depth(ages, maxiter=100)
    slab_flux = rates * thicknesses
    df["slab_flux (m^2/yr)"] = slab_flux
    return df


# def reconstruct_by_topologies(
#     topological_features,
#     rotation_model,
#     points,
#     start_time,
#     end_time,
#     time_step=1.0,
#     verbose=False,
#     intermediate_steps=False,
# ):
#     """Simple and efficient reconstruction by topologies (does not account for
#     collisions).

#     Parameters
#     ----------
#     topological_features : FeatureCollection
#         Topological features for plate reconstruction.
#     rotation_model : RotationModel
#         Rotation model for plate reconstruction.
#     points : array_like
#         Points to reconstruct. Must be an array_like of shape (n, 2),
#         where n is the number of points; each row represents a lat-lon
#         coordinate pair.
#     start_time : float
#         Start time of reconstruction (Ma).
#     end_time : float
#         End time of reconstruction (Ma).
#     time_step : float, default: 1.0
#         Time step (Myr).
#     verbose : bool, default: False
#         Print log to stderr.
#     intermediate_steps : bool, default: False
#         Return all intermediate timesteps.

#     Returns
#     -------
#     result
#         If `intermediate_steps == True`, `result` is a 2-tuple of
#         `(times, points)`, where `times` is a numpy.ndarray of timesteps
#         and `points` is a list of point coordinate arrays corresponding
#         to the timesteps int `times`.
#         If `intermediate_steps == False`, result is a single point coordinate
#         array corresponding to the coordinates at `end_time`.

#     Notes
#     -----
#     The output coordinate arrays are (n, 2) lat-lon ndarrays, where n is the
#     number of input points.
#     """
#     topological_features = pygplates.FeaturesFunctionArgument(
#         topological_features
#     ).get_features()
#     rotation_model = pygplates.RotationModel(rotation_model)

#     if (
#         (start_time > end_time and time_step > 0.0)
#         or (start_time < end_time and time_step < 0.0)
#     ):
#         time_step = time_step * -1.0

#     array_input = isinstance(points, np.ndarray)
#     try:
#         points = pygplates.PointOnSphere(points)
#     except Exception:
#         pass
#     mp = pygplates.MultiPointOnSphere(points)
#     points = list(mp.get_points())

#     times = np.arange(start_time, end_time + time_step, time_step)
#     if intermediate_steps:
#         if array_input:
#             step_p = np.vstack(
#                 [
#                     np.reshape(i.to_lat_lon_array(), (1, -1))
#                     for i in points
#                 ]
#             )
#         else:
#             step_p = points
#         step_points = [step_p]
#     if verbose:
#         print(
#             "Reconstructing by topologies from {} Ma to {} Ma".format(
#                 start_time, end_time,
#             ),
#             file=stderr,
#         )
#     for time_index, new_time in enumerate(times):
#         if time_index == 0:
#             continue
#         if verbose:
#             print("Time now: {} Ma".format(new_time), file=stderr)
#         previous_time = float(times[time_index - 1])
#         new_time = float(new_time)
#         new_points = _reconstruct_timestep(
#             points,
#             previous_time,
#             new_time,
#             topological_features,
#             rotation_model,
#         )
#         points = new_points
#         if intermediate_steps:
#             if array_input:
#                 step_p = np.vstack(
#                     [
#                         np.reshape(i.to_lat_lon_array(), (1, -1))
#                         for i in points
#                     ]
#                 )
#             else:
#                 step_p = points
#             step_points.append(step_p)
#             del step_p
#         del new_points

#     if array_input:
#         points = np.vstack(
#             [
#                 np.reshape(i.to_lat_lon_array(), (1, -1))
#                 for i in points
#             ]
#         )
#     if intermediate_steps:
#         return times, step_points
#     return points


# def _reconstruct_timestep(
#     points,
#     previous_time,
#     new_time,
#     topological_features,
#     rotation_model,
# ):
#     topologies = []
#     pygplates.resolve_topologies(
#         topological_features,
#         rotation_model,
#         topologies,
#         previous_time,
#         resolve_topology_types=pygplates.ResolveTopologyType.boundary | pygplates.ResolveTopologyType.line,
#     )
#     topologies = [
#         i for i in topologies
#         if isinstance(i.get_resolved_geometry(), pygplates.PolygonOnSphere)
#     ]
#     polygon_plate_ids = [
#         i.get_feature().get_reconstruction_plate_id() for i in topologies
#     ]
#     topological_polygons = [
#         i.get_resolved_geometry() for i in topologies
#     ]
#     point_plate_ids = find_polygons(
#         points,
#         topological_polygons,
#         polygon_plate_ids,
#     )
#     rotations_dict = {
#         None: pygplates.FiniteRotation.create_identity_rotation()
#     }
#     for i in point_plate_ids:
#         if i in rotations_dict or i is None:
#             continue
#         rot = rotation_model.get_rotation(
#             to_time=new_time,
#             from_time=previous_time,
#             moving_plate_id=i,
#             use_identity_for_missing_plate_ids=True,
#         )
#         rotations_dict[i] = rot
#     new_points = []
#     for point, plate_id in zip(points, point_plate_ids):
#         rot = rotations_dict[plate_id]
#         new_point = rot * point
#         new_points.append(new_point)
#     return new_points


def reconstruct_by_topologies(
    data,
    plate_reconstruction=None,
    rotation_model=None,
    topological_features=None,
    times=None,
    verbose=False,
):
    """Simple and relatively efficient reconstruction by topologies (does not
    account for collisions).

    Parameters
    ----------
    data : DataFrame or str
        Input data frame or filename. Must contain the following columns:
        `age (Ma)`, `lon`, `lat`.
    plate_reconstruction : PlateReconstruction, optional
        Plate motion and topology model to use for reconstruction. If
        `plate_reconstruction` is `None`, both `topological_features`
        and `rotation_model` must not be `None`.
    rotation_model : RotationModel or sequence of Feature or str, optional
        Rotation model for plate reconstruction.
    topological_features : sequence of Feature or str, optional
        Topological features for plate reconstruction.
    times : sequence of float, optional
        Times at which to perform reconstruction. If `None`, will be determined
        from `data`.
    verbose : bool, default: False
        Print log to stderr.

    Returns
    -------
    DataFrame
        A copy of `data` with additional columns `lon_{t}` and `lat_{t}` for each
        reconstruction time `t`.
    """
    if plate_reconstruction is not None:
        rotation_model = plate_reconstruction.rotation_model
        topological_features = plate_reconstruction.topology_features
    if rotation_model is None:
        raise TypeError("Rotation model must be provided")
    if topological_features is None:
        raise TypeError("Topological features must be provided")

    data = load_data(data, copy=False)
    if times is None:
        times = np.arange(data["age (Ma)"].round().max() + 1)
    times = np.sort(times)
    recon_cols = [*[f"lon_{t}" for t in times], *[f"lat_{t}" for t in times]]

    df_recon = pd.DataFrame(columns=recon_cols, index=data.index, dtype=np.float_)
    data = data.join(df_recon)

    if not isinstance(rotation_model, pygplates.RotationModel):
        rotation_model = pygplates.RotationModel(rotation_model)
    topological_features = pygplates.FeaturesFunctionArgument(
        topological_features
    ).get_features()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PerformanceWarning)
        for t, subset in data.groupby("age (Ma)"):
            for i in subset.index:
                data.at[i, f"lon_{t:0.0f}"] = data.at[i, "lon"]
                data.at[i, f"lat_{t:0.0f}"] = data.at[i, "lat"]

        for t in times[::-1]:
            if verbose and t % 10 == 0:
                print(f"Reconstructing to {t:0.0f} Ma", file=stderr)
            if t == min(times):
                break
            old_lon_col = f"lon_{t:0.0f}"
            old_lat_col = f"lat_{t:0.0f}"
            new_lon_col = f"lon_{t - 1:0.0f}"
            new_lat_col = f"lat_{t - 1:0.0f}"

            subset = data[~data[old_lon_col].isna()]
            lons = subset[old_lon_col]
            lats = subset[old_lat_col]
            try:
                points = pygplates.MultiPointOnSphere(np.column_stack((lats, lons)))
            except pygplates.InsufficientPointsForMultiPointConstructionError as err:
                raise RuntimeError(
                    f"Reconstruction failed at time {t}"
                ) from err
            reconstructed_points = reconstruct_points(
                rotation_model,
                topological_features,
                reconstruction_begin_time=t,
                reconstruction_end_time=t - 1,
                reconstruction_time_interval=1.0,
                points=points,
                detect_collisions=None,
            )
            new_lats, new_lons = zip(*[i.to_lat_lon() for i in reconstructed_points])
            for i, new_lon, new_lat in zip(subset.index, new_lons, new_lats):
                data.at[i, new_lon_col] = new_lon
                data.at[i, new_lat_col] = new_lat
    return data


def format_feature_name(s, bold=False):
    """Make feature names easier to read in plots."""
    s = s.replace("_", " ")
    s = s[0].capitalize() + s[1:]

    replace = {
        "(cm/yr)": r"($\mathrm{cm \; {yr}^{-1}}$)",
        "(m)": r"($\mathrm{m}$)",
        "(m^3/m^2)": r"($\mathrm{m^3 \; m^{-2}}$)",
        "(m^2/yr)": r"($\mathrm{m^2 \; {yr}^{-1}}$)",
        "(Ma)": r"($\mathrm{Ma}$)",
        "(degrees)": r"($\mathrm{\degree}$)",
        "(km)": r"($\mathrm{km}$)",
        "(km/Myr)": r"($\mathrm{km \; {Myr}^{-1}}$)",
    }
    if bold:
        replace = {
            key: value.replace(r"\mathrm", r"\mathbf")
            for key, value in replace.items()
        }
    for key, value in replace.items():
        s = s.replace(key, value)

    return s


def load_data(
    data: _PathOrDataFrame,
    verbose: bool = False,
    copy: bool = True,
) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        if verbose:
            print(f"Loading data from file: {data}", file=stderr)
        data = pd.read_csv(data)
    elif copy:
        data = pd.DataFrame(data)
    return data


def filter_topological_features(
    filenames: Union[str, Iterable[str]]
) -> pygplates.FeatureCollection:
    topological_features = []
    for fc, filename in pygplates.FeaturesFunctionArgument(
        filenames
    ).get_files():
        if os.path.basename(filename).lower().startswith("inactive"):
            to_add = [
                feature for feature in fc
                if feature.get_feature_type().to_qualified_string()
                != "gpml:TopologicalNetwork"
            ]
        else:
            to_add = list(fc)
        to_add = [
            feature for feature in to_add
            if feature.get_feature_type().to_qualified_string()
            != "gpml:TopologicalSlabBoundary"
        ]
        topological_features.extend(to_add)
    return pygplates.FeatureCollection(topological_features)
