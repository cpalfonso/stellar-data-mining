"""Miscellaneous useful functions."""
from sys import stderr

import numpy as np
import pandas as pd
import pygplates
from gplately import EARTH_RADIUS
from gplately.tools import plate_isotherm_depth
from ptt.utils.points_in_polygons import find_polygons

# Water thickness parameters
AVERAGE_OCEAN_FLOOR_SEDIMENT_SURFACE_POROSITY = 0.66
AVERAGE_OCEAN_FLOOR_SEDIMENT_POROSITY_DECAY = 1333

# Slab dip parameters
SLAB_DIP_COEFF_A = 0.00639289
SLAB_DIP_COEFF_B = 8.00315437
ARC_DEPTH = 125.0  # km
PLATE_THICKNESS_ISOTHERM = 1150.0  # degrees Celsius


def calculate_water_thickness(
    data,
    porosity_decay=AVERAGE_OCEAN_FLOOR_SEDIMENT_POROSITY_DECAY,
    surface_porosity=AVERAGE_OCEAN_FLOOR_SEDIMENT_SURFACE_POROSITY,
    inplace=False,
):
    """Calculate sediment pore water volume density.

    Parameters
    ----------
    data : str, pandas.DataFrame, or array_like
        Sediment thickness data. If `data` is a CSV filename or data frame,
        the 'sediment_thickness (m)' column will be used. Otherwise,
        `data` will be assumed to be an array of sediment thickness values.
    porosity_decay : float, default: 1333
        Porosity decay constant for pore space calculation.
    surface_porosity : float, default: 0.66
        Surface porosity value for pore space calculation.
    inplace : bool, default: False
        If `data` is a data frame, modify it in-place.

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        If `data` was an array_like of sediment thickness values, the output
        will be a numpy.ndarray of water thickness values. Otherwise, `data`
        is returned, with an additional column: 'water_thickness (m)'.
    """
    if isinstance(data, str):
        data = pd.read_csv(data)
    if isinstance(data, pd.DataFrame) and not inplace:
        data = data.copy()

    if isinstance(data, pd.DataFrame):
        sediment_thickness = np.array(data["sediment_thickness (m)"])
    else:
        sediment_thickness = np.array(data)

    water_thickness = (
        porosity_decay
        * surface_porosity
        * (1 - np.exp(-1.0 * sediment_thickness / porosity_decay))
    )

    if isinstance(data, pd.DataFrame):
        data["water_thickness (m)"] = water_thickness
        return data
    else:
        return water_thickness


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


def reconstruct_by_topologies(
    topological_features,
    rotation_model,
    points,
    start_time,
    end_time,
    time_step=1.0,
    verbose=False,
    intermediate_steps=False,
):
    """Simple and efficient reconstruction by topologies (does not account for
    collisions).

    Parameters
    ----------
    topological_features : FeatureCollection
        Topological features for plate reconstruction.
    rotation_model : RotationModel
        Rotation model for plate reconstruction.
    points : array_like
        Points to reconstruct. Must be an array_like of shape (n, 2),
        where n is the number of points; each row represents a lat-lon
        coordinate pair.
    start_time : float
        Start time of reconstruction (Ma).
    end_time : float
        End time of reconstruction (Ma).
    time_step : float, default: 1.0
        Time step (Myr).
    verbose : bool, default: False
        Print log to stderr.
    intermediate_steps : bool, default: False
        Return all intermediate timesteps.

    Returns
    -------
    result
        If `intermediate_steps == True`, `result` is a 2-tuple of
        `(times, points)`, where `times` is a numpy.ndarray of timesteps
        and `points` is a list of point coordinate arrays corresponding
        to the timesteps int `times`.
        If `intermediate_steps == False`, result is a single point coordinate
        array corresponding to the coordinates at `end_time`.

    Notes
    -----
    The output coordinate arrays are (n, 2) lat-lon ndarrays, where n is the
    number of input points.
    """
    topological_features = pygplates.FeaturesFunctionArgument(
        topological_features
    ).get_features()
    rotation_model = pygplates.RotationModel(rotation_model)

    if (
        (start_time > end_time and time_step > 0.0)
        or (start_time < end_time and time_step < 0.0)
    ):
        time_step = time_step * -1.0

    array_input = isinstance(points, np.ndarray)
    try:
        points = pygplates.PointOnSphere(points)
    except Exception:
        pass
    mp = pygplates.MultiPointOnSphere(points)
    points = list(mp.get_points())

    times = np.arange(start_time, end_time + time_step, time_step)
    if intermediate_steps:
        if array_input:
            step_p = np.vstack(
                [
                    np.reshape(i.to_lat_lon_array(), (1, -1))
                    for i in points
                ]
            )
        else:
            step_p = points
        step_points = [step_p]
    if verbose:
        print(
            "Reconstructing by topologies from {} Ma to {} Ma".format(
                start_time, end_time,
            ),
            file=stderr,
        )
    for time_index, new_time in enumerate(times):
        if time_index == 0:
            continue
        if verbose:
            print("Time now: {} Ma".format(new_time), file=stderr)
        previous_time = float(times[time_index - 1])
        new_time = float(new_time)
        new_points = _reconstruct_timestep(
            points,
            previous_time,
            new_time,
            topological_features,
            rotation_model,
        )
        points = new_points
        if intermediate_steps:
            if array_input:
                step_p = np.vstack(
                    [
                        np.reshape(i.to_lat_lon_array(), (1, -1))
                        for i in points
                    ]
                )
            else:
                step_p = points
            step_points.append(step_p)
            del step_p
        del new_points

    if array_input:
        points = np.vstack(
            [
                np.reshape(i.to_lat_lon_array(), (1, -1))
                for i in points
            ]
        )
    if intermediate_steps:
        return times, step_points
    return points


def _reconstruct_timestep(
    points,
    previous_time,
    new_time,
    topological_features,
    rotation_model,
):
    topologies = []
    pygplates.resolve_topologies(
        topological_features,
        rotation_model,
        topologies,
        previous_time,
    )
    polygon_plate_ids = [
        i.get_feature().get_reconstruction_plate_id() for i in topologies
    ]
    topological_polygons = [
        i.get_resolved_geometry() for i in topologies
    ]
    point_plate_ids = find_polygons(
        points,
        topological_polygons,
        polygon_plate_ids,
    )
    rotations_dict = {
        None: pygplates.FiniteRotation.create_identity_rotation()
    }
    for i in point_plate_ids:
        if i in rotations_dict or i is None:
            continue
        rot = rotation_model.get_rotation(
            to_time=new_time,
            from_time=previous_time,
            moving_plate_id=i,
            use_identity_for_missing_plate_ids=True,
        )
        rotations_dict[i] = rot
    new_points = []
    for point, plate_id in zip(points, point_plate_ids):
        rot = rotations_dict[plate_id]
        new_point = rot * point
        new_points.append(new_point)
    return new_points
