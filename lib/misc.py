from sys import stderr
from tempfile import NamedTemporaryFile
import warnings

import numpy as np
import pandas as pd
import pygplates
from gplately import EARTH_RADIUS
from gplately.tools import plate_isotherm_depth
from joblib import Parallel, delayed
from ptt.utils.points_in_polygons import find_polygons

# Water thickness parameters
AVERAGE_OCEAN_FLOOR_SEDIMENT_SURFACE_POROSITY = 0.66
AVERAGE_OCEAN_FLOOR_SEDIMENT_POROSITY_DECAY = 1333

# Slab dip parameters
SLAB_DIP_COEFF_A = 0.00639289
SLAB_DIP_COEFF_B = 8.00315437
ARC_DEPTH = 125.0  # km
PLATE_THICKNESS_ISOTHERM = 1150.0  # degrees Celsius


def log(*args, **kwargs):
    """Print function that defaults to stderr, not stdout"""
    file = kwargs.pop("file", stderr)
    print(*args, file=file, **kwargs)


def calculate_water_thickness(
    data,
    porosity_decay=AVERAGE_OCEAN_FLOOR_SEDIMENT_POROSITY_DECAY,
    surface_porosity=AVERAGE_OCEAN_FLOOR_SEDIMENT_SURFACE_POROSITY,
    inplace=False,
):
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


def calculate_slab_dip(df, inplace=False):
    if not inplace:
        df = df.copy()

    # Segment lengths in m
    segment_lengths = (
        np.deg2rad(np.array(df["arc_segment_length (degrees)"]))
        * EARTH_RADIUS
        * 1.0e3
    )
    # Rates in m/yr
    convergence_rates = (
        np.array(df["convergence_rate_orthogonal (cm/yr)"]) * 1.0e-2
    )
    trench_migration_rates = (
        np.array(df["trench_velocity_orthogonal (cm/yr)"]) * 1.0e-2
    )
    subducting_plate_velocities = (
        np.array(df["subducting_plate_absolute_velocity_orthogonal (cm/yr)"])
        * 1.0e-2
    )

    # Only consider positive convergence rates
    convergence_rates = np.clip(convergence_rates, 0.0, np.inf)

    seafloor_ages = np.array(df["seafloor_age (Ma)"])
    plate_thicknesses = plate_isotherm_depth(
        seafloor_ages, temp=PLATE_THICKNESS_ISOTHERM, n=100
    )

    subduction_volume_rates = (
        plate_thicknesses * segment_lengths * convergence_rates
    )
    subduction_volume_rates *= 1.0e-9  # convert m^3/yr to km^3/yr

    vratio = (convergence_rates + trench_migration_rates) / (
        np.clip(convergence_rates, 1.0e-22, np.inf)
    )
    vratio[subducting_plate_velocities < 0.0] *= -1.0
    vratio = np.clip(vratio, 0.0, 1.0)

    slab_dips = (
        SLAB_DIP_COEFF_A * (convergence_rates * vratio) * plate_thicknesses
        + SLAB_DIP_COEFF_B
    )
    arc_distances = ARC_DEPTH / np.tan(np.deg2rad(slab_dips))

    df["slab_dip (degrees)"] = slab_dips
    df["arc_trench_distance (km)"] = arc_distances

    return df


def calculate_slab_flux(df, inplace=False):
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

    # if isinstance(points, pygplates.PointOnSphere):
    #     points = [points]
    # if isinstance(points, np.ndarray):
    #     array_input = True
    #     mp = pygplates.MultiPointOnSphere(points)
    #     points = list(mp.get_points())
    # else:
    #     array_input = False

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


def extract_coordinate_history(
    lats,
    lons,
    ages,
    names,
    rotation_model,
    topological_features=None,
    plate_ids=None,
    data=None,
    reindex=False,
    nprocs=1,
    verbose=False,
):
    name_col = "name"
    if data is not None:
        if isinstance(lats, str):
            lats = data[lats]
        if isinstance(lons, str):
            lons = data[lons]
        if isinstance(plate_ids, str):
            plate_ids = data[plate_ids]
        if isinstance(ages, str):
            ages = data[ages]
        if isinstance(names, str):
            name_col = names
            names = data[names]

    lats = np.array(lats).flatten()
    lons = np.array(lons).flatten()
    ages = np.array(ages).flatten()
    names = np.array(names).flatten()
    ages_dtype = ages.dtype
    names_dtype = names.dtype

    if plate_ids is not None:
        plate_ids = np.array(plate_ids).flatten()
        data_out = _extract_coordinates_by_plate_id(
            lats=lats,
            lons=lons,
            ages=ages,
            names=names,
            rotation_model=rotation_model,
            plate_ids=plate_ids,
            name_col=name_col,
        )
    elif topological_features is None:
        raise TypeError(
            "One of either `plate_ids` or `topological_features`"
            " must be provided"
        )
    else:
        ages = np.around(ages)
        max_time = np.nanmax(ages)
        times = np.arange(max_time + 1.0)
        if isinstance(rotation_model, pygplates.RotationModel) or nprocs == 1:
            # Cannot serialise RotationModel types, cannot run in parallel
            if nprocs != 1:
                warnings.warn(
                    "Cannot use `RotationModel` types in parallel; falling "
                    "back on serial execution.",
                    category=RuntimeWarning,
                )
            results = [
                _extract_coordinates_by_topologies(
                    time,
                    lats[ages == time],
                    lons[ages == time],
                    names[ages == time],
                    topological_features,
                    rotation_model,
                    name_col,
                )
                for time in times
            ]
        else:
            p = Parallel(nprocs, verbose=10 * int(verbose))
            with NamedTemporaryFile(mode="w", suffix=".gpml") as features, \
                 NamedTemporaryFile(mode="w", suffix=".rot") as rotations:
                pygplates.FeatureCollection(
                    pygplates.FeaturesFunctionArgument(
                        topological_features
                    ).get_features()
                ).write(features.name)
                pygplates.FeatureCollection(
                    pygplates.FeaturesFunctionArgument(
                        rotation_model
                    ).get_features()
                ).write(rotations.name)

                results = p(
                    delayed(_extract_coordinates_by_topologies)(
                        time,
                        lats[ages == time],
                        lons[ages == time],
                        names[ages == time],
                        features.name,
                        rotations.name,
                        name_col,
                    )
                    for time in times
                )
        data_out = pd.concat([i for i in results if i is not None])

    data_out[name_col] = data_out[name_col].astype(names_dtype)
    data_out["time"] = data_out["time"].astype(ages_dtype)
    if reindex:
        data_out = data_out.set_index([name_col, "time"], inplace=False)
    return data_out


def _extract_coordinates_by_plate_id(
    lats,
    lons,
    ages,
    names,
    rotation_model,
    plate_ids,
    name_col="name",
):
    rotation_model = pygplates.RotationModel(rotation_model)

    point_features = []
    for name, lat, lon, plate_id, age in zip(
        names,
        lats,
        lons,
        plate_ids,
        ages,
    ):
        try:
            plate_id = int(plate_id)
        except Exception:
            continue
        feature = pygplates.Feature()
        point = pygplates.PointOnSphere(float(lat), float(lon))
        feature.set_geometry(point)
        feature.set_valid_time(float(age), 0.0)
        feature.set_reconstruction_plate_id(plate_id)
        feature.set_name(str(name))
        point_features.append(feature)

    max_time = np.nanmax(ages)
    data_out = {
        name_col: [],
        "time": [],
        "lon": [],
        "lat": [],
        "plate_id": [],
    }
    for time in np.arange(max_time + 1.0):
        valid_points = [
            i for i in point_features
            if i.is_valid_at_time(time)
        ]
        reconstructed = []
        pygplates.reconstruct(
            valid_points,
            rotation_model,
            reconstructed,
            float(time),
        )
        for i in reconstructed:
            feature = i.get_feature()
            name = feature.get_name()
            plate_id = int(feature.get_reconstruction_plate_id())
            geom = i.get_reconstructed_geometry()
            coords = geom.to_lat_lon_array()
            ncoords = coords.shape[0]
            lats = list(coords[:, 0])
            lons = list(coords[:, 1])
            data_out[name_col].extend([name] * ncoords)
            data_out["time"].extend([time] * ncoords)
            data_out["lon"].extend(lons)
            data_out["lat"].extend(lats)
            data_out["plate_id"].extend([plate_id] * ncoords)
    data_out = pd.DataFrame(data_out)
    return data_out


def _extract_coordinates_by_topologies(
    time,
    lats,
    lons,
    names,
    topological_features,
    rotation_model,
    name_col="name",
):
    if np.size(lats) == 0:
        return None

    lats = np.array(lats).reshape((-1, 1))
    lons = np.array(lons).reshape((-1, 1))
    names = np.array(names)
    points = np.hstack((lats, lons))

    reconstruction_times, reconstructed_points = reconstruct_by_topologies(
        topological_features,
        rotation_model,
        points,
        0.0,
        time,
        intermediate_steps=True,
    )
    data_out = {}
    for reconstruction_time, p in zip(reconstruction_times, reconstructed_points):
        if reconstruction_time not in data_out:
            data_out[reconstruction_time] = ([], [])
        data_out[reconstruction_time][0].append(names)
        data_out[reconstruction_time][1].append(p)
    for key in data_out.keys():
        data_out[key] = (
            np.concatenate(data_out[key][0]),
            np.vstack(data_out[key][1])
        )
    df_out = {
        name_col: [],
        "time": [],
        "lat": [],
        "lon": [],
    }
    for time in data_out.keys():
        names = np.ravel(data_out[time][0])
        coords = data_out[time][1]
        for name, lat, lon in zip(names, coords[:, 0], coords[:, 1]):
            df_out[name_col].append(name)
            df_out["time"].append(time)
            df_out["lat"].append(lat)
            df_out["lon"].append(lon)
    df_out = pd.DataFrame(df_out)
    return df_out


def rolling_average(x, window=1, mode="valid"):
    return np.convolve(x, np.ones(window), mode=mode) / window
