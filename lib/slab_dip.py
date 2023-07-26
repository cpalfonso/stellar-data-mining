"""Wrapper around slabdip.predictor.SlabDipper."""
import warnings

import numpy as np
import pandas as pd
from gplately.tools import plate_isotherm_depth
from slabdip.predictor import (
    SlabDipper,
    default_variables as DEFAULT_VARIABLES,
    default_DataFrame as DEFAULT_TRAINING_DATA,
)

ARC_DEPTH = 125.0  # km

_COLUMNS_USED = {
    "seafloor_age (Ma)",
    "plate_thickness (m)",
    "convergence_rate_orthogonal (cm/yr)",
    "trench_velocity_orthogonal (cm/yr)",
    "subducting_plate_absolute_velocity (cm/yr)",
    "seafloor_spreading_rate (km/Myr)",
    "convergence_obliquity (degrees)",
    "convergence_rate (cm/yr)",
}


def calculate_slab_dip(
    data,
    training_data=None,
    model=None,
    predictors=DEFAULT_VARIABLES,
    response="slab_dip",
):
    """Calculate slab dip using SlabDipper.

    Parameters
    ----------
    data : str or pandas.DataFrame
        Subduction zone data to calculate slab dip.
    training_data : optional
    model : optional
    predictors : optional
    response : optional

    Returns
    -------
    pandas.DataFrame
        Copy of `data` with two additional columns: 'slab_dip (degrees)' and
        'arc_trench_distance (km)'.
    """
    if isinstance(data, str):
        data = pd.read_csv(data)
    else:
        data = pd.DataFrame(data)

    if training_data is None:
        training_data = DEFAULT_TRAINING_DATA
    X = training_data[predictors]
    y = training_data[response]

    dipper = SlabDipper(
        sklearn_regressor=model,
        X=X,
        y=y,
    )
    subset = _COLUMNS_USED.intersection(set(data.columns))
    data = data.dropna(subset=subset)

    # predictor_variables = np.array(extract_default_variables(data))

    predicted = dipper.predict(X=extract_default_variables(data))

    data["slab_dip (degrees)"] = predicted
    arc_distances = ARC_DEPTH / np.tan(np.deg2rad(predicted))
    data["arc_trench_distance (km)"] = arc_distances
    return data


def extract_default_variables(data):
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    ages = np.array(data["seafloor_age (Ma)"])

    if "plate_thickness (m)" in data.columns.values:
        thicknesses = np.array(data["plate_thickness (m)"])
    else:
        thicknesses = plate_isotherm_depth(ages, maxiter=10000)

    density = crust_density(thicknesses)
    vratio = (
        data["convergence_rate_orthogonal (cm/yr)"] * 1.0e-2
        + data["trench_velocity_orthogonal (cm/yr)"] * 1.0e-2
    ) / (
        data["convergence_rate_orthogonal (cm/yr)"] * 1.0e-2 + 1.0e-22
    )
    vratio[data["subducting_plate_absolute_velocity (cm/yr)"] < 0] *= -1
    vratio = np.clip(np.array(vratio), 0, 1)

    spreading_rate = np.array(data["seafloor_spreading_rate (km/Myr)"])
    spreading_rate *= 1.0e3 * 1.0e-6  # convert to m/yr

    out = pd.DataFrame(
        {
            "angle": data["convergence_obliquity (degrees)"],
            "total_vel": data["convergence_rate (cm/yr)"] * 1.0e-2,
            "vel": data["convergence_rate_orthogonal (cm/yr)"] * 1.0e-2,
            "trench_vel": data["trench_velocity_orthogonal (cm/yr)"] * 1.0e-2,
            "vratio": vratio,
            "slab_age": ages,
            "slab_thickness": thicknesses,
            "spreading_rate": spreading_rate,
            "density": density,
        }
    )
    return out


def crust_density(thickness, data=None):
    if data is not None:
        thickness = data[thickness]
    thickness = np.array(thickness)

    h_c = 7.0e3  # thickness of crust
    h_s = 43e3 # thickness of spinel field
    h_g = 55e3 # thickness of garnet field
    h_total = h_c + h_s + h_g

    rho_a = 3300  # density of asthenosphere
    rho_c = 2900 # density of crust
    rho_s = 3330 # density of spinel
    rho_g0, rho_g1 = 3370, 3340 # density of garnet (upper, lower)

    h_c = np.minimum(thickness, h_c)
    h_s = np.minimum(thickness - h_c, h_s)
    h_g = thickness - h_c - h_s

    rho_g = 0.5 * (rho_g0 + (h_g * (rho_g1 - rho_g0) / 55.0e3) + rho_g1)
    # rho_g = np.mean(
    #     [
    #         rho_g0,
    #         (h_g * (rho_g1 - rho_g0) / 55.0e3) + rho_g1,
    #     ]
    # )

    rho_plate = np.full_like(thickness, rho_a)  # plate thickness = 0
    np.divide(
        rho_c * h_c + rho_s * h_s + rho_g * h_g,
        h_total,
        out=rho_plate,
        where=h_total != 0.0,
    )

    # rho_plate = (
    #     rho_c * h_c
    #     + rho_s * h_s
    #     + rho_g * h_g
    # ) / (
    #     h_total
    # )

    # rho_plate[np.isnan(rho_plate)] = rho_a  # plate thickness = 0
    return rho_plate
