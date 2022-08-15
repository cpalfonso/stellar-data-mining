import numpy as np
import pandas as pd
from gplately import EARTH_RADIUS
from gplately.tools import plate_isotherm_depth

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
