"""This module contains a simple wrapper around the subduction convergence
workflow from PlateTectonicTools, allowing for parallelisation.
"""
import math
import os
import tempfile
import warnings
from sys import stderr
from typing import (
    Optional,
    Sequence,
    Union,
)

import numpy as np
import pandas as pd
from gplately import (
    PlateReconstruction,
    EARTH_RADIUS,
)
from ptt import subduction_convergence

from .misc import _PathLike

INCREMENT = 1


def run_calculate_convergence(
    nprocs: int,
    min_time: float,
    max_time: float,
    topology_filenames: Optional[Sequence[str]] = None,
    rotation_filenames: Optional[Union[Sequence[str], str]] = None,
    output_dir: _PathLike = ".",
    verbose: bool = False,
    plate_reconstruction: Optional[PlateReconstruction] = None,
):
    """Wrapper to call `calculate_convergence` in parallel.

    Parameters
    ----------
    nprocs : int
        Number of processes to use.
    min_time, max_time : float
        Minimum and maximum time steps to extract convergence data.
    topology_filenames : sequence of str
        Files containing topological features. Ignored if `plate_reconstruction`
        is not `None`.
    rotation_filenames : sequence of str
        Files containing rotation features. Ignored if `plate_reconstruction`
        is not `None`.
    output_dir : str
        Output directory to save convergence data .csv files.
    verbose : bool, default: False
        Print log to stderr.
    plate_reconstruction : PlateReconstruction, optional
        Obtain topologies and rotations from a `PlateReconstruction` object
        instead of filenames.
    """
    if not os.path.exists(output_dir):
        if verbose:
            print(
                "Output directory does not exist; creating now: "
                + str(output_dir),
                file=stderr,
            )
        os.makedirs(output_dir, exist_ok=True)

    if plate_reconstruction is None:
        if topology_filenames is None or rotation_filenames is None:
            raise TypeError(
                "Either `topology_filenames` and `rotation_filenames` "
                "or `plate_reconstruction` must be specified."
            )
        plate_reconstruction = PlateReconstruction(
            rotation_model=rotation_filenames,
            topology_features=topology_filenames,
        )

    times = np.arange(min_time, max_time + INCREMENT, INCREMENT)
    if nprocs == 1:
        data = [
            _parallel_func(
                plate_reconstruction=plate_reconstruction,
                time=t,
                ignore_warnings=True,
            )
            for t in times
        ]
    else:
        from joblib import Parallel, delayed

        with Parallel(nprocs, verbose=10 if verbose else 0) as parallel:
            data = parallel(
                delayed(_parallel_func)(
                    plate_reconstruction=plate_reconstruction,
                    time=t,
                    ignore_warnings=True,
                )
                for t in times
            )
    data = pd.concat(data)
    for col in (
        "distance_to_trench_edge (degrees)",
        "distance_from_trench_start (degrees)",
    ):
        if col not in data.columns:
            continue
        x_km = np.deg2rad(data[col]) * EARTH_RADIUS
        data[col.replace("(degrees)", "(km)")] = x_km
        data = data.drop(columns=col, errors="ignore")
    return data


def calculate_convergence(
    min_time: float,
    max_time: float,
    topology_filenames: Sequence[str],
    rotation_filenames: Union[Sequence[str], str],
    output_dir: _PathLike,
):
    """Wrapper around `subduction_convergence_over_time` from
    PlateTectonicTools that also adds column names to the output files.

    Parameters
    ----------
    min_time, max_time : float
        Minimum and maximum time steps to extract convergence data.
    topology_filenames : sequence of str
        Files containing topological features.
    rotation_filenames : sequence of str
        Files containing rotation features.
    output_dir : str
        Output directory to save convergence data .csv files.
    """
    output_prefix = os.path.join(output_dir, "convergence")
    output_extension = "csv"

    sampling_distance = math.radians(0.1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        subduction_convergence.subduction_convergence_over_time(
            output_filename_prefix=output_prefix,
            output_filename_extension=output_extension,
            rotation_filenames=rotation_filenames,
            topology_filenames=topology_filenames,
            threshold_sampling_distance_radians=sampling_distance,
            time_young=float(min_time),
            time_old=float(max_time),
            time_increment=float(INCREMENT),
            anchor_plate_id=0,
            output_distance_to_nearest_edge_of_trench=True,
            output_distance_to_start_edge_of_trench=True,
            output_convergence_velocity_components=True,
            output_trench_absolute_velocity_components=True,
            output_subducting_absolute_velocity=True,
            output_subducting_absolute_velocity_components=True,
        )

    output_filenames = [
        output_prefix + f"_{time:0.2f}." + output_extension
        for time in range(int(min_time), int(max_time) + INCREMENT, INCREMENT)
    ]

    column_names = (
        "lon",
        "lat",
        "convergence_rate (cm/yr)",
        "convergence_obliquity (degrees)",
        "trench_velocity (cm/yr)",
        "trench_velocity_obliquity (degrees)",
        "arc_segment_length (degrees)",
        "trench_normal_angle (degrees)",
        "subducting_plate_ID",
        "trench_plate_ID",
        "distance_to_trench_edge (degrees)",
        "distance_from_trench_start (degrees)",
        "convergence_rate_orthogonal (cm/yr)",
        "convergence_rate_parallel (cm/yr)",
        "trench_velocity_orthogonal (cm/yr)",
        "trench_velocity_parallel (cm/yr)",
        "subducting_plate_absolute_velocity (cm/yr)",
        "subducting_plate_absolute_obliquity (degrees)",
        "subducting_plate_absolute_velocity_orthogonal (cm/yr)",
        "subducting_plate_absolute_velocity_parallel (cm/yr)",
    )

    for output_filename in output_filenames:
        df = pd.read_csv(
            output_filename,
            sep=r"\s+",
            header=None,
            names=column_names,
            index_col=False,
        )
        df.to_csv(output_filename, index=False)


def _parallel_func(
    plate_reconstruction: PlateReconstruction,
    time: float,
    tessellation_threshold_radians: float = 0.001,
    ignore_warnings: bool = True,
) -> pd.DataFrame:
    data = plate_reconstruction.tessellate_subduction_zones(
        time=time,
        tessellation_threshold_radians=tessellation_threshold_radians,
        ignore_warnings=ignore_warnings,
        output_distance_to_nearest_edge_of_trench=True,
        output_distance_to_start_edge_of_trench=True,
        output_convergence_velocity_components=True,
        output_trench_absolute_velocity_components=True,
        output_subducting_absolute_velocity=True,
        output_subducting_absolute_velocity_components=True,
    )
    column_names = (
        "lon",
        "lat",
        "convergence_rate (cm/yr)",
        "convergence_obliquity (degrees)",
        "trench_velocity (cm/yr)",
        "trench_velocity_obliquity (degrees)",
        "arc_segment_length (degrees)",
        "trench_normal_angle (degrees)",
        "subducting_plate_ID",
        "trench_plate_ID",
        "distance_to_trench_edge (degrees)",
        "distance_from_trench_start (degrees)",
        "convergence_rate_orthogonal (cm/yr)",
        "convergence_rate_parallel (cm/yr)",
        "trench_velocity_orthogonal (cm/yr)",
        "trench_velocity_parallel (cm/yr)",
        "subducting_plate_absolute_velocity (cm/yr)",
        "subducting_plate_absolute_obliquity (degrees)",
        "subducting_plate_absolute_velocity_orthogonal (cm/yr)",
        "subducting_plate_absolute_velocity_parallel (cm/yr)",
    )
    out = pd.DataFrame(
        data,
        columns=column_names,
    )
    out["age (Ma)"] = np.float_(time)
    return out
