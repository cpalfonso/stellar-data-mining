"""This module contains a simple wrapper around the subduction convergence
workflow from PlateTectonicTools, allowing for parallelisation.
"""
import math
import os
import warnings
from sys import stderr
from typing import Sequence, Union

import numpy as np
import pandas as pd
from ptt import subduction_convergence

from .misc import _PathLike

INCREMENT = 1


def run_calculate_convergence(
    nprocs: int,
    min_time: float,
    max_time: float,
    topology_filenames: Sequence[str],
    rotation_filenames: Union[Sequence[str], str],
    output_dir: _PathLike,
    verbose: bool = False,
):
    """Wrapper to call `calculate_convergence` in parallel.

    Parameters
    ----------
    nprocs : int
        Number of processes to use.
    min_time, max_time : float
        Minimum and maximum time steps to extract convergence data.
    topology_filenames : sequence of str
        Files containing topological features.
    rotation_filenames : sequence of str
        Files containing rotation features.
    output_dir : str
        Output directory to save convergence data .csv files.
    verbose : bool, default: False
        Print log to stderr.
    """
    if not os.path.exists(output_dir):
        if verbose:
            print(
                "Output directory does not exist; creating now: "
                + str(output_dir),
                file=stderr,
            )
        os.makedirs(output_dir, exist_ok=True)

    if nprocs == 1:
        calculate_convergence(
            min_time=min_time,
            max_time=max_time,
            topology_filenames=topology_filenames,
            rotation_filenames=rotation_filenames,
            output_dir=output_dir,
        )
    else:
        from joblib import Parallel, delayed

        times = np.array_split(
            np.arange(min_time, max_time + INCREMENT, INCREMENT),
            nprocs,
        )
        start_times = [i[0] for i in times]
        end_times = [i[-1] for i in times]

        v = 10 if verbose else 0
        p = Parallel(nprocs, verbose=v)
        p(
            delayed(calculate_convergence)(
                start_time,
                end_time,
                topology_filenames,
                rotation_filenames,
                output_dir,
            )
            for start_time, end_time in zip(start_times, end_times)
        )


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
