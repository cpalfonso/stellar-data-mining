import math
import os
import warnings

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from ptt import subduction_convergence

INCREMENT = 1


def run_calculate_convergence(
    nprocs,
    min_time,
    max_time,
    topology_filenames,
    rotation_filenames,
    output_dir,
    verbose=False,
):
    if nprocs == 1:
        calculate_convergence(
            min_time=min_time,
            max_time=max_time,
            topology_filenames=topology_filenames,
            rotation_filenames=rotation_filenames,
            output_dir=output_dir,
        )
    else:
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
    min_time,
    max_time,
    topology_filenames,
    rotation_filenames,
    output_dir,
):
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
        output_prefix + f"_{time}.00." + output_extension
        for time in range(min_time, max_time + INCREMENT, INCREMENT)
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
