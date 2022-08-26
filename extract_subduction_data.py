import os
import yaml
from sys import stderr
from tempfile import TemporaryDirectory

import pandas as pd
from gplately.tools import plate_isotherm_depth

from lib.calculate_convergence import run_calculate_convergence
from lib.export_plate_model import run_export_plate_model
from lib.misc import (
    calculate_slab_dip,
    calculate_slab_flux,
    calculate_water_thickness,
)

DEFAULT_CONFIG_FILENAME = "subduction_config_Merdith.yml"


def main(
    min_time,
    max_time,
    model_dir,
    rotation_filenames,
    topology_filenames,
    coastline_filenames,
    model_export_dir,
    output_filename,
    agegrid_dir=None,
    sedthick_dir=None,
    carbonate_dir=None,
    nprocs=1,
    verbose=False,
):
    rotation_filenames = [
        os.path.join(model_dir, i) for i in rotation_filenames
    ]
    topology_filenames = [
        os.path.join(model_dir, i) for i in topology_filenames
    ]
    coastline_filenames = [
        os.path.join(model_dir, i) for i in coastline_filenames
    ]

    # Export plate model
    if not os.path.isdir(model_export_dir):
        if verbose:
            print(
                "Plate model export directory does not exist;",
                "creating now: " + model_export_dir,
                file=stderr,
            )
        os.makedirs(model_export_dir, exist_ok=True)
    run_export_plate_model(
        nprocs=nprocs,
        min_time=min_time,
        max_time=max_time,
        topology_filenames=topology_filenames,
        rotation_filenames=rotation_filenames,
        coastline_filenames=coastline_filenames,
        output_dir=model_export_dir,
        verbose=verbose,
    )

    with TemporaryDirectory() as convergence_dir, \
         TemporaryDirectory() as plate_maps_dir, \
         TemporaryDirectory() as subducted_quantities_dir:
        # Extract convergence data from plate model
        run_calculate_convergence(
            nprocs=nprocs,
            min_time=min_time,
            max_time=max_time,
            topology_filenames=topology_filenames,
            rotation_filenames=rotation_filenames,
            output_dir=convergence_dir,
            verbose=verbose,
        )

        if any(
            [
                agegrid_dir is not None,
                sedthick_dir is not None,
                carbonate_dir is not None,
            ]
        ):
            _raster_data(
                nprocs=nprocs,
                min_time=min_time,
                max_time=max_time,
                convergence_dir=convergence_dir,
                model_export_dir=model_export_dir,
                plate_maps_dir=plate_maps_dir,
                agegrid_dir=agegrid_dir,
                sedthick_dir=sedthick_dir,
                carbonate_dir=carbonate_dir,
                subducted_quantities_dir=subducted_quantities_dir,
                output_filename=output_filename,
                verbose=verbose,
            )
        else:
            subduction_data = []
            for time in range(min_time, max_time + 1):
                filename = os.path.join(
                    convergence_dir,
                    "convergence_{:.2f}.csv".format(float(time)),
                )
                df = pd.read_csv(filename)
                df["age (Ma)"] = time
                subduction_data.append(df)
            subduction_data = pd.concat(subduction_data)
            subduction_data.to_csv(output_filename, index=False)

    if verbose:
        print(
            "Subduction data written to output file: " + output_filename,
            file=stderr,
        )


def _raster_data(
    nprocs,
    min_time,
    max_time,
    convergence_dir,
    model_export_dir,
    plate_maps_dir,
    agegrid_dir,
    sedthick_dir,
    carbonate_dir,
    subducted_quantities_dir,
    output_filename,
    verbose=False,
):
    from lib.calculate_subducted_quantities import (
        run_calculate_subducted_quantities,
    )
    from lib.coregister_ocean_rasters import run_coregister_ocean_rasters
    from lib.create_plate_maps import run_create_plate_map

    # Rasterise plate topologies (used later)
    run_create_plate_map(
        nprocs=nprocs,
        min_time=min_time,
        max_time=max_time,
        input_dir=model_export_dir,
        output_dir=plate_maps_dir,
        resolution=0.5,
        verbose=verbose,
    )

    subduction_data = run_coregister_ocean_rasters(
        nprocs=nprocs,
        min_time=min_time,
        max_time=max_time,
        input_data=convergence_dir,
        plates_dir=plate_maps_dir,
        agegrid_dir=agegrid_dir,
        sedthick_dir=sedthick_dir,
        carbonate_dir=carbonate_dir,
        output_dir=None,
        combined_filename=None,
        verbose=verbose,
    )

    subduction_data["plate_thickness (m)"] = plate_isotherm_depth(
        subduction_data["seafloor_age (Ma)"],
        maxiter=100,
    )
    subduction_data["water_thickness (m)"] = calculate_water_thickness(
        subduction_data["sediment_thickness (m)"]
    )
    subduction_data = calculate_slab_dip(subduction_data)
    subduction_data = calculate_slab_flux(subduction_data)

    # Calculate cumulative subducted sediments, water, etc.
    subducted_quantities = (
        "sediment_thickness (m)",
        "carbonate_thickness (m)",
        "plate_thickness (m)",
        "water_thickness (m)",
    )
    run_calculate_subducted_quantities(
        subduction_data=subduction_data,
        quantities=subducted_quantities,
        output_dirs=subducted_quantities_dir,
        min_time=min_time,
        max_time=max_time,
        nprocs=nprocs,
        verbose=verbose,
    )
    subduction_data = run_coregister_ocean_rasters(
        nprocs=nprocs,
        min_time=min_time,
        max_time=max_time,
        input_data=subduction_data,
        plates_dir=plate_maps_dir,
        output_dir=None,
        combined_filename=output_filename,
        subducted_thickness_dir=os.path.join(
            subducted_quantities_dir,
            "plate_thickness",
        ),
        subducted_sediments_dir=os.path.join(
            subducted_quantities_dir,
            "sediment_thickness",
        ),
        subducted_carbonates_dir=os.path.join(
            subducted_quantities_dir,
            "carbonate_thickness",
        ),
        subducted_water_dir=os.path.join(
            subducted_quantities_dir,
            "water_thickness",
        ),
        verbose=verbose,
    )


def read_config_file(filename, flatten=False):
    with open(filename, "r") as f:
        try:
            out = yaml.load(f, yaml.CLoader)
        except Exception:
            out = yaml.load(f, yaml.Loader)
    if flatten:
        d = {}
        for i in out.keys():
            for j in out[i].keys():
                d[j] = out[i][j]
        out = d
    return out


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description=(
            "Extract various subduction-related data to tabular format."
        )
    )
    parser.add_argument(
        dest="config_file",
        metavar="CONFIG_FILE",
        default=DEFAULT_CONFIG_FILENAME,
        nargs="?",
        help="config. filename; default: `{}`".format(DEFAULT_CONFIG_FILENAME),
    )
    parsed_args = parser.parse_args()
    kwargs = read_config_file(parsed_args.config_file, flatten=True)
    main(**kwargs)
