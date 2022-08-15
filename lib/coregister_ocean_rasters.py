import os

import numpy as np
import pandas as pd
from skimage.transform import resize
from sklearn.neighbors import NearestNeighbors
import xarray as xr

INCREMENT = 1


def run_coregister_ocean_rasters(
    nprocs,
    min_time,
    max_time,
    input_data,
    plates_dir,
    output_dir,
    combined_filename,
    agegrid_dir=None,
    sedthick_dir=None,
    carbonate_dir=None,
    subducted_thickness_dir=None,
    subducted_sediments_dir=None,
    subducted_carbonates_dir=None,
    subducted_water_dir=None,
    verbose=False,
):
    times = range(min_time, max_time + INCREMENT, INCREMENT)

    if isinstance(input_data, str):
        if os.path.isdir(input_data):
            input_data = [
                pd.read_csv(
                    os.path.join(
                        input_data,
                        "convergence_{:.2f}.csv".format(time),
                    )
                )
                for time in times
            ]
        else:
            input_data = pd.read_csv(input_data)
    if isinstance(input_data, pd.DataFrame):
        input_data = [
            (input_data[input_data["age (Ma)"] == time]).copy()
            for time in times
        ]

    if nprocs == 1:
        out = [
            coregister_ocean_rasters(
                time,
                input_data_t,
                plates_dir,
                agegrid_dir,
                sedthick_dir,
                carbonate_dir,
                output_dir,
                subducted_thickness_dir,
                subducted_sediments_dir,
                subducted_carbonates_dir,
                subducted_water_dir,
            )
            for time, input_data_t in zip(times, input_data)
        ]
    else:
        from joblib import Parallel, delayed

        p = Parallel(nprocs, verbose=10 * int(verbose))
        out = p(
            delayed(coregister_ocean_rasters)(
                time,
                input_data_t,
                plates_dir,
                agegrid_dir,
                sedthick_dir,
                carbonate_dir,
                output_dir,
                subducted_thickness_dir,
                subducted_sediments_dir,
                subducted_carbonates_dir,
                subducted_water_dir,
            )
            for time, input_data_t in zip(times, input_data)
        )

    out = pd.concat(out, ignore_index=True)
    if combined_filename is not None:
        out.to_csv(combined_filename, index=False)
    return out


def coregister_ocean_rasters(
    time,
    df,
    plates_dir,
    agegrid_dir,
    sedthick_dir,
    carbonate_dir,
    output_dir,
    subducted_thickness_dir=None,
    subducted_sediments_dir=None,
    subducted_carbonates_dir=None,
    subducted_water_dir=None,
):
    plates_filename = os.path.join(plates_dir, "plate_ids_{}Ma.nc".format(time))

    if agegrid_dir is None:
        agegrid_filename = None
    else:
        agegrid_filename = os.path.join(
            agegrid_dir, "seafloor_age_mask_{}.0Ma.nc".format(time)
        )
        if not os.path.isfile(agegrid_filename):
            raise FileNotFoundError(
                "Age grid file not found: " + agegrid_filename
            )

    if sedthick_dir is None:
        sedthick_filename = None
    else:
        sedthick_filename = _get_sedthick_filename(
            sedthick_dir=sedthick_dir, time=time
        )
        if not os.path.isfile(sedthick_filename):
            raise FileNotFoundError(
                "Sediment thickness file not found: " + sedthick_filename
            )

    if carbonate_dir is None:
        carbonate_filename = None
    else:
        carbonate_filename = os.path.join(
            carbonate_dir, "uncompacted_carbonate_thickness_{}Ma.nc".format(time)
        )
        if not os.path.isfile(carbonate_filename):
            raise FileNotFoundError(
                "Carbonate thickness file not found: " + carbonate_filename
            )

    if subducted_thickness_dir is None:
        subducted_thickness_filename = None
    else:
        subducted_thickness_filename = os.path.join(
            subducted_thickness_dir,
            "cumulative_density_{}Ma.nc".format(time),
        )
        if not os.path.isfile(subducted_thickness_filename):
            raise FileNotFoundError(
                "Subducted plate volume file not found: "
                + subducted_thickness_filename
            )

    if subducted_sediments_dir is None:
        subducted_sediments_filename = None
    else:
        subducted_sediments_filename = os.path.join(
            subducted_sediments_dir,
            "cumulative_density_{}Ma.nc".format(time),
        )
        if not os.path.isfile(subducted_sediments_filename):
            raise FileNotFoundError(
                "Subducted sediments file not found: "
                + subducted_sediments_filename
            )

    if subducted_carbonates_dir is not None:
        subducted_carbonates_filename = os.path.join(
            subducted_carbonates_dir,
            "cumulative_density_{}Ma.nc".format(time),
        )
        if not os.path.isfile(subducted_carbonates_filename):
            raise FileNotFoundError(
                "Subducted carbonate sediments file not found: "
                + subducted_carbonates_filename
            )
    else:
        subducted_carbonates_filename = None
    if subducted_water_dir is not None:
        subducted_water_filename = os.path.join(
            subducted_water_dir,
            "cumulative_density_{}Ma.nc".format(time),
        )
        if not os.path.isfile(subducted_water_filename):
            raise FileNotFoundError(
                "Subducted water file not found: " + subducted_water_filename
            )
    else:
        subducted_water_filename = None

    df["seafloor_age (Ma)"] = np.nan
    df["age (Ma)"] = time
    with xr.open_dataset(plates_filename) as dset:
        plates = np.array(dset["plate_id"])
    plates[np.isnan(plates)] = -1
    plates = plates.astype(np.int_)

    raster_data = {}
    for filename, name in zip(
        (
            agegrid_filename,
            sedthick_filename,
            carbonate_filename,
            subducted_thickness_filename,
            subducted_sediments_filename,
            subducted_carbonates_filename,
            subducted_water_filename,
        ),
        (
            "agegrid",
            "sedthick",
            "carbonate",
            "subducted_thickness",
            "subducted_sediments",
            "subducted_carbonates",
            "subducted_water",
        ),
    ):
        if filename is None:
            continue
        raster_data[name] = {}
        with xr.open_dataset(filename) as dset:
            raster = np.array(dset["z"])
            lon = np.array(dset["lon"])
            lat = np.array(dset["lat"])

        if raster.shape != plates.shape:
            raster = resize(raster, plates.shape, order=1, mode="wrap")
            lon = np.linspace(lon.min(), lon.max(), raster.shape[1])
            lat = np.linspace(lat.min(), lat.max(), raster.shape[0])

        raster_data[name]["data"] = raster
        raster_data[name]["lon"] = lon
        raster_data[name]["lat"] = lat

    column_names = {
        "agegrid": "seafloor_age (Ma)",
        "sedthick": "sediment_thickness (m)",
        "carbonate": "carbonate_thickness (m)",
        "subducted_thickness": "subducted_plate_volume (m)",
        "subducted_sediments": "subducted_sediment_volume (m)",
        "subducted_carbonates": "subducted_carbonates_volume (m)",
        "subducted_water": "subducted_water_volume (m)",
    }
    for plate_id in df["subducting_plate_ID"].unique():
        df_plate = df[df["subducting_plate_ID"] == plate_id]
        lon_points = np.array(df_plate["lon"]).reshape((-1, 1))
        lat_points = np.array(df_plate["lat"]).reshape((-1, 1))
        coords_points = np.deg2rad(np.hstack((lat_points, lon_points)))

        for name in raster_data:
            raster = raster_data[name]["data"]
            column_name = column_names[name]
            plate_mask = np.logical_and(plates == plate_id, ~np.isnan(raster))
            if plate_mask.sum() == 0:
                continue
            raster_plate = raster[plate_mask].flatten()
            lon_data, lat_data = np.meshgrid(
                raster_data[name]["lon"],
                raster_data[name]["lat"],
            )
            lon_data = lon_data[plate_mask].flatten().reshape((-1, 1))
            lat_data = lat_data[plate_mask].flatten().reshape((-1, 1))
            coords_data = np.deg2rad(np.hstack((lat_data, lon_data)))
            neigh = NearestNeighbors(metric="haversine", n_jobs=1)
            neigh.fit(coords_data)

            distances, indices = neigh.kneighbors(
                coords_points, n_neighbors=1, return_distance=True
            )
            distances = np.rad2deg(distances)
            indices = indices.flatten()
            values = raster_plate[indices]
            df.loc[df["subducting_plate_ID"] == plate_id, column_name] = values

    if output_dir is not None:
        output_filename = os.path.join(
            output_dir, "subduction_data_{}Ma.csv".format(time)
        )
        df.to_csv(output_filename, index=False)

    return df


def _get_sedthick_filename(sedthick_dir, time):
    filenames = [i for i in os.listdir(sedthick_dir) if i.endswith(".nc")]
    for filename in filenames:
        try:
            resolution = _extract_sedthick_resolution(filename)
            break
        except Exception:
            pass
    else:
        raise ValueError(
            "Could not find sediment thickness files in directory: "
            + sedthick_dir
        )

    sedthick_filename = os.path.join(
        sedthick_dir, "sed_thick_{}d_{}.nc".format(resolution, time)
    )
    return sedthick_filename


def _extract_sedthick_resolution(filename):
    no_extension = os.extsep.join(filename.split(os.extsep)[:-1])
    split = no_extension.split("_")
    resolution = split[2].rstrip("d")
    return resolution
