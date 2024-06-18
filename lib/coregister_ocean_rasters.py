"""Functions to join subduction zone kinematic data to
time-dependent ocean plate raster data.
"""
import os
import warnings
from sys import stderr
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import pygplates
import xarray as xr
from gplately import PlateReconstruction
from sklearn.neighbors import NearestNeighbors
from skimage.transform import resize

from .create_plate_maps import create_plate_map
from .misc import (
    _PathLike,
    _PathOrDataFrame,
    _FeatureCollectionInput,
    _RotationModelInput,
)

INCREMENT = 1


def run_coregister_ocean_rasters(
    nprocs: int,
    times: Sequence[float],
    input_data: Union[_PathLike, Sequence[pd.DataFrame]],
    output_dir: Optional[_PathLike] = None,
    combined_filename: Optional[_PathLike] = None,
    plate_reconstruction: Optional[PlateReconstruction] = None,
    topology_features: Optional[_FeatureCollectionInput] = None,
    rotation_model: Optional[_RotationModelInput] = None,
    plates_dir: Optional[_PathLike] = None,
    agegrid_dir: Optional[_PathLike] = None,
    sedthick_dir: Optional[_PathLike] = None,
    carbonate_dir: Optional[_PathLike] = None,
    co2_dir: Optional[_PathLike] = None,
    subducted_thickness_dir: Optional[_PathLike] = None,
    subducted_sediments_dir: Optional[_PathLike] = None,
    subducted_carbonates_dir: Optional[_PathLike] = None,
    subducted_water_dir: Optional[_PathLike] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Join time-dependent subduction zone data to raster data.

    Parameters
    ----------
    nprocs : int
        The number of processes to use.
    times : sequence of float.
        The time steps of the data.
    input_data : str
        The input subduction zone data directory.
    output_dir : str, optional
        If provided, write joined data to CSV files in this directory.
    combined_filename : str, optional
        If provided, write combined joined data to this CSV file.
    plate_reconstruction : PlateReconstruction, optional
        Plate reconstruction used to restrict raster data to downgoing plate
        only.
    topology_features : FeatureCollection, optional
        Topological features used to restrict raster data to
        downgoing plate only. Used if `plate_reconstruction` is not provided.
    rotation_model : RotationModel, optional
        Rotation model used to restrict raster data to
        downgoing plate only. Used if `plate_reconstruction` is not provided.
    plates_dir : str, optional
        Directory containing rasterised topological plate maps (required if
        `topology_features` and `rotation_model` are not provided).
    agegrid_dir : str, optional
        Directory containing seafloor age raster data.
    sedthick_dir : str, optional
        Directory containing sediment thickness raster data.
    carbonate_dir : str, optional
        Directory containing carbonate thickness raster data.
    co2_dir : str, optional
        Directory containing C02 thickness raster data.
    subducted_thickness_dir : str, optional
        Directory containing cumulative subducted plate thickness raster data.
    subducted_sediments_dir : str, optional
        Directory containing cumulative subducted sediment raster data.
    subducted_carbonates_dir : str, optional
        Directory containing cumulative subducted carbonates raster data.
    subducted_water_dir : str, optional
        Directory containing cumulative subducted water raster data.
    verbose : bool, default: False
        Print log to stderr.

    Returns
    -------
    DataFrame
        The joined dataset.
    """
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

    if output_dir is not None and not os.path.isdir(output_dir):
        if verbose:
            print(
                "Output directory does not exist; creating now: "
                + str(output_dir),
                file=stderr,
            )
        os.makedirs(output_dir, exist_ok=True)

    if nprocs == 1:
        out = _run_subset(
            times=times,
            dfs=input_data,
            agegrid_dir=agegrid_dir,
            sedthick_dir=sedthick_dir,
            carbonate_dir=carbonate_dir,
            co2_dir=co2_dir,
            output_dir=output_dir,
            plate_reconstruction=plate_reconstruction,
            topology_features=topology_features,
            rotation_model=rotation_model,
            plates_dir=plates_dir,
            subducted_thickness_dir=subducted_thickness_dir,
            subducted_sediments_dir=subducted_sediments_dir,
            subducted_carbonates_dir=subducted_carbonates_dir,
            subducted_water_dir=subducted_water_dir,
        )
    else:
        from joblib import Parallel, delayed

        times_split = np.array_split(times, nprocs)
        df_array = np.empty(len(input_data), dtype="object")
        for i, df in enumerate(input_data):
            df_array[i] = df
        input_data_split = np.array_split(df_array, nprocs)

        with Parallel(nprocs, verbose=int(verbose)) as parallel:
            results = parallel(
                delayed(_run_subset)(
                    times=t,
                    dfs=d,
                    agegrid_dir=agegrid_dir,
                    sedthick_dir=sedthick_dir,
                    carbonate_dir=carbonate_dir,
                    co2_dir=co2_dir,
                    output_dir=output_dir,
                    plate_reconstruction=plate_reconstruction,
                    topology_features=topology_features,
                    rotation_model=rotation_model,
                    plates_dir=plates_dir,
                    subducted_thickness_dir=subducted_thickness_dir,
                    subducted_sediments_dir=subducted_sediments_dir,
                    subducted_carbonates_dir=subducted_carbonates_dir,
                    subducted_water_dir=subducted_water_dir,
                )
                for t, d in zip(times_split, input_data_split)
            )
        out = []
        for i in results:
            out.extend(i)

    out = pd.concat(out, ignore_index=True)
    if combined_filename is not None:
        out.to_csv(combined_filename, index=False)
    return out


def _run_subset(
    times,
    dfs,
    agegrid_dir,
    sedthick_dir,
    carbonate_dir,
    co2_dir,
    output_dir,
    plate_reconstruction=None,
    topology_features=None,
    rotation_model=None,
    plates_dir=None,
    **kwargs
):
    if plates_dir is None and plate_reconstruction is None:
        if not isinstance(topology_features, pygplates.FeatureCollection):
            topology_features = pygplates.FeatureCollection(
                pygplates.FeaturesFunctionArgument(
                    topology_features
                ).get_features()
            )
        if not isinstance(rotation_model, pygplates.RotationModel):
            rotation_model = pygplates.RotationModel(rotation_model)

    return [
        coregister_ocean_rasters(
            time=t,
            df=df,
            agegrid_dir=agegrid_dir,
            sedthick_dir=sedthick_dir,
            carbonate_dir=carbonate_dir,
            co2_dir=co2_dir,
            output_dir=output_dir,
            plate_reconstruction=plate_reconstruction,
            topology_features=topology_features,
            rotation_model=rotation_model,
            plates_dir=plates_dir,
            **kwargs,
        )
        for t, df in zip(times, dfs)
    ]


def coregister_ocean_rasters(
    time: float,
    df: _PathOrDataFrame,
    agegrid_dir: _PathLike,
    sedthick_dir: _PathLike,
    carbonate_dir: _PathLike,
    co2_dir: _PathLike,
    output_dir: _PathLike,
    plate_reconstruction: Optional[PlateReconstruction] = None,
    topology_features: Optional[_FeatureCollectionInput] = None,
    rotation_model: Optional[_RotationModelInput] = None,
    plates_dir: Optional[_PathLike] = None,
    subducted_thickness_dir: Optional[_PathLike] = None,
    subducted_sediments_dir: Optional[_PathLike] = None,
    subducted_carbonates_dir: Optional[_PathLike] = None,
    subducted_water_dir: Optional[_PathLike] = None,
    **kwargs
) -> pd.DataFrame:
    """Join subduction zone data to raster data.

    Parameters
    ----------
    time : float
        The time step of the data.
    df : str or DataFrame
        The input subduction zone data.
    agegrid_dir : str
        Directory containing seafloor age raster data.
    sedthick_dir : str
        Directory containing sediment thickness raster data.
    carbonate_dir : str
        Directory containing carbonate thickness raster data.
    co2_dir : str
        Directory containing C02 thickness raster data.
    output_dir : str, optional
        If provided, write joined data to a CSV file in this directory.
    plate_reconstruction : PlateReconstruction, optional
        Plate reconstruction used to restrict raster data to downgoing plate
        only.
    topology_features : FeatureCollection, optional
        Topological features used to restrict raster data to
        downgoing plate only. Used if `plate_reconstruction` is not provided.
    rotation_model : RotationModel, optional
        Rotation model used to restrict raster data to
        downgoing plate only. Used if `plate_reconstruction` is not provided.
    plates_dir : str, optional
        Directory containing rasterised topological plate maps (required if
        `topology_features` and `rotation_model` are not provided).
    subducted_thickness_dir : str, optional
        Directory containing cumulative subducted plate thickness raster data.
    subducted_sediments_dir : str, optional
        Directory containing cumulative subducted sediment raster data.
    subducted_carbonates_dir : str, optional
        Directory containing cumulative subducted carbonates raster data.
    subducted_water_dir : str, optional
        Directory containing cumulative subducted water raster data.
    **kwargs : dict
        Any further keyword arguments are passed along to create_plate_map.

    Returns
    -------
    DataFrame
        The joined dataset.
    """
    if isinstance(df, str):
        df = pd.read_csv(df)
    else:
        df = pd.DataFrame(df)

    if plates_dir is None:
        dset = create_plate_map(
            time=time,
            plate_reconstruction=plate_reconstruction,
            topology_features=topology_features,
            rotation_model=rotation_model,
            **kwargs,
        )
        plates = np.array(dset["plate_id"])
    else:
        plates_filename = os.path.join(
            plates_dir,
            "plate_ids_{}Ma.nc".format(time),
        )
        with xr.open_dataset(plates_filename) as dset:
            plates = np.array(dset["plate_id"])
    plates[np.isnan(plates)] = -1
    plates = plates.astype(np.int_)

    if agegrid_dir is None:
        agegrid_filename = None
    else:
        agegrid_filename = os.path.join(
            agegrid_dir, f"seafloor_age_{time:0.0f}Ma.nc"
        )
        if not os.path.isfile(agegrid_filename):
            raise FileNotFoundError(
                "Age grid file not found: " + agegrid_filename
            )

    if sedthick_dir is None:
        sedthick_filename = None
    else:
        sedthick_filename = os.path.join(
            sedthick_dir, f"sediment_thickness_{time:0.0f}Ma.nc"
        )
        if not os.path.isfile(sedthick_filename):
            raise FileNotFoundError(
                "Sediment thickness file not found: " + sedthick_filename
            )

    if co2_dir is None:
        co2_filename = None
    else:
        co2_filename = os.path.join(
            co2_dir,
            "crustal_co2_{}Ma.nc".format(time),
        )
        if not os.path.isfile(co2_filename):
            raise FileNotFoundError(
                "Crustal CO2 file not found: " + co2_filename
            )

    if carbonate_dir is None:
        carbonate_filename = None
    else:
        carbonate_filename = os.path.join(
            carbonate_dir, "carbonate_thickness_{}Ma.nc".format(time)
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

    raster_data = {}
    for filename, name in zip(
        (
            agegrid_filename,
            agegrid_filename,
            sedthick_filename,
            carbonate_filename,
            co2_filename,
            subducted_thickness_filename,
            subducted_sediments_filename,
            subducted_carbonates_filename,
            subducted_water_filename,
        ),
        (
            "agegrid",
            "spreadrate",
            "sedthick",
            "carbonate",
            "co2",
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
            if name == "agegrid":
                varname = "seafloor_age"
            elif name == "spreadrate":
                varname = "spreading_rate"
            else:
                varname = "z"
            raster = np.array(dset[varname])
            try:
                lon = np.array(dset["lon"])
            except KeyError:
                lon = np.array(dset["x"])
            try:
                lat = np.array(dset["lat"])
            except KeyError:
                lat = np.array(dset["y"])

        if raster.shape != plates.shape:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                raster = resize(raster, plates.shape, order=1, mode="wrap")
            lon = np.linspace(lon.min(), lon.max(), raster.shape[1])
            lat = np.linspace(lat.min(), lat.max(), raster.shape[0])

        raster_data[name]["data"] = raster
        raster_data[name]["lon"] = lon
        raster_data[name]["lat"] = lat

    column_names = {
        "agegrid": "seafloor_age (Ma)",
        "spreadrate": "seafloor_spreading_rate (km/Myr)",
        "sedthick": "sediment_thickness (m)",
        "carbonate": "carbonate_thickness (m)",
        "co2": "co2_volume (m^3/m^2)",
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
