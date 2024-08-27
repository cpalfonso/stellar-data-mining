"""Functions to rasterise topological plate models."""
import os
import warnings
from sys import stderr
from typing import (
    List,
    Optional,
    Sequence,
    Union,
)

import numpy as np
import pygplates
from gplately import (
    PlateReconstruction,
    PlotTopologies,
    Raster,
)
from rasterio.enums import MergeAlg
from rasterio.features import rasterize
from rasterio.transform import from_bounds

INCREMENT = 1
DEFAULT_RESOLUTION = 0.1


def run_create_plate_map(
    nprocs: int,
    times: Sequence[float],
    topology_features: pygplates.FeatureCollection,
    rotation_model: pygplates.RotationModel,
    output_dir: Optional[Union[os.PathLike, str]] = None,
    resolution: float = DEFAULT_RESOLUTION,
    tessellate_degrees: Optional[float] = None,
    return_output: bool = True,
    verbose: bool = False,
) -> Optional[List[Raster]]:
    """Rasterise a topological plate model at the given times.

    Parameters
    ----------
    nprocs : int
        Number of processes to use.
    times : sequence of float
        Times at which to rasterise the plate model.
    topology_features : FeatureCollection, optional
        Topological features for plate reconstruction.
    rotation_model : RotationModel, optional
        Rotation model for plate reconstruction.
    output_dir : str, optional
        If provided, write rasters to netCDF files in this directory.
    resolution : float, default: 0.1
        Resolution of the raster (degrees).
    tessellate_degrees : float, optional
        Densify topological geometries to this resolution (arc-degrees)
        before resolving plate model.
    return_output : bool, default False
        Return output (in xarray `Dataset` format).
    verbose : bool, default: False
        Print log to stderr.

    Returns
    -------
    sequence of gplately.Raster
        The rasterised plate model, if `return_output = True`.
    """
    if output_dir is not None:
        output_dir = os.path.abspath(output_dir)
        if not os.path.exists(output_dir):
            if verbose:
                print(
                    "Output directory does not exist; "
                    + f"creating now: {output_dir}",
                    file=stderr,
                )
            os.makedirs(output_dir, exist_ok=True)

    if nprocs == 1:
        return _run_subset(
            times=times,
            topology_features=topology_features,
            rotation_model=rotation_model,
            output_dir=output_dir,
            resolution=resolution,
            tessellate_degrees=tessellate_degrees,
            return_output=return_output,
            verbose=verbose,
        )
    else:
        from joblib import Parallel, delayed

        # Only load plate model files `nprocs` times
        subsets = np.array_split(times, nprocs)
        with Parallel(nprocs, verbose=int(verbose)) as parallel:
            results_nested = parallel(
                delayed(_run_subset)(
                    times=i,
                    topology_features=topology_features,
                    rotation_model=rotation_model,
                    output_dir=output_dir,
                    resolution=resolution,
                    tessellate_degrees=tessellate_degrees,
                    return_output=return_output,
                    verbose=verbose,
                )
                for i in subsets
            )
        if return_output:
            results = []
            for i in results_nested:
                results.extend(i)
            return results
    return None


def _run_subset(
    times,
    topology_features,
    rotation_model,
    output_dir=None,
    resolution=DEFAULT_RESOLUTION,
    tessellate_degrees=None,
    return_output=True,
    verbose=False,
):
    plate_reconstruction = PlateReconstruction(
        rotation_model=rotation_model,
        topology_features=topology_features,
    )
    results = []
    for time in times:
        if output_dir is None:
            output_filename = None
        else:
            output_filename = os.path.join(
                output_dir,
                f"plate_ids_{int(np.around(time))}Ma.nc",
            )
        result = create_plate_map(
            time=time,
            plate_reconstruction=plate_reconstruction,
            resolution=resolution,
            tessellate_degrees=tessellate_degrees,
            output_filename=output_filename,
            verbose=verbose,
        )
        if return_output:
            results.append(result)
    if return_output:
        return results


def create_plate_map(
    time: float,
    plate_reconstruction: Optional[PlateReconstruction] = None,
    topology_features: Optional[pygplates.FeatureCollection] = None,
    rotation_model: Optional[pygplates.RotationModel] = None,
    resolution: float = DEFAULT_RESOLUTION,
    tessellate_degrees: Optional[float] = None,
    output_filename: Optional[Union[os.PathLike, str]] = None,
    verbose: bool = False,
) -> Raster:
    """Rasterise a topological plate model at a given time.

    Parameters
    ----------
    time : float
        Time at which to rasterise the plate model.
    plate_reconstruction : PlateReconstruction, optional
        Plate reconstruction to rasterise.
    topology_features : FeatureCollection, optional
        Topological features for plate reconstruction. Used if
        `plate_reconstruction` is not provided.
    rotation_model : RotationModel, optional
        Rotation model for plate reconstruction. Used if
        `plate_reconstruction` is not provided.
    resolution : float, default: 0.1
        Resolution of the raster (degrees).
    tessellate_degrees : float, optional
        Densify topological geometries to this resolution (arc-degrees)
        before resolving plate model.
    output_filename : str, optional
        If provided, write raster to this netCDF file.
    verbose : bool, default: False
        Print log to stderr.

    Returns
    -------
    gplately.Raster
        The rasterised plate model.
    """
    time = float(time)
    resolution = float(resolution)
    if tessellate_degrees is None:
        tessellate_degrees = resolution
    tessellate_degrees = float(tessellate_degrees)

    if not isinstance(plate_reconstruction, PlateReconstruction):
        if (topology_features is None) or (rotation_model is None):
            raise TypeError(
                "Either plate_reconstruction or both of "
                + "topology_features and rotation_model "
                + "must be provided"
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ImportWarning)
            plate_reconstruction = PlateReconstruction(
                rotation_model=rotation_model,
                topology_features=topology_features,
            )
    gplot = PlotTopologies(plate_reconstruction)
    gplot.time = time

    topologies = gplot.get_all_topologies(
        tessellate_degrees=tessellate_degrees,
    )
    topologies["feature_type"] = topologies["feature_type"].astype(str)

    sort_key = lambda ftype: ftype.apply(
        lambda s: {
            "gpml:TopologicalNetwork": 0,
            "gpml:OceanicCrust": 1,
            "gpml:TopologicalClosedPlateBoundary": 2,
        }.get(s, -1)
    )
    minx = -180
    maxx = 180
    miny = -90
    maxy = 90

    lons = np.arange(minx, maxx + resolution, resolution)
    lats = np.arange(miny, maxy + resolution, resolution)
    nx = lons.size
    ny = lats.size
    transform = from_bounds(
        minx,
        miny,
        maxx,
        maxy,
        nx,
        ny,
    )

    topologies = topologies.sort_values(by="feature_type", key=sort_key)
    shapes = zip(
        topologies["geometry"],
        topologies["reconstruction_plate_ID"],
    )
    grid = rasterize(
        shapes=shapes,
        out_shape=(ny, nx),
        fill=-1,
        dtype=np.int_,
        merge_alg=MergeAlg.replace,
        transform=transform,
    )
    # Output is always upper-left origin
    grid = np.flipud(grid)  # convert to lower-left
    raster = Raster(grid, extent="global", origin="lower")
    if output_filename is not None:
        if verbose:
            print(
                " - Writing output file: "
                + os.path.basename(output_filename),
                file=stderr,
                flush=True,
            )
        raster.save_to_netcdf4(output_filename)
    return raster
