import os

import geopandas as gpd
import numpy as np
from rasterio.enums import MergeAlg
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import xarray as xr

INCREMENT = 1


def run_create_plate_map(
    nprocs,
    min_time,
    max_time,
    input_dir,
    output_dir,
    resolution,
    verbose=False,
):
    times = range(min_time, max_time + INCREMENT, INCREMENT)

    if nprocs == 1:
        for time in times:
            create_plate_map(
                time,
                input_dir,
                output_dir,
                resolution,
            )
    else:
        from joblib import Parallel, delayed

        p = Parallel(nprocs, verbose=10 * int(verbose))
        p(
            delayed(create_plate_map)(
                time,
                input_dir,
                output_dir,
                resolution,
            )
            for time in times
        )


def create_plate_map(time, input_dir, output_dir, resolution):
    input_filename = os.path.join(
        input_dir, "plate_polygons_{}Ma.shp".format(time)
    )
    output_filename = os.path.join(output_dir, "plate_ids_{}Ma.nc".format(time))
    gdf = gpd.read_file(input_filename)
    gdf = gdf[gdf["GPGIM_TYPE"] == "gpml:TopologicalClosedPlateBoundary"]
    gdf["PLATEID1"] = (gdf["PLATEID1"]).astype("int32")

    shapes = zip(gdf["geometry"], gdf["PLATEID1"])
    lons = np.arange(-180.0, 180.0 + resolution, resolution)
    lats = np.arange(-90.0, 90.0 + resolution, resolution)
    nx = lons.size
    ny = lats.size

    arr = rasterize(
        shapes=shapes,
        out_shape=(ny, nx),
        fill=-1,
        dtype=np.int32,
        merge_alg=MergeAlg.replace,
        transform=from_bounds(-180.0, -90.0, 180.0, 90.0, nx, ny),
    )
    arr = np.flipud(arr)

    dset = xr.Dataset(
        data_vars={
            "plate_id": (("lat", "lon"), arr),
        },
        coords={
            "lon": lons,
            "lat": lats,
        },
    )
    encoding = {"plate_id": {"dtype": "int32", "zlib": True}}
    dset.to_netcdf(output_filename, encoding=encoding)
    return dset
