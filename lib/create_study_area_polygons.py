# from itertools import repeat
# from multiprocessing import Pool, set_start_method
import os
# from platform import system

import geopandas as gpd
from gplately import EARTH_RADIUS
from gplately.geometry import wrap_geometries
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
import warnings

INCREMENT = 1

DEFAULT_SZ_BUFFER_DISTANCE = 6.0  # degrees

# if system == "Darwin":
#     try:
#         set_start_method("spawn")
#     except RuntimeError:
#         pass


def run_create_study_area_polygons(
    nprocs,
    min_time,
    max_time,
    input_dir,
    output_dir,
    buffer_distance=DEFAULT_SZ_BUFFER_DISTANCE,
    verbose=False,
):
    times = range(min_time, max_time + INCREMENT, INCREMENT)

    v = 10 if verbose else 0
    p = Parallel(nprocs, verbose=v)
    p(
        delayed(create_study_area_polygons)(
            time,
            input_dir,
            output_dir,
            buffer_distance,
        )
        for time in times
    )

    # if nprocs == 1:
    #     for time in times:
    #         create_study_area_polygons(
    #             time=time,
    #             input_dir=input_dir,
    #             output_dir=output_dir,
    #             buffer_distance=buffer_distance,
    #         )
    # else:
    #     n = len(times)
    #     chunk_size = int(n / nprocs / 4)
    #     chunk_size = max([chunk_size, 1])
    #     args = zip(
    #         times,
    #         repeat(input_dir, n),
    #         repeat(output_dir, n),
    #         repeat(buffer_distance, n),
    #     )
    #     with Pool(nprocs) as pool:
    #         pool.starmap(create_study_area_polygons, args, chunksize=chunk_size)
    #         pool.close()
    #         pool.join()


def create_study_area_polygons(
    time, input_dir, output_dir, buffer_distance=DEFAULT_SZ_BUFFER_DISTANCE
):
    topologies_filename = os.path.join(
        input_dir, "plate_boundaries_{}Ma.shp".format(time)
    )
    plate_polygons_filename = os.path.join(
        input_dir, "plate_polygons_{}Ma.shp".format(time)
    )
    output_filename = os.path.join(
        output_dir, "study_area_{}Ma.shp".format(time)
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)

        topologies = gpd.read_file(topologies_filename)
        topologies = topologies[
            (topologies["over"] != -1) & (topologies["over"] != 0)
        ]
        topologies = topologies.explode(ignore_index=True)
        global_bounds_buffer = 1.0e0
        mask = box(
            -180.0 + global_bounds_buffer,
            -90.0 + global_bounds_buffer,
            180.0 - global_bounds_buffer,
            90.0 - global_bounds_buffer,
        )
        topologies = topologies.clip(mask)
        buffered = {}
        for _, row in topologies.iterrows():
            _buffer_sz(row, buffer_distance, topologies.crs, out=buffered)
        buffered = gpd.GeoDataFrame(
            buffered, geometry="geometry", crs=topologies.crs
        )

        plate_polygons = gpd.read_file(plate_polygons_filename)
        clipped = []
        for plate_id in buffered["over"].unique():
            intersection = gpd.overlay(
                buffered[buffered["over"] == plate_id],
                plate_polygons[plate_polygons["PLATEID1"] == plate_id],
            )
            if len(intersection) > 0:
                clipped.append(intersection)
        clipped = gpd.GeoDataFrame(pd.concat(clipped, ignore_index=True))
        clipped.to_file(output_filename)


def _buffer_sz(row, distance_degrees, crs, out):
    geom = gpd.GeoSeries(row["geometry"], crs=crs)
    point = geom.representative_point()
    proj = "+proj=aeqd +lat_0={} +lon_0={} +x_0=0 +y_0=0".format(
        point.y, point.x
    )
    projected = geom.to_crs(proj)

    distance_metres = np.deg2rad(distance_degrees) * EARTH_RADIUS * 1000.0
    projected_buffered = projected.buffer(distance_metres)
    buffered = projected_buffered.to_crs(crs)
    geometry_out = buffered[0]
    # geometries_out = wrap(geometry_out, central_meridian=0.0)
    geometries_out = wrap_geometries(
        geometry_out, central_meridian=0.0, tessellate_degrees=0.1
    )
    if isinstance(geometries_out, BaseGeometry):
        geometries_out = [geometries_out]

    for i in geometries_out:
        for column_name in row.index:
            if column_name == "geometry":
                continue
            if column_name not in out:
                out[column_name] = [row[column_name]]
            else:
                out[column_name].append(row[column_name])
        if "geometry" not in out:
            # out["geometry"] = [buffered[0]]
            out["geometry"] = [i]
        else:
            # out["geometry"].append(buffered[0])
            out["geometry"].append(i)
