"""Functions to create time-dependent study area polygons along subduction
zones.
"""
import os
import warnings
from sys import stderr
from typing import (
    Hashable,
    Iterable,
    List,
    Optional,
    Sequence,
)

import geopandas as gpd
import numpy as np
import pandas as pd
import pygplates
from gplately import (
    PlateReconstruction,
    PlotTopologies,
    EARTH_RADIUS,
)
from gplately.geometry import (
    pygplates_to_shapely,
    wrap_geometries,
)
from joblib import Parallel, delayed
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from shapely.ops import linemerge

from .misc import (
    _FeatureCollectionInput,
    _PathLike,
    _RotationModelInput,
)

INCREMENT = 1

DEFAULT_SZ_BUFFER_DISTANCE = 6.0  # degrees


def run_create_study_area_polygons(
    nprocs: int,
    times: Sequence[float],
    plate_reconstruction: Optional[PlateReconstruction] = None,
    topological_features: Optional[_FeatureCollectionInput] = None,
    rotation_model: Optional[_RotationModelInput] = None,
    output_dir: _PathLike = os.curdir,
    buffer_distance: float = DEFAULT_SZ_BUFFER_DISTANCE,
    verbose: bool = False,
    return_output: bool = False,
) -> Optional[List[gpd.GeoDataFrame]]:
    """Create study area polygons at the given times.

    Parameters
    ----------
    nprocs : int
        Number of processes to use.
    times : sequence of float
        Times at which to extract study area.
    plate_reconstruction: PlateReconstruction, optional
        Plate reconstruction to use. If `None`, then both
        `topological_features` and `rotation_model` must be provided.
    topological_features : FeatureCollection
        Topological features for plate reconstruction.
    rotation_model : RotationModel
        Rotation model for plate reconstruction.
    output_dir : str, default: current directory
        Write output shapefiles to this directory.
    buffer_distance : float, default: 6.0
        Width of subduction zone study area (arc degrees).
    verbose : bool, default: False
        Print log to stderr.
    return_output : bool, default: False
        Return output (in GeoDataFrame format).

    Returns
    -------
    sequence of GeoDataFrame
        The subduction zone polygons (if `return_output = True`).
    """
    if plate_reconstruction is None:
        if topological_features is None or rotation_model is None:
            raise TypeError(
                "Either `plate_reconstruction` or both "
                "`topological_features` and `rotation_model` "
                "must not be None."
            )

    if output_dir is not None and not os.path.isdir(output_dir):
        if verbose:
            print(
                "Output directory does not exist; creating now: "
                + output_dir,
                file=stderr,
            )
        os.makedirs(output_dir, exist_ok=True)

    times_split = np.array_split(times, nprocs)
    with Parallel(nprocs, verbose=int(verbose)) as parallel:
        results = parallel(
            delayed(_multiple_timesteps)(
                times=t,
                plate_reconstruction=plate_reconstruction,
                topological_features=topological_features,
                rotation_model=rotation_model,
                output_dir=output_dir,
                buffer_distance=buffer_distance,
                return_output=return_output,
            )
            for t in times_split
        )
    if return_output:
        out = []
        for i in results:
            out.extend(i)
        return out
    return None


def _multiple_timesteps(
    times: Sequence[float],
    buffer_distance: float,
    return_output: bool,
    plate_reconstruction: Optional[PlateReconstruction] = None,
    topological_features: Optional[_FeatureCollectionInput] = None,
    rotation_model: Optional[_RotationModelInput] = None,
    output_dir: _PathLike = os.curdir,
):
    if plate_reconstruction is None:
        if not isinstance(topological_features, pygplates.FeatureCollection):
            topological_features = pygplates.FeatureCollection(
                pygplates.FeaturesFunctionArgument(
                    topological_features
                ).get_features()
            )
        if not isinstance(rotation_model, pygplates.RotationModel):
            rotation_model = pygplates.RotationModel(rotation_model)

    out = []
    for time in times:
        out.append(
            create_study_area_polygons(
                time=time,
                plate_reconstruction=plate_reconstruction,
                topological_features=topological_features,
                rotation_model=rotation_model,
                output_dir=output_dir,
                buffer_distance=buffer_distance,
                return_output=return_output,
            )
        )
    if return_output:
        return out


def create_study_area_polygons(
    time: float,
    plate_reconstruction: Optional[PlateReconstruction] = None,
    topological_features: Optional[_FeatureCollectionInput] = None,
    rotation_model: Optional[_RotationModelInput] = None,
    output_dir: _PathLike = os.curdir,
    buffer_distance: float = DEFAULT_SZ_BUFFER_DISTANCE,
    clip_to_overriding_plate: bool = False,
    return_output: bool = False,
) -> Optional[gpd.GeoDataFrame]:
    """Create study area polygons at a given time.

    Parameters
    ----------
    time : float
        Time at which to extract study area.
    plate_reconstruction: PlateReconstruction, optional
        Plate reconstruction to use. If `None`, then both
        `topological_features` and `rotation_model` must be provided.
    topological_features : FeatureCollection
        Topological features for plate reconstruction.
    rotation_model : RotationModel
        Rotation model for plate reconstruction.
    output_dir : str, default: current directory
        Write output shapefile to this directory.
    buffer_distance : float, default: 6.0
        Width of subduction zone study area (arc degrees).
    clip_to_overriding_plate : bool, default: False
        Clip output polygons to topological polygon corresponding to
        the subduction zone's overriding plate ID.
    return_output : bool, default: False
        Return output (in GeoDataFrame format).

    Returns
    -------
    GeoDataFrame
        The subduction zone polygons (if `return_output = True`).
    """
    if plate_reconstruction is None:
        if not isinstance(topological_features, pygplates.FeatureCollection):
            topological_features = pygplates.FeatureCollection(
                pygplates.FeaturesFunctionArgument(
                    topological_features
                ).get_features()
            )
        if not isinstance(rotation_model, pygplates.RotationModel):
            rotation_model = pygplates.RotationModel(rotation_model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ImportWarning)
            plate_reconstruction = PlateReconstruction(
                rotation_model=rotation_model,
                topology_features=topological_features,
            )
    else:
        topological_features = plate_reconstruction.topology_features
        rotation_model = plate_reconstruction.rotation_model

    gplot = PlotTopologies(plate_reconstruction)
    gplot.time = float(time)
    plate_polygons = gplot.get_all_topologies()
    plate_polygons["feature_type"] = plate_polygons["feature_type"].astype(str)
    plate_types = {
        "gpml:TopologicalClosedPlateBoundary",
        "gpml:OceanicCrust",
        "gpml:TopologicalNetwork",
    }
    plate_polygons = plate_polygons[
        plate_polygons["feature_type"].isin(plate_types)
    ]

    topologies = _extract_overriding_plates(
        time=time,
        topological_features=topological_features,
        rotation_model=rotation_model,
    )
    plate_polygons.crs = topologies.crs

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)

        topologies = topologies[
            (topologies["over"] != -1)
            & (topologies["over"] != 0)
            & (topologies["polarity"] != "None")
        ]
        topologies = topologies.explode(ignore_index=True)
        for i in topologies.index:
            if topologies.at[i, "polarity"].lower() != "left":
                topologies.at[i, "geometry"] = topologies.at[i, "geometry"].reverse()
                topologies.at[i, "polarity"] = "Left"
        topologies = _merge_lines(topologies)
        buffered = {}
        for _, row in topologies.iterrows():
            _buffer_sz(row, buffer_distance, topologies.crs, out=buffered)
        buffered = gpd.GeoDataFrame(
            buffered, geometry="geometry", crs=topologies.crs
        )

        if clip_to_overriding_plate:
            clipped = []
            for plate_id in buffered["over"].unique():
                intersection = gpd.overlay(
                    buffered[buffered["over"] == plate_id],
                    plate_polygons[plate_polygons["reconstruction_plate_ID"] == plate_id],
                )
                if len(intersection) > 0:
                    clipped.append(intersection)
            clipped = gpd.GeoDataFrame(pd.concat(clipped, ignore_index=True))
            clipped = clipped[["name", "polarity", "feature_type", "over", "geometry"]]
            clipped = clipped.rename(
                columns={"over": "plate_id", "feature_type": "ftype"}
            )
            buffered = gpd.GeoDataFrame(clipped, geometry="geometry")

    if not buffered.geometry.is_valid.all():
        buffered.geometry = buffered.buffer(0)

    if output_dir is not None:
        output_filename = os.path.join(
            output_dir, f"study_area_{time:0.0f}Ma.geojson"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            buffered.to_file(output_filename)
    if return_output:
        return buffered
    return None


def _buffer_sz(row, distance_degrees, crs, out):
    geom = gpd.GeoSeries(row["geometry"], crs=crs)
    point = geom.representative_point()
    proj = "+proj=aeqd +lat_0={} +lon_0={} +x_0=0 +y_0=0".format(
        point.y, point.x
    )
    projected = geom.to_crs(proj)

    distance_metres = np.deg2rad(distance_degrees) * EARTH_RADIUS * 1000.0
    direction = 1.0 if str(row["polarity"]).lower() == "left" else -1.0
    projected_buffered = projected.buffer(
        distance_metres * direction,
        single_sided=True,
    )
    buffered = projected_buffered.to_crs(crs)
    geometry_out = buffered[0]
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
            out["geometry"] = [i]
        else:
            out["geometry"].append(i)
    return out


def _extract_overriding_plates(
    time,
    topological_features,
    rotation_model,
):
    resolved_sections = []
    pygplates.resolve_topologies(
        topological_features,
        rotation_model,
        [],  # discard boundaries/networks
        float(time),
        resolved_sections,
    )

    # Ignore flat slab topologies
    slab_types = {
        pygplates.FeatureType.gpml_slab_edge,
        pygplates.FeatureType.gpml_topological_slab_boundary,
    }
    resolved_sections = [
        i
        for i in resolved_sections
        if i.get_topological_section_feature().get_feature_type()
        not in slab_types
    ]

    geometries = []
    polarities = []
    names = []
    feature_types = []
    feature_ids = []
    plate_ids = []
    overriding_plates = []
    subducting_plates = []
    left_plates = []
    right_plates = []
    shared_1s = []
    shared_2s = []
    for i in resolved_sections:
        for segment in i.get_shared_sub_segments():
            geometry = segment.get_resolved_geometry()
            geometry = pygplates_to_shapely(geometry, tessellate_degrees=0.1)

            polarity = segment.get_feature().get_enumeration(
                pygplates.PropertyName.gpml_subduction_polarity,
                "None",
            )
            if polarity == "Unknown":
                polarity = "None"
            valid_polarities = {"None", "Left", "Right"}
            if polarity not in valid_polarities:
                warnings.warn(
                    "Unknown polarity: {}".format(polarity), RuntimeWarning
                )
                continue

            name = segment.get_feature().get_name()
            if "flat slab" in name.lower():
                continue

            feature_type = (
                segment.get_feature().get_feature_type().to_qualified_string()
            )
            feature_id = segment.get_feature().get_feature_id().get_string()
            plate_id = segment.get_feature().get_reconstruction_plate_id(-1)
            tmp = segment.get_overriding_and_subducting_plates()
            if tmp is None:
                overriding_plate = -1
                subducting_plate = -1
            else:
                overriding_plate, subducting_plate = tmp
                overriding_plate = (
                    overriding_plate.get_feature().get_reconstruction_plate_id(
                        -1
                    )
                )
                subducting_plate = (
                    subducting_plate.get_feature().get_reconstruction_plate_id(
                        -1
                    )
                )
            del tmp
            left_plate = segment.get_feature().get_left_plate(-1)
            right_plate = segment.get_feature().get_right_plate(-1)

            sharing_topologies = segment.get_sharing_resolved_topologies()
            if len(sharing_topologies) > 0:
                shared_1 = (
                    sharing_topologies[0]
                    .get_feature()
                    .get_reconstruction_plate_id(-1)
                )
            else:
                shared_1 = -1
            if len(sharing_topologies) > 1:
                shared_2 = (
                    sharing_topologies[1]
                    .get_feature()
                    .get_reconstruction_plate_id(-1)
                )
            else:
                shared_2 = -1

            geometries.append(geometry)
            polarities.append(polarity)
            names.append(name)
            feature_types.append(feature_type)
            feature_ids.append(feature_id)
            plate_ids.append(plate_id)
            overriding_plates.append(overriding_plate)
            subducting_plates.append(subducting_plate)
            left_plates.append(left_plate)
            right_plates.append(right_plate)
            shared_1s.append(shared_1)
            shared_2s.append(shared_2)

    gdf = gpd.GeoDataFrame(
        {
            "polarity": polarities,
            "geometry": geometries,
            "name": names,
            "type": feature_types,
            "id": feature_ids,
            "plate_id": plate_ids,
            "over": overriding_plates,
            "subd": subducting_plates,
            "left": left_plates,
            "right": right_plates,
            "shared_1": shared_1s,
            "shared_2": shared_2s,
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    return gdf


def _merge_lines(
    data: gpd.GeoDataFrame,
    groupby: Iterable[Hashable] = ("polarity", "type", "over"),
):
    out = []
    for gb_vals, grouped in data.groupby(list(groupby)):
        geom = linemerge(grouped.geometry.to_list())
        if isinstance(geom, BaseMultipartGeometry):
            geom = list(geom.geoms)
        else:
            geom = [geom]
        gb_data = {
            "geometry": geom,
            **{
                gb_col: gb_val
                for gb_col, gb_val
                in zip(groupby, gb_vals)
            }
        }
        if "name" not in gb_data.keys():
            gb_data["name"] = ":".join(grouped["name"].unique())
        out.append(
            gpd.GeoDataFrame(gb_data, geometry="geometry")
        )
    out = gpd.GeoDataFrame(
        pd.concat(out, ignore_index=True),
        geometry="geometry",
        crs=data.crs,
    )
    return out
