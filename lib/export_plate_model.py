import os
import warnings

import geopandas as gpd
from gplately.geometry import pygplates_to_shapely
from gplately import gpml
import pygplates

INCREMENT = 1


def run_export_plate_model(
    nprocs,
    min_time,
    max_time,
    topology_filenames,
    rotation_filenames,
    coastline_filenames,
    output_dir,
    verbose=False,
):
    times = range(min_time, max_time + INCREMENT, INCREMENT)

    if nprocs == 1:
        for time in times:
            export_plate_model(
                time,
                topology_filenames,
                rotation_filenames,
                coastline_filenames,
                output_dir,
            )
    else:
        from joblib import Parallel, delayed

        p = Parallel(nprocs, verbose=10 * int(verbose))
        p(
            delayed(export_plate_model)(
                time,
                topology_filenames,
                rotation_filenames,
                coastline_filenames,
                output_dir,
            )
            for time in times
        )


def export_plate_model(
    time,
    topology_filenames,
    rotation_filenames,
    coastline_filenames,
    output_dir,
):
    rotation_model = pygplates.RotationModel(rotation_filenames)
    inactive_filenames = [
        i for i in topology_filenames
        if "inactive" in os.path.basename(i).lower()
    ]
    topology_collection = pygplates.FeatureCollection(
        pygplates.FeaturesFunctionArgument(topology_filenames).get_features()
    )
    coastline_collection = pygplates.FeatureCollection(
        pygplates.FeaturesFunctionArgument(coastline_filenames).get_features()
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning, append=True)
        _export_coastlines(
            time=time,
            rotation_model=rotation_model,
            coastline_features=coastline_collection,
            output_dir=output_dir,
        )

    active_filenames = [
        i for i in topology_filenames if i not in inactive_filenames
    ]
    active_topologies_dict = gpml.get_topological_references(
        active_filenames,
        id_type=str,
    )
    referenced_features = []
    for i in active_topologies_dict.values():
        referenced_features.extend(i)
    referenced_features.extend(active_topologies_dict.keys())
    referenced_features = set(referenced_features)

    feature_dict = gpml.create_feature_dict(
        topology_collection,
        id_type=str,
    )
    tmp = pygplates.FeatureCollection()
    for feature in topology_collection:
        id = feature.get_feature_id().get_string()
        if id in referenced_features:
            tmp.add(feature_dict[id])
    topology_collection = tmp
    del tmp

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning, append=True)
        _export_topologies(
            time=time,
            rotation_model=rotation_model,
            topology_features=topology_collection,
            output_dir=output_dir,
        )


def _export_coastlines(time, rotation_model, coastline_features, output_dir):
    output_filename = os.path.join(
        output_dir, "coastlines_{}Ma.shp".format(time)
    )
    pygplates.reconstruct(
        coastline_features,
        rotation_model,
        output_filename,
        float(time),
    )


def _export_topologies(
    time,
    rotation_model,
    topology_features,
    output_dir,
):
    polygons_filename = os.path.join(
        output_dir, "plate_polygons_{}Ma.shp".format(time)
    )
    pygplates.resolve_topologies(
        topology_features,
        rotation_model,
        polygons_filename,
        float(time),
    )

    boundaries_filename = os.path.join(
        output_dir, "plate_boundaries_{}Ma.shp".format(time)
    )

    resolved_sections = []
    pygplates.resolve_topologies(
        topology_features,
        rotation_model,
        [],  # discard resolved boundaries/networks
        float(time),
        resolved_sections,
    )
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
    gdf.to_file(boundaries_filename)
