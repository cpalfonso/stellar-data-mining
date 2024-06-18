import glob
import os
from tempfile import NamedTemporaryFile
from typing import Optional

import pygplates
from gplately import (
    PlateReconstruction,
    PlotTopologies,
)
from plate_model_manager import PlateModelManager

from .check_files import check_plate_model  # re-export
from .misc import filter_topological_features

PMM = PlateModelManager()


def _fetch(model_name: str, model_dir: str = "plate_model"):
    if model_name not in PMM.get_available_model_names():
        raise ValueError(
            f"Invalid plate model name: {model_name}"
        )
    model = PMM.get_model(
        model_name,
        data_dir=model_dir,
    )
    if model is None:
        raise ValueError(
            f"Invalid plate model name: {model_name}"
        )
    return model


def get_plate_reconstruction(
    model_name: Optional[str] = None,
    model_dir: str = "plate_model",
    anchor_plate_id: int = 0,
    filter_topologies: bool = False,
):
    """Get a `PlateReconstruction` object from a model name and directory.

    If `model_name` is `None`, use local files in `model_dir`.

    Parameters
    ----------
    model_name : str, optional
        Name of the model to fetch from PMM.
    model_dir : str, default: 'plate_model'
        Directory to store files.
    anchor_plate_id : int, default: 0
        Anchor plate ID of the plate model.
    filter_topologies : bool, default: False
        Remove inactive deforming networks and flat slab topologies from
        the model. Requires writing to a temporary file.

    Returns
    -------
    PlateReconstruction
        The output plate model.
    tf : tempfile._TemporaryFileWrapper, optional
        If `filter_topologies` is True, a handle to the temporary
        file containing the topologies will also be returned, to prevent
        it from being cleaned up by the garbage collector and the file
        potentially being deleted.

    Raises
    ------
    ValueError
        If `model_name` is not recognised by PMM.
    """
    if model_name is None:
        globs = ["*.gpml", "*.gpmlz"]
        rotation_files = []
        topology_files = []
        static_polygons = []
        for g in globs:
            filenames = glob.glob(os.path.join(model_dir, "**", g), recursive=True)
            topology_files.extend(filenames)
            rotation_files.extend(filenames)
            static_polygons.extend(
                [
                    i for i in filenames
                    if "static" in os.path.basename(i).lower()
                    and "polygon" in os.path.basename(i).lower()
                ]
            )
        rotation_files.extend(
            glob.glob(os.path.join(model_dir, "**", "*.rot"), recursive=True)
        )

    else:
        model = _fetch(model_name, os.path.join(model_dir, ".downloaded"))
        rotation_files = model.get_rotation_model()
        topology_files = model.get_layer("Topologies")
        static_polygons = model.get_layer("StaticPolygons")

    if filter_topologies:
        topology_features = filter_topological_features(topology_files)
        tf = NamedTemporaryFile(suffix=".gpml")
        topology_features.write(tf.name)
        topology_files = [tf.name]

    plate_reconstruction = PlateReconstruction(
        rotation_model=rotation_files,
        topology_features=topology_files,
        static_polygons=static_polygons,
        anchor_plate_id=anchor_plate_id,
    )
    if filter_topologies:
        return plate_reconstruction, tf
    return plate_reconstruction


def get_plot_topologies(
    model_name: Optional[str] = None,
    model_dir: str = "plate_model",
    anchor_plate_id: int = 0,
    time: Optional[int] = None,
    plate_reconstruction: Optional[PlateReconstruction] = None,
    filter_topologies: bool = False,
):
    """Get a `PlotTopologies` object from a model name and directory.

    If `model_name` is `None`, use local files in `model_dir`.

    Parameters
    ----------
    model_name : str, optional
        Name of the model to fetch from PMM.
    model_dir : str, default: 'plate_model'
        Directory to store files.
    anchor_plate_id : int, default: 0
        Anchor plate ID of the plate model.
    time : int, optional
        If provided, set the `PlotTopologies` object to this time.
    plate_reconstruction: PlateReconstruction, optional
        If provided, use this `PlateReconstruction` instead of fetching
        all files from PMM.
    filter_topologies : bool, default: False
        Remove inactive deforming networks and flat slab topologies from
        the model. Requires writing to a temporary file.

    Returns
    -------
    `PlotTopologies`

    Raises
    ------
    ValueError
        If `model_name` is not recognised by PMM.
    """
    if plate_reconstruction is None:
        plate_reconstruction = get_plate_reconstruction(
            model_name=model_name,
            model_dir=os.path.join(model_dir, ".downloaded"),
            anchor_plate_id=anchor_plate_id,
            filter_topologies=filter_topologies,
        )

    if model_name is None:
        file_exts = ["*.gpml", "*.gpmlz"]
        coastlines = []
        for g in file_exts:
            filenames = glob.glob(os.path.join(model_dir, "**", g), recursive=True)
            for filename in filenames:
                basename = os.path.basename(filename)
                if "coast" in basename.lower():
                    coastlines.append(filename)
    else:
        model = _fetch(model_name, os.path.join(model_dir, ".downloaded"))
        coastlines = model.get_layer("Coastlines")

    return PlotTopologies(
        plate_reconstruction=plate_reconstruction,
        coastlines=coastlines,
        anchor_plate_id=anchor_plate_id,
        time=time,
    )
