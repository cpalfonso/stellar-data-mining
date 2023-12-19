import os
from typing import (
    Sequence,
    Union,
)

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry.base import BaseGeometry

_DIRNAME = os.path.abspath(os.path.dirname(__file__))
DEFAULT_REGIONS_FILE = os.path.join(
    _DIRNAME,
    "..",
    "source_data",
    "regions.shp",
)


def assign_regions(
    points: Union[Sequence[BaseGeometry], gpd.GeoDataFrame, gpd.GeoSeries],
    regions: Union[str, gpd.GeoDataFrame] = DEFAULT_REGIONS_FILE,
) -> pd.Series:
    if isinstance(points, gpd.GeoDataFrame):
        # Copy
        points = gpd.GeoDataFrame(points)
    else:
        if hasattr(points, "index") and not callable(points.index):
            index = points.index
        else:
            index = np.arange(len(points))
        points = gpd.GeoDataFrame(
            {"geometry": points},
            index=index,
        )

    if isinstance(regions, gpd.GeoDataFrame):
        # Copy
        regions = gpd.GeoDataFrame(regions)
    else:
        # Read from file
        regions = gpd.read_file(regions)

    return (
        points
            .sjoin(regions, how="left")
            .replace({"region": {np.nan: "Other"}})
            ["region"]
    )
