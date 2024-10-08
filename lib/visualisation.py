"""Functions for creating plots of prospectivity maps."""
import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pygplates
from gplately import (
    PlateReconstruction,
    PlotTopologies,
    Raster,
)
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

FIGURE_SIZE_ORTHOGRAPHIC = (10, 10)
FIGURE_SIZE_MOLLWEIDE = (16, 8)
FONT_SIZE = 20
TICK_SIZE = FONT_SIZE * 0.7
TITLE_SIZE = FONT_SIZE * 1.8
BACKGROUND_COLOUR = "0.95"
TESSELLATE_DEGREES = 0.1

COASTLINES_KWARGS = {
    "edgecolor": "grey",
    "linewidth": 0.5,
    "facecolor": "lightgrey",
    "zorder": 1,
    "tessellate_degrees": TESSELLATE_DEGREES,
}
TOPOLOGIES_KWARGS = {
    "color": "black",
    "zorder": COASTLINES_KWARGS["zorder"] + 1,
    "tessellate_degrees": TESSELLATE_DEGREES,
}
RIDGES_KWARGS = {
    "color": "red",
    "zorder": TOPOLOGIES_KWARGS["zorder"] + 0.1,
    "tessellate_degrees": TESSELLATE_DEGREES,
}
TEETH_KWARGS = {
    "size": 8,
    "aspect": 0.7,
    "spacing": 0.15,
    "markerfacecolor": "black",
    "markeredgecolor": "black",
    "zorder": TOPOLOGIES_KWARGS["zorder"],
}
SCATTER_KWARGS = {
    "linestyle": "none",
    "marker": "*",
    "markersize": 15,
    "markerfacecolor": "red",
    "markeredgecolor": "black",
    "markeredgewidth": 0.6,
    "transform": ccrs.PlateCarree(),
    "zorder": TEETH_KWARGS["zorder"] + 1,
}
IMSHOW_KWARGS = {
    "cmap": "RdYlBu_r",
    "vmin": 0,
    "vmax": 100,
    "alpha": 0.9,
    "zorder": 0.5 * (TOPOLOGIES_KWARGS["zorder"] + TEETH_KWARGS["zorder"]),
}
SAVEFIG_KWARGS = {
    "dpi": 250,
    "bbox_inches": "tight",
}


def plot(
    gplot,
    probabilities,
    projection=None,
    time=None,
    positives=None,
    output_filename=None,
    central_meridian=None,
    imshow_kwargs=None,
):
    """Create plot of probability raster grid.

    Parameters
    ----------
    gplot : gplately.PlotTopologies or dict
        PlotTopologies object, or dictionary containing all necessary
        keywoard arguments to create a PlotTopologies object.
    probabilities : gplately.Raster
        Probability raster grid.
    projection : str or cartopy.crs.Projection, default: "mollweide"
        Map projection to use.
    time : float
        Timestep to plot.
    positives : str or pandas.DataFrame
        Data frame containing known positive observations, for plotting on
        the map.
    output_filename : str, optional
        If provided, save image to this filename.
    central_meridian : float, default: 0.0
        Central meridian of map projection.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not isinstance(gplot, PlotTopologies):
        gplot = _get_gplot(**gplot)

    if time is not None and gplot.time != time:
        gplot.time = time

    if positives is not None:
        if isinstance(positives, str):
            positives = pd.read_csv(positives)
        else:
            positives = pd.DataFrame(positives)
        # Restrict to positives only
        positives = positives[positives["label"] == "positive"]
        if time is not None:
            # Restrict to +/- 2.5 Myr
            positives = positives[
                (positives["age (Ma)"] - float(time)).abs()
                <= 2.5
            ]

    if projection is None or str(projection).lower() == "mollweide":
        projection = ccrs.Mollweide(
            central_meridian if central_meridian is not None else 0
        )
    elif str(projection).lower() == "orthographic":
        projection = ccrs.Orthographic(
            central_meridian if central_meridian is not None else 0,
            10,
        )
    if not isinstance(projection, ccrs.CRS):
        raise TypeError(f"Invalid projection {projection}")

    if isinstance(projection, ccrs.Mollweide):
        figsize = FIGURE_SIZE_MOLLWEIDE
    elif isinstance(projection, ccrs.Orthographic):
        figsize = FIGURE_SIZE_ORTHOGRAPHIC
    else:  # determine figure size automatically
        figsize = None

    raster = Raster(probabilities)
    raster.data *= 100.0

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(
        [0.1, 0.1, 0.8, 0.8],
        projection=projection,
        facecolor=BACKGROUND_COLOUR,
    )
    cax = fig.add_axes([0.94, 0.1, 0.03, 0.8])

    imshow_kwargs = {} if imshow_kwargs is None else dict(imshow_kwargs)
    # imshow_kwargs = IMSHOW_KWARGS.copy().update(imshow_kwargs)
    tmp = IMSHOW_KWARGS.copy()
    tmp.update(imshow_kwargs)
    imshow_kwargs = tmp
    cmap = colormaps[imshow_kwargs.pop("cmap")]

    im = raster.imshow(
        ax=ax,
        cmap=cmap,
        **imshow_kwargs,
    )
    gplot.plot_coastlines(
        ax=ax,
        central_meridian=central_meridian,
        **COASTLINES_KWARGS,
    )
    gplot.plot_all_topological_sections(
        ax=ax,
        central_meridian=central_meridian,
        **TOPOLOGIES_KWARGS,
    )
    gplot.plot_ridges_and_transforms(
        ax=ax,
        central_meridian=central_meridian,
        **RIDGES_KWARGS,
    )
    gplot.plot_subduction_teeth(ax=ax, **TEETH_KWARGS)

    if positives is not None:
        ax.plot(
            "lon",
            "lat",
            data=positives,
            **SCATTER_KWARGS,
        )

    sm = ScalarMappable(
        norm=Normalize(
            imshow_kwargs.get("vmin", 0),
            imshow_kwargs.get("vmax", 100),
        ),
        cmap=cmap,
    )
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    ax.set_global()

    cbar.ax.set_ylim(0, 100)
    cbar.ax.fill_between(
        x=cbar.ax.get_xlim(),
        y1=imshow_kwargs.get("vmax", 100),
        y2=100,
        color=cmap.get_over(),
        # alpha=imshow_kwargs.get("alpha", 1),
        zorder=-1,
    )
    cbar.ax.fill_between(
        x=cbar.ax.get_xlim(),
        y1=0,
        y2=imshow_kwargs.get("vmin", 0),
        color=cmap.get_under(),
        # alpha=imshow_kwargs.get("alpha", 1),
        zorder=-1,
    )
    cbar.ax.set_title(
        "Deposit\nprobability (%)",
        fontsize=FONT_SIZE,
        y=1.025,
    )
    cbar.ax.tick_params(labelsize=TICK_SIZE)

    if time is not None:
        ax.set_title(
            f"{time} Ma",
            fontsize=TITLE_SIZE,
            y=1.04,
        )

    if output_filename is not None:
        fig.savefig(output_filename, **SAVEFIG_KWARGS)
        plt.close(fig)
    return fig


def _get_gplot(
    rotation_model,
    topology_features=None,
    static_polygons=None,
    coastlines=None,
    continents=None,
    COBs=None,
    time=None,
    anchor_plate_id=0,
):
    reconstruction = PlateReconstruction(
        rotation_model=rotation_model,
        topology_features=pygplates.FeatureCollection(
            [
                i for i in pygplates.FeaturesFunctionArgument(
                    topology_features
                ).get_features()
                if i.get_feature_type().to_qualified_string()
                != "gpml:TopologicalSlabBoundary"
            ]
    ),
        static_polygons=static_polygons,
    )
    gplot = PlotTopologies(
        plate_reconstruction=reconstruction,
        coastlines=coastlines,
        continents=continents,
        COBs=COBs,
        time=time,
        anchor_plate_id=anchor_plate_id,
    )
    return gplot
