import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from gplately import (
    PlateReconstruction,
    PlotTopologies,
    Raster,
)

FIGURE_SIZE_ORTHOGRAPHIC = (10, 10)
FIGURE_SIZE_MOLLWEIDE = (16, 9)
FONT_SIZE = 20
TICK_SIZE = FONT_SIZE * 0.7
TITLE_SIZE = FONT_SIZE * 1.8
BACKGROUND_COLOUR = "0.95"

COASTLINES_KWARGS = {
    "edgecolor": "grey",
    "facecolor": "lightgrey",
    "zorder": 1,
}
TOPOLOGIES_KWARGS = {
    "color": "black",
    "zorder": COASTLINES_KWARGS["zorder"] + 1,
}
RIDGES_KWARGS = {
    "color": "red",
    "zorder": TOPOLOGIES_KWARGS["zorder"] + 0.1,
}
TEETH_KWARGS = {
    "size": 250.0e3,
    "facecolor": "black",
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
    "alpha": 0.7,
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
    central_meridian=0.0,
):
    if not isinstance(gplot, PlotTopologies):
        gplot = _get_gplot(**gplot)

    if time is not None and gplot.time != time:
        gplot.time = time

    if positives is not None:
        if not isinstance(positives, pd.DataFrame):
            try:
                positives = pd.read_csv(positives)
            except Exception:
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
        projection = ccrs.Mollweide()
    elif str(projection).lower() == "orthographic":
        projection = ccrs.Orthographic(-100, 10)
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

    im = raster.imshow(ax=ax, **IMSHOW_KWARGS)
    gplot.plot_coastlines(
        ax=ax,
        tessellate_degrees=0.1,
        central_meridian=central_meridian,
        **COASTLINES_KWARGS,
    )
    gplot.plot_all_topologies(
        ax=ax,
        tessellate_degrees=0.1,
        central_meridian=central_meridian,
        **TOPOLOGIES_KWARGS,
    )
    gplot.plot_ridges_and_transforms(
        ax=ax,
        tessellate_degrees=0.1,
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

    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    ax.set_global()

    cbar.ax.set_ylim(0, 100)
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
        topology_features=topology_features,
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
