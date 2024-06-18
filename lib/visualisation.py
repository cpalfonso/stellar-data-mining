"""Functions for creating plots of prospectivity maps."""
from typing import (
    Any,
    Mapping,
    Optional,
    Union,
)

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygplates
from gplately import (
    PlateReconstruction,
    PlotTopologies,
    Raster,
)
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from .misc import reconstruct_by_topologies

FIGURE_SIZE_ORTHOGRAPHIC = (10, 10)
FIGURE_SIZE_MOLLWEIDE = (16, 8)
FONT_SIZE = 20
TICK_SIZE = FONT_SIZE * 0.7
TITLE_SIZE = FONT_SIZE * 1.8
SUPTITLE_SIZE = FONT_SIZE * 2.52
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
    "markersize": 20,
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
    imshow_kwargs: Optional[Mapping[str, Any]] = None,
    coastlines_kwargs: Optional[Mapping[str, Any]] = None,
    topologies_kwargs: Optional[Mapping[str, Any]] = None,
    ridges_kwargs: Optional[Mapping[str, Any]] = None,
    teeth_kwargs: Optional[Mapping[str, Any]] = None,
    scatter_kwargs: Optional[Mapping[str, Any]] = None,
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
    imshow_kwargs = {} if imshow_kwargs is None else imshow_kwargs
    imshow_kwargs = _copy_update_dict(IMSHOW_KWARGS, imshow_kwargs)

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
        # if time is not None:
        #     # Restrict to +/- 2.5 Myr
        #     positives = positives[
        #         (positives["age (Ma)"] - float(time)).abs()
        #         <= 2.5
        #     ]

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

    cmap = colormaps[imshow_kwargs["cmap"]]

    sm = _prepare_axes(
        ax=ax,
        gplot=gplot,
        raster=raster,
        time=time,
        positives=positives,
        central_meridian=central_meridian,
        imshow_kwargs=imshow_kwargs,
        coastlines_kwargs=coastlines_kwargs,
        topologies_kwargs=topologies_kwargs,
        ridges_kwargs=ridges_kwargs,
        teeth_kwargs=teeth_kwargs,
        scatter_kwargs=scatter_kwargs,
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

    # if time is not None:
    #     ax.set_title(
    #         f"{time} Ma",
    #         fontsize=TITLE_SIZE,
    #         y=1.04,
    #     )

    if output_filename is not None:
        fig.savefig(output_filename, **SAVEFIG_KWARGS)
        plt.close(fig)
    return fig


def plot_comparison(
    gplot,
    probs1,
    probs2,
    projection=None,
    time=None,
    positives1=None,
    positives2=None,
    output_filename=None,
    central_meridian=None,
    imshow_kwargs=None,
    title1=None,
    title2=None,
):
    if not isinstance(gplot, PlotTopologies):
        gplot = _get_gplot(**gplot)

    if time is not None and gplot.time != time:
        gplot.time = time

    if positives1 is not None:
        if isinstance(positives1, str):
            positives1 = pd.read_csv(positives1)
        else:
            positives1 = pd.DataFrame(positives1)
        # Restrict to positives only
        positives1 = positives1[positives1["label"] == "positive"]
        # if time is not None:
        #     # Restrict to +/- 2.5 Myr
        #     positives1 = positives1[
        #         (positives1["age (Ma)"] - float(time)).abs()
        #         <= 2.5
        #     ]
    if positives2 is not None:
        if isinstance(positives2, str):
            positives2 = pd.read_csv(positives2)
        else:
            positives2 = pd.DataFrame(positives2)
        # Restrict to positives only
        positives2 = positives2[positives2["label"] == "positive"]
        # if time is not None:
        #     # Restrict to +/- 2.5 Myr
        #     positives2 = positives2[
        #         (positives2["age (Ma)"] - float(time)).abs()
        #         <= 2.5
        #     ]

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
        figsize = (
            FIGURE_SIZE_MOLLWEIDE[0],
            FIGURE_SIZE_MOLLWEIDE[1] * 1.8,
        )
        nrows = 2
        ncols = 1
    elif isinstance(projection, ccrs.Orthographic):
        figsize = (
            FIGURE_SIZE_ORTHOGRAPHIC[0] * 1.8,
            FIGURE_SIZE_ORTHOGRAPHIC[1],
        )
        nrows = 1
        ncols = 2
    else:  # determine figure size automatically
        figsize = None
        nrows = 2
        ncols = 1

    fig, axs = plt.subplots(
        nrows, ncols,
        figsize=figsize,
        subplot_kw={"projection": projection},
        facecolor=BACKGROUND_COLOUR,
    )
    cax_pos = [
        axs[0].get_position().x0,
        axs[1].get_position().y0 * 0.5,
        axs[1].get_position().x1 - axs[0].get_position().x0,
        axs[1].get_position().y0 * 0.2,
    ]
    cax = fig.add_axes(cax_pos)

    imshow_kwargs = {} if imshow_kwargs is None else dict(imshow_kwargs)
    tmp = IMSHOW_KWARGS.copy()
    tmp.update(imshow_kwargs)
    imshow_kwargs = tmp
    cmap = colormaps[imshow_kwargs.pop("cmap")]

    for ax, probabilities, positives, title in zip(
        axs,
        (probs1, probs2),
        (positives1, positives2),
        (title1, title2),
    ):
        raster = Raster(probabilities)
        raster.data *= 100.0
        raster.imshow(
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
            if (
                f"lon_{time:0.0f}" not in positives.columns
                or f"lat_{time:0.0f}" not in positives.columns
            ):
                positives = reconstruct_by_topologies(
                    data=positives,
                    plate_reconstruction=gplot.plate_reconstruction,
                    times=np.arange(time, positives["age (Ma)"].round().max() + 1),
                    verbose=False,
                )
            _add_deposits(
                ax=ax,
                deposits=positives,
                time=time,
                **SCATTER_KWARGS,
            )
        # if positives is not None:
        #     _add_deposits(
        #         ax=ax,
        #         deposits=positives,
        #         time=time,
        #         **SCATTER_KWARGS,
        #     )
            # ax.plot(
            #     "lon",
            #     "lat",
            #     data=positives,
            #     **SCATTER_KWARGS,
            # )
        if title is not None:
            ax.set_title(title, fontsize=TITLE_SIZE)
        ax.set_global()

    sm = ScalarMappable(
        norm=Normalize(
            imshow_kwargs.get("vmin", 0),
            imshow_kwargs.get("vmax", 100),
        ),
        cmap=cmap,
    )
    cbar_orientation = "vertical" if nrows == 2 else "horizontal"
    cbar = fig.colorbar(
        sm,
        cax=cax,
        orientation=cbar_orientation,
    )
    if cbar_orientation == "vertical":
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
    else:
        cbar.ax.set_xlim(0, 100)
        cbar.ax.fill_betweenx(
            y=cbar.ax.get_ylim(),
            x1=imshow_kwargs.get("vmax", 100),
            x2=100,
            color=cmap.get_over(),
            zorder=-1,
        )
        cbar.ax.fill_betweenx(
            y=cbar.ax.get_ylim(),
            x1=0,
            x2=imshow_kwargs.get("vmin", 0),
            color=cmap.get_under(),
            zorder=-1,
        )
        cbar.ax.set_xlabel(
            "Deposit probability (%)",
            fontsize=FONT_SIZE,
        )
    cbar.ax.tick_params(labelsize=TICK_SIZE)

    if time is not None:
        fig.suptitle(
            f"{time} Ma",
            fontsize=SUPTITLE_SIZE,
            y=0.94,
        )

    if output_filename is not None:
        fig.savefig(output_filename, **SAVEFIG_KWARGS)
        plt.close(fig)
    return fig


def plot_difference(
    gplot,
    probs1,
    probs2,
    projection=None,
    time=None,
    positives=None,
    output_filename=None,
    central_meridian=None,
    imshow_kwargs=None,
    title=None,
):
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
        # if time is not None:
        #     # Restrict to +/- 2.5 Myr
        #     positives = positives[
        #         (positives["age (Ma)"] - float(time)).abs()
        #         <= 2.5
        #     ]

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

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(
        [0.1, 0.1, 0.8, 0.8],
        projection=projection,
        facecolor=BACKGROUND_COLOUR,
        label="map",
    )
    cax = fig.add_axes([0.94, 0.1, 0.03, 0.8], label="cbar")

    imshow_kwargs = {} if imshow_kwargs is None else dict(imshow_kwargs)
    tmp = IMSHOW_KWARGS.copy()
    tmp.update(imshow_kwargs)
    imshow_kwargs = tmp
    cmap = colormaps[imshow_kwargs.pop("cmap")]

    if not isinstance(probs1, Raster):
        probs1 = Raster(probs1)
    if not isinstance(probs2, Raster):
        probs2 = Raster(probs2)
    raster = (probs2 - probs1.data) * 100.0
    raster.imshow(
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
        if (
            f"lon_{time:0.0f}" not in positives.columns
            or f"lat_{time:0.0f}" not in positives.columns
        ):
            positives = reconstruct_by_topologies(
                data=positives,
                plate_reconstruction=gplot.plate_reconstruction,
                times=np.arange(time, positives["age (Ma)"].round().max() + 1),
                verbose=False,
            )
        _add_deposits(
            ax=ax,
            deposits=positives,
            time=time,
            **SCATTER_KWARGS,
        )
    # if positives is not None:
    #     _add_deposits(
    #         ax=ax,
    #         deposits=positives,
    #         time=time,
    #         **SCATTER_KWARGS,
    #     )
        # ax.plot(
        #     "lon",
        #     "lat",
        #     data=positives,
        #     **SCATTER_KWARGS,
        # )
    if title is not None:
        ax.set_title(title, fontsize=TITLE_SIZE)
    ax.set_global()

    sm = ScalarMappable(
        norm=Normalize(
            imshow_kwargs.get("vmin", 0),
            imshow_kwargs.get("vmax", 100),
        ),
        cmap=cmap,
    )
    cbar_orientation = "vertical"
    cbar = fig.colorbar(
        sm,
        cax=cax,
        orientation=cbar_orientation,
    )
    # cbar.ax.set_ylim(0, 100)
    # cbar.ax.fill_between(
    #     x=cbar.ax.get_xlim(),
    #     y1=imshow_kwargs.get("vmax", 100),
    #     y2=100,
    #     color=cmap.get_over(),
    #     zorder=-1,
    # )
    # cbar.ax.fill_between(
    #     x=cbar.ax.get_xlim(),
    #     y1=0,
    #     y2=imshow_kwargs.get("vmin", 0),
    #     color=cmap.get_under(),
    #     zorder=-1,
    # )
    cbar.ax.set_title(
        "Deposit probability\ndifference (%)",
        fontsize=FONT_SIZE,
        y=1.03,
    )
    cbar.ax.tick_params(labelsize=TICK_SIZE)

    if title is None and time is not None:
        title = f"{time:0.0f} Ma"
    if title is not None:
        ax.set_title(title, fontsize=TITLE_SIZE, y=1.04)

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


def _add_deposits(
    ax: Axes,
    deposits: Union[str, pd.DataFrame],
    time: Optional[float] = None,
    window_size: float = 2.5,
    **kwargs
):
    if not isinstance(deposits, pd.DataFrame):
        deposits = pd.read_csv(deposits)
    if "label" in deposits.columns:
        deposits = deposits[deposits["label"] == "positive"]

    if time is not None and "age (Ma)" in deposits.columns:
        if (
            f"lon_{time:0.0f}" in deposits.columns
            and f"lat_{time:0.0f}" in deposits.columns
        ):
            alpha = kwargs.pop("alpha", 1.0)
            markersize = kwargs.pop("markersize", 20.0)
            zorder = kwargs.pop("zorder", 1)
            oldalpha = kwargs.pop("oldalpha", alpha * 0.25)
            oldmarkersize = kwargs.pop("oldmarkersize", markersize * 0.5)
            oldzorder = kwargs.pop("oldzorder", zorder - 0.01)

            new_deposits = deposits[
                (deposits["age (Ma)"] >= time)
                & ((deposits["age (Ma)"] - time) <= window_size)
            ]
            old_deposits = deposits[
                deposits["age (Ma)"] > (time + window_size)
            ]
            out = []
            out.extend(
                ax.plot(
                    new_deposits[f"lon_{time:0.0f}"],
                    new_deposits[f"lat_{time:0.0f}"],
                    alpha=alpha,
                    markersize=markersize,
                    zorder=zorder,
                    **kwargs
                )
            )
            out.extend(
                ax.plot(
                    old_deposits[f"lon_{time:0.0f}"],
                    old_deposits[f"lat_{time:0.0f}"],
                    alpha=oldalpha,
                    markersize=oldmarkersize,
                    zorder=oldzorder,
                    **kwargs
                )
            )
            return out

        # Reconstructed coordinates do not exist, but time is not None
        deposits = deposits[
            (deposits["age (Ma)"] - time).abs() <= window_size
        ]

    return ax.plot(
        deposits["lon"],
        deposits["lat"],
        **kwargs,
    )


def _prepare_axes(
    ax: Axes,
    gplot: PlotTopologies,
    raster: Raster,
    time: float,
    positives: Optional[pd.DataFrame] = None,
    positives_window_size: float = 2.5,
    central_meridian: float = 0.0,
    imshow_kwargs: Optional[Mapping[str, Any]] = None,
    coastlines_kwargs: Optional[Mapping[str, Any]] = None,
    topologies_kwargs: Optional[Mapping[str, Any]] = None,
    ridges_kwargs: Optional[Mapping[str, Any]] = None,
    teeth_kwargs: Optional[Mapping[str, Any]] = None,
    scatter_kwargs: Optional[Mapping[str, Any]] = None,
    **kwargs
):
    imshow_kwargs = {} if imshow_kwargs is None else imshow_kwargs
    imshow_kwargs = _copy_update_dict(IMSHOW_KWARGS, imshow_kwargs)

    coastlines_kwargs = {} if coastlines_kwargs is None else coastlines_kwargs
    coastlines_kwargs = _copy_update_dict(COASTLINES_KWARGS, coastlines_kwargs)

    topologies_kwargs = {} if topologies_kwargs is None else topologies_kwargs
    topologies_kwargs = _copy_update_dict(TOPOLOGIES_KWARGS, topologies_kwargs)

    ridges_kwargs = {} if ridges_kwargs is None else ridges_kwargs
    ridges_kwargs = _copy_update_dict(RIDGES_KWARGS, ridges_kwargs)

    teeth_kwargs = {} if teeth_kwargs is None else teeth_kwargs
    teeth_kwargs = _copy_update_dict(TEETH_KWARGS, teeth_kwargs)

    scatter_kwargs = {} if scatter_kwargs is None else scatter_kwargs
    scatter_kwargs = _copy_update_dict(SCATTER_KWARGS, scatter_kwargs)

    raster.imshow(
        ax=ax,
        **imshow_kwargs,
    )
    gplot.plot_coastlines(
        ax=ax,
        central_meridian=central_meridian,
        **coastlines_kwargs,
    )
    gplot.plot_all_topological_sections(
        ax=ax,
        central_meridian=central_meridian,
        **topologies_kwargs,
    )
    gplot.plot_ridges_and_transforms(
        ax=ax,
        central_meridian=central_meridian,
        **ridges_kwargs,
    )
    gplot.plot_subduction_teeth(ax=ax, **teeth_kwargs)

    if (
        positives is not None
        and isinstance(positives, pd.DataFrame)
        and positives.shape[0] > 0
    ):
        if (
            f"lon_{time:0.0f}" not in positives.columns
            or f"lat_{time:0.0f}" not in positives.columns
        ):
            positives = reconstruct_by_topologies(
                data=positives,
                plate_reconstruction=gplot.plate_reconstruction,
                times=np.arange(time, positives["age (Ma)"].round().max() + 1),
                verbose=False,
            )
        _add_deposits(
            ax=ax,
            deposits=positives,
            time=time,
            window_size=positives_window_size,
            **scatter_kwargs,
        )
    ax.set_global()

    if time is not None:
        ax.set_title(
            f"{time} Ma",
            fontsize=kwargs.get("fontsize", TITLE_SIZE),
            y=1.04,
        )

    sm = ScalarMappable(
        norm=Normalize(
            imshow_kwargs["vmin"],
            imshow_kwargs["vmax"],
        ),
        cmap=imshow_kwargs["cmap"],
    )

    return sm


def _copy_update_dict(old: dict, new: dict):
    tmp = dict(old).copy()
    tmp.update(dict(new))
    return tmp
