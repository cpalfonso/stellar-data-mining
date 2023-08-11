from itertools import (
    combinations,
    product,
)
from sys import stderr

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from scipy.stats import kendalltau
from sklearn.base import BaseEstimator

from .format_feature_names import format_feature_name


def calculate_importances(clf, names):
    if not isinstance(clf, BaseEstimator):
        clf = load(clf)
    if hasattr(clf, "feature_importances_"):
        importances = np.array(clf.feature_importances_)
    elif hasattr(clf, "estimators_"):
        importances = np.array(
            [
                i.feature_importances_
                for i in clf.estimators_
            ]
        ).mean(axis=0)
    else:
        raise TypeError(
            "Could not extract feature importances from "
            f"type {type(clf)}"
        )

    names = np.array(names, dtype="object")
    return pd.Series(importances, index=names)


def get_ranks(importances, zero_index=False):
    ordered = np.array(importances.sort_values(ascending=False).index)
    ranks = pd.Series(np.arange(np.size(ordered)), index=ordered)
    if not zero_index:
        ranks += 1
    return ranks


def plot_importances(
    clf,
    names,
    normalise=True,
    num_to_keep=6,
    ax=None,
    title=None,
    **kwargs
):
    figsize = kwargs.pop("figsize", (10, num_to_keep * 0.5))
    font_size = kwargs.pop("fontsize", num_to_keep * 2)
    xlabel_size = kwargs.pop("xlabel_size", font_size * 1.25)
    title_size = kwargs.pop("titlesize", font_size * 1.35)
    facecolor = kwargs.pop("facecolor", "lightgrey")
    edgecolor = kwargs.pop("edgecolor", "black")
    height = kwargs.pop("height", 1.0)

    importances = calculate_importances(clf, names)
    importances = importances.sort_values(ascending=False)
    if normalise:
        importances = importances / importances.max()
    if num_to_keep is None:
        num_to_keep = np.size(importances)

    ylocs = -1 * np.arange(num_to_keep)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    ax.barh(
        y=ylocs,
        width=importances.iloc[:num_to_keep],
        height=height,
        facecolor=facecolor,
        edgecolor=edgecolor,
        zorder=1,
        **kwargs,
    )
    ax.yaxis.set_ticks(ylocs)
    ax.yaxis.set_ticklabels(
        [
            format_feature_name(i)
            for i in importances.iloc[:num_to_keep].index
        ]
    )
    ax.tick_params(labelsize=font_size)
    ax.grid(linestyle="dashed", color="grey")
    ax.set_axisbelow(True)

    xlabel = "Gini importance"
    if normalise:
        xlabel = "Relative " + xlabel
    ax.set_xlabel(xlabel, fontsize=xlabel_size)
    if title is None:
        title = "Feature importances"
        if num_to_keep is not None:
            title += f" (top {num_to_keep} features)"
    ax.set_title(title, fontsize=title_size)

    return fig, ax


def plot_correlations(
    clfs,
    names,
    verbose=False,
    title=None,
    kendalltau_kw=None,
    text_kw=None,
    **kwargs
):
    if kendalltau_kw is None:
        kendalltau_kw = {}
    alternative = kendalltau_kw.pop("alternative", "greater")

    if text_kw is None:
        text_kw = {}
    color = text_kw.pop("color", "black")
    ha = text_kw.pop("ha", "center")
    va = text_kw.pop("va", "center")

    cmap = kwargs.pop("cmap", "plasma")
    interpolation = kwargs.pop("interpolation", "none")
    vmin = kwargs.pop("vmin", -1)
    vmax = kwargs.pop("vmax", 1)

    importances = {
        region: calculate_importances(model, names)
        for region, model in clfs.items()
    }
    ranks = {
        region: get_ranks(values)
        for region, values in importances.items()
    }
    ranks = pd.concat(
        [
            pd.DataFrame(rank_values, columns=[region])
            for region, rank_values in ranks.items()
        ],
        axis="columns",
    )

    columns = np.array(ranks.columns)
    n = len(columns)
    vals = np.empty((n, n))
    pvals = np.empty((n, n))
    for i, j in product(range(n), repeat=2):
        cola = columns[i]
        colb = columns[j]
        result = kendalltau(
            ranks[cola],
            ranks[colb],
            alternative=alternative,
            **kendalltau_kw,
        )
        vals[i, j] = result.statistic
        pvals[i, j] = result.pvalue

    if verbose:
        for i, j in combinations(range(n), 2):
            cola = columns[i]
            colb = columns[j]
            val = vals[i, j]
            pval = pvals[i, j]
            print(
                f"{cola}/{colb}: "
                + f"tau = {val:0.2f}, p = {pval:0.2f}",
                file=stderr,
            )

    fontsize = text_kw.pop("fontsize", n * 4)
    figsize = (n * 1.5, n * 1.5)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    cax = fig.add_axes([0.1, 0.05, 0.8, 0.03])
    im = ax.matshow(
        vals,
        interpolation=interpolation,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        **kwargs,
    )

    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.ax.set_xlabel(r"Kendall's $\tau$", fontsize=fontsize)
    cbar.ax.set_xticks(np.arange(-1, 1.5, 0.5))
    cbar.ax.tick_params(labelsize=fontsize)

    for i, j in product(range(n), repeat=2):
        text = (
            r"$\tau = "
            + f"{vals[i, j]:0.2f}"
            + "$"
            + "\n"
            + r"($p = "
            + f"{pvals[i, j]:0.2f}"
            + r"$)"
        )
        ax.text(
            j, i,
            text,
            color=color,
            ha=ha,
            va=va,
            fontsize=fontsize,
        )
    ax.set_xticks(range(n))
    ax.set_xticklabels(columns)
    ax.set_yticks(range(n))
    ax.set_yticklabels(columns)
    ax.tick_params(labelsize=fontsize, bottom=False)

    if title is None:
        title = (
            r"Correlation (Kendall's $\tau$)"
            + "\nof feature importance rankings"
        )
    fig.suptitle(
        title,
        fontsize=fontsize * 1.25,
        y=1.1,
    )

    return fig
