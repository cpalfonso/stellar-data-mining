import os
import warnings
from collections import defaultdict, namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from .misc import format_feature_name

DendroResult = namedtuple(
    "DendroResult",
    ["dendrogram", "distances", "figure"],
)


def create_dendrogram(
    X,
    show_plot=True,
    distance_threshold=None,
    dendro_kwargs=None,
    thresh_kwargs=None,
    subplots_kwargs=None,
    tick_kwargs=None,
    label_kwargs=None,
):
    thresh_kwargs = {} if thresh_kwargs is None else dict(thresh_kwargs)
    subplots_kwargs = {} if subplots_kwargs is None else dict(subplots_kwargs)
    dendro_kwargs = {} if dendro_kwargs is None else dict(dendro_kwargs)
    tick_kwargs = {} if tick_kwargs is None else dict(tick_kwargs)
    label_kwargs = {} if label_kwargs is None else dict(label_kwargs)

    corr = spearmanr(X).correlation
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)

    orientation = dendro_kwargs.get("orientation", "top")
    if show_plot:
        # shape = (1, 2) if orientation in {"top", "bottom"} else (2, 1)
        # figsize = subplots_kw.pop(
        #     "figsize",
        #     (15, 8) if orientation in {"top", "bottom"} else (8, 15)
        # )
        # fig, (ax1, ax2) = plt.subplots(
        #     *shape,
        #     figsize=figsize,
        # )
        figsize = subplots_kwargs.pop(
            "figsize",
            (7, 9) if orientation in {"top", "bottom"} else (9, 7)
        )
        fig, ax1 = plt.subplots(figsize=figsize, **subplots_kwargs)
    else:
        fig = None

    # We convert the correlation matrix to a distance matrix before performing
    # hierarchical clustering using Ward's linkage.
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))
    dendro = hierarchy.dendrogram(
        dist_linkage,
        labels=X.columns.to_list(),
        ax=ax1 if show_plot else None,
        no_plot=not show_plot,
        leaf_rotation=90 if orientation in {"top", "bottom"} else 0,
        **dendro_kwargs
    )
    # dendro_idx = np.arange(0, len(dendro["ivl"]))

    if show_plot:
        feature_names_formatted = [
            format_feature_name(i) for i in dendro["ivl"]
        ]
        tick_size = tick_kwargs.pop("labelsize", 10)
        label_size = label_kwargs.pop("fontsize", 12)
        if orientation in {"top", "bottom"}:
            ax1.set_xticks(ax1.get_xticks(), feature_names_formatted)
            ax1.set_ylabel("Distance", fontsize=label_size, **label_kwargs)
        else:
            ax1.set_yticks(ax1.get_yticks(), feature_names_formatted)
            ax1.set_xlabel("Distance", fontsize=label_size, **label_kwargs)
        ax1.tick_params(labelsize=tick_size, **tick_kwargs)
        if distance_threshold is not None:
            if orientation in {"top", "bottom"}:
                func = ax1.axhline
            else:
                func = ax1.axvline
            func(
                distance_threshold,
                linestyle=thresh_kwargs.pop("linestyle", "dashed"),
                color=thresh_kwargs.pop("color", "grey"),
                zorder=thresh_kwargs.pop("zorder", 10),
                **thresh_kwargs,
            )

        # ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
        # ax2.set_xticks(dendro_idx)
        # ax2.set_yticks(dendro_idx)
        # ax2.set_xticklabels(feature_names_formatted, rotation="vertical")
        # ax2.set_yticklabels(feature_names_formatted)
        fig.tight_layout()

    return DendroResult(
        dendrogram=dendro,
        distances=dist_linkage,
        figure=fig,
    )


def select_features(Z, t, criterion="distance", names=None):
    cluster_ids = hierarchy.fcluster(Z=Z, t=t, criterion=criterion)
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]

    if names is not None:
        selected_features = names[selected_features]
    return selected_features
