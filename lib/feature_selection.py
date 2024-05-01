import os
import warnings
from collections import defaultdict, namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
# from scipy.optimize import differential_evolution
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


def correlation_dendrogram(
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

    # Remove constant columns
    X = X.loc[:, X.std() != 0.0]

    corr = spearmanr(X).correlation
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)

    orientation = dendro_kwargs.get("orientation", "top")
    if show_plot:
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


def dendrogram_from_model(model, **kwargs):
    # https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#plot-hierarchical-clustering-dendrogram
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    labels = kwargs.pop("labels", "auto")
    if labels == "auto" and hasattr(model, "feature_names_in_"):
        labels = [
            format_feature_name(i)
            for i in model.feature_names_in_
        ]
    no_plot = kwargs.pop("no_plot", False)
    ax = kwargs.pop("ax", None)

    # Plot the corresponding dendrogram
    dendro = hierarchy.dendrogram(
        linkage_matrix,
        labels=labels,
        no_plot=no_plot,
        ax=ax,
        **kwargs,
    )
    return DendroResult(
        dendrogram=dendro,
        distances=model.distances_,
        figure=None if no_plot else plt.gcf(),
    )


def distance_threshold_from_model(model):
    if model.distance_threshold is not None:
        return model.distance_threshold

    linkage_matrix = linkage_from_model(model)

    c = np.array(
        [
            _n_clusters_from_dist_linkage(
                i,
                linkage_matrix,
            ) for i in model.distances_
        ]
    )
    return np.min(model.distances_[c <= model.n_clusters_])


def linkage_from_model(model):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    return linkage_matrix


def n_clusters_from_dist_model(dist, model):
    linkage_matrix = linkage_from_model(model)
    return _n_clusters_from_dist_linkage(dist, linkage_matrix)


def _n_clusters_from_dist_linkage(dist, linkage):
    if np.size(dist) != 1:
        raise ValueError(f"Invalid size: {np.size(dist)}")
    dist = np.ravel(dist)[0]
    clusters = hierarchy.fcluster(linkage, dist, criterion="distance")
    n = np.unique(clusters).size
    return n


def clusters_from_model(model):
    return {
        i: features_from_model_cluster(model, i)
        for i in np.unique(model.labels_)
    }


def features_from_model_cluster(model, cluster):
    labels = model.labels_
    feature_names = model.feature_names_in_
    return list(feature_names[labels == int(cluster)])
