import time
from sys import stderr

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from .pu import get_xy

_METRICS = {
    "roc_auc": roc_auc_score,
    "average_precision": average_precision_score,
    "balanced_accuracy": balanced_accuracy_score,
    "accuracy": accuracy_score,
    "f1": f1_score,
}
_PROB_METRICS = {
    "roc_auc",
    "average_precision",
}


def perform_cv(
    clf,
    data,
    cv=None,
    thresh=0.5,
    random_state=None,
    pu=True,
    stratify=None,
    verbose=False,
    get_xy_kw=None,
    return_models=False,
    thresh_method=None,
    label="label",
):
    if not isinstance(data, pd.DataFrame):
        try:
            data = pd.read_csv(data)
        except Exception:
            data = pd.DataFrame(data)

    if cv is None:
        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=random_state,
        )
    if get_xy_kw is None:
        get_xy_kw = {}
    n_splits = cv.get_n_splits()

    if thresh == "auto":
        if thresh_method is None:
            thresh_method = "f1"
        if thresh_method not in {"balanced_accuracy", "accuracy", "f1"}:
            raise ValueError(f"Invalid thresh_method: {thresh_method}")
        auto_thresh = True
    else:
        auto_thresh = False

    regions = ("All", "NAm", "SAm")
    # size = len(regions) * n_splits
    # output = {
    #     i: np.full(size, np.nan, dtype=np.float_)
    #     for i in _METRICS.keys()
    # }
    output = {i: [] for i in _METRICS.keys()}
    output["region"] = []
    for which in ("test", "train"):
        output[f"n_{which}"] = []
    for which in ("fit", "predict"):
        output[f"time_{which}"] = []
    if return_models:
        output["model"] = []
    if auto_thresh:
        output["prob_thresh"] = []

    if stratify is None:
        stratify = data[label]
    elif stratify in data.columns:
        stratify = data[stratify]
    elif np.size(stratify) != data.shape[0]:
        raise ValueError(
            f"Invalid `stratify` parameter: {stratify}"
        )
    split = cv.split(data, stratify)

    if pu:
        labels = {"positive", "unlabeled", "unlabelled"}
    else:
        labels = {"positive", "negative"}

    for i, (train_idx, test_idx) in enumerate(split):
        if verbose:
            print(
                f"Current fold: {i + 1} of {n_splits}",
                file=stderr,
            )
        train_data = data.loc[train_idx, :]
        train_data = train_data[train_data[label].isin(labels)]
        x_train, y_train = get_xy(train_data, **get_xy_kw)
        n_train = np.size(y_train)

        test_data = data.loc[test_idx, :]
        test_data = test_data[
            test_data[label].isin({"positive", "negative"})
        ]
        x_test, y_test = get_xy(test_data, **get_xy_kw)

        model = clone(clf)
        t0 = time.time()
        model.fit(x_train, y_train)
        fit_time = time.time() - t0

        probs = np.full_like(y_test, np.nan, dtype=np.float_)
        region_data = {}
        for region in ("NAm", "SAm"):
            indices_region = np.where(test_data["region"] == region)[0]
            x_region = x_test[indices_region, :]
            t0 = time.time()
            probs_region = model.predict_proba(x_region)[:, 1]
            predict_time = time.time() - t0
            probs[indices_region] = probs_region
            region_data[region] = {
                "indices": indices_region,
                "time": predict_time,
            }
        region_data["All"] = {
            "indices": np.arange(np.size(probs)),
            "time": region_data["NAm"]["time"] + region_data["SAm"]["time"],
        }

        if auto_thresh:
            test_thresholds = np.linspace(0, 1, 100)
            score_func = _METRICS[thresh_method]
            scores = np.zeros_like(test_thresholds)
            for thresh_i, test_threshold in enumerate(test_thresholds):
                scores[thresh_i] = score_func(y_test, probs >= test_threshold)
            thresh = np.median(test_thresholds[scores == np.nanmax(scores)])
        preds = (probs >= thresh).astype(np.int_)

        for region in region_data.keys():
            indices_region = region_data[region]["indices"]
            n_region = np.size(indices_region)
            y_region = y_test[indices_region]
            probs_region = probs[indices_region]
            preds_region = preds[indices_region]

            for metric, function in _METRICS.items():
                arg = probs_region if metric in _PROB_METRICS else preds_region
                value = function(y_region, arg)
                output[metric].append(value)

            output["n_train"].append(n_train)
            output["n_test"].append(n_region)
            output["time_fit"].append(fit_time)
            output["time_predict"].append(region_data[region]["time"])
            output["region"].append(region)
            if return_models:
                output["model"].append(model)
            if auto_thresh:
                output["prob_thresh"].append(thresh)

    for key in tuple(output.keys()):
        output[key] = np.array(output[key])

    # for i, (train_idx, test_idx) in enumerate(split):
    #     if verbose:
    #         print(
    #             f"Current fold: {i + 1} of {n_splits}",
    #             file=stderr,
    #         )
    #     train_data = data.loc[train_idx, :]
    #     train_data = train_data[train_data[label].isin(labels)]
    #     x_train, y_train = get_xy(train_data, **get_xy_kw)
    #     output["n_train"][i] = np.size(y_train)

    #     test_data = data.loc[test_idx, :]
    #     test_data = test_data[
    #         test_data[label].isin({"positive", "negative"})
    #     ]
    #     x_test, y_test = get_xy(test_data, **get_xy_kw)
    #     output["n_test"][i] = np.size(y_test)

    #     model = clone(clf)
    #     t0 = time.time()
    #     model.fit(x_train, y_train)
    #     t1 = time.time()
    #     probs = model.predict_proba(x_test)[:, 1]
    #     t2 = time.time()
    #     output["time_fit"][i] = t1 - t0
    #     output["time_predict"][i] = t2 - t1

    #     if auto_thresh:
    #         test_thresholds = np.linspace(0, 1, 100)
    #         score_func = _METRICS[thresh_method]
    #         scores = np.zeros_like(test_thresholds)
    #         for thresh_i, test_threshold in enumerate(test_thresholds):
    #             scores[thresh_i] = score_func(y_test, probs >= test_threshold)
    #         thresh = np.median(test_thresholds[scores == np.nanmax(scores)])
    #         output["prob_thresh"][i] = thresh

    #     preds = (probs >= thresh).astype(np.int_)

    #     regions_dict = {i: {} for i in regions}
    #     for region in regions:
    #         indices = np.where(test_data["region"] == region)[0]
    #         regions_dict[region]["n"] = np.size(indices)
    #         regions_dict[region]["y_test"] = y_test[indices]
    #         regions_dict[region]["probs"] = probs[indices]
    #         regions_dict[region]["preds"] = preds[indices]
    #     for metric, function in _METRICS.items():
    #         if metric in _PROB_METRICS:
    #             arg = probs
    #             key = "probs"
    #         else:
    #             arg = preds
    #             key = "preds"
    #         value = function(y_test, arg)
    #         output[metric][i] = value
    #         for region in regions:
    #             n_region = regions_dict[region]["n"]
    #             output["n_test_" + region][i] = n_region
    #             if n_region <= 1:
    #                 continue
    #             value_region = function(
    #                 regions_dict[region]["y_test"],
    #                 regions_dict[region][key],
    #             )
    #             output[metric + "_" + region][i] = value_region

    #     if return_models:
    #         output["model"][i] = model

    return output
