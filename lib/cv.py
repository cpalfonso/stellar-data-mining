import time
from sys import stderr

import numpy as np
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
    verbose=False,
    get_xy_kw=None,
    return_models=False,
    thresh_method=None,
):
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

    output = {
        i: np.zeros(n_splits, dtype=np.float_)
        for i in _METRICS.keys()
    }
    output["n_train"] = np.zeros(n_splits, dtype=np.int_)
    output["time_fit"] = np.zeros(n_splits, dtype=np.float_)
    output["n_test"] = np.zeros(n_splits, dtype=np.int_)
    output["time_predict"] = np.zeros(n_splits, dtype=np.float_)
    if return_models:
        output["model"] = np.empty(n_splits, dtype="object")
    if auto_thresh:
        output["prob_thresh"] = np.zeros(n_splits, dtype=np.float_)
    split = cv.split(data.drop(columns="label"), data["label"])
    for i, (train_idx, test_idx) in enumerate(split):
        if verbose:
            print(
                f"Current fold: {i + 1} of {n_splits}",
                file=stderr,
            )
        train_data = data.loc[train_idx, :]
        if pu:
            labels = {"positive", "unlabeled", "unlabelled"}
        else:
            labels = {"positive", "negative"}
        train_data = train_data[train_data["label"].isin(labels)]
        x_train, y_train = get_xy(train_data, **get_xy_kw)
        output["n_train"][i] = np.size(y_train)

        test_data = data.loc[test_idx, :]
        test_data = test_data[
            test_data["label"].isin({"positive", "negative"})
        ]
        x_test, y_test = get_xy(test_data, **get_xy_kw)
        output["n_test"][i] = np.size(y_test)

        model = clone(clf)
        t0 = time.time()
        model.fit(x_train, y_train)
        t1 = time.time()
        probs = model.predict_proba(x_test)[:, 1]
        t2 = time.time()
        output["time_fit"][i] = t1 - t0
        output["time_predict"][i] = t2 - t1

        if auto_thresh:
            test_thresholds = np.linspace(0, 1, 100)
            score_func = _METRICS[thresh_method]
            scores = np.zeros_like(test_thresholds)
            for thresh_i, test_threshold in enumerate(test_thresholds):
                scores[thresh_i] = score_func(y_test, probs >= test_threshold)
            thresh = np.median(test_thresholds[scores == np.nanmax(scores)])
            output["prob_thresh"][i] = thresh

        preds = (probs >= thresh).astype(np.int_)

        for metric, function in _METRICS.items():
            if metric in _PROB_METRICS:
                value = function(y_test, probs)
            else:
                value = function(y_test, preds)
            output[metric][i] = value
        if return_models:
            output["model"][i] = model

    return output
