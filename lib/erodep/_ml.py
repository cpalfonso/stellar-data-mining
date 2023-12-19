import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import pandas as pd
import numpy as np
import scipy.stats
from numpy.typing import (
    ArrayLike,
    NDArray,
)
from pulearn.bagging import BaggingPuClassifier
from scipy.optimize import differential_evolution
from scipy.stats import _continuous_distns
from scipy.stats._distn_infrastructure import rv_continuous
from sklearn.base import clone
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    RobustScaler,
)
from sklearn.svm import SVC

from ..misc import (
    _PathOrDataFrame,
    load_data,
)

__all__ = [
    "clean_outliers",
    "fit_distribution",
    "ll",
    "create_classifier",
    "bootstrap_resample",
]

_SVC_KW: Dict[str, Any] = {
    "kernel": "rbf",
    "probability": True,
}
_BAGGINGPUCLASSIFIER_KW: Dict[str, Any] = {
    "n_estimators": 25,
}


class ClampedSVC(SVC):
    def __init__(
        self,
        *,
        x_min=-np.inf,
        x_max=np.inf,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        shrinking=True,
        probability=False,
        tol=1e-3,
        cache_size=200,
        class_weight=None,
        verbose=False,
        max_iter=-1,
        decision_function_shape="ovr",
        break_ties=False,
        random_state=None,
    ):
        super().__init__(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            C=C,
            shrinking=shrinking,
            probability=probability,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state,
        )
        self._x_min = x_min
        self._x_max = x_max


    @property
    def x_min(self):
        return self._x_min

    @property
    def x_max(self):
        return self._x_max


    def predict_proba(self, X):
        probs = super().predict_proba(X)
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        invalid_inds = np.logical_or(
            np.any(X > self.x_max, axis=1),
            np.any(X < self.x_min, axis=1),
        )
        probs[invalid_inds, 0] = 1
        probs[invalid_inds, 1] = 0
        return probs


def clean_outliers(
    data: ArrayLike,
    contamination: Union[float, str] = "auto",
    n_jobs: int = 1,
    **kwargs
) -> NDArray:
    data = np.array(data)
    forest = IsolationForest(
        n_jobs=n_jobs,
        contamination=contamination,
        **kwargs
    )
    if data.ndim == 1:
        x = np.reshape(data, (-1, 1))
    else:
        x = data
    forest_result = forest.fit_predict(x)
    if data.ndim == 1:
        data = data[forest_result == 1]
    else:
        data = data[forest_result == 1, :]
    return data


def fit_distribution(
    data: ArrayLike,
    dist: Union[str, rv_continuous] = "gengamma",
    optimizer: Optional[Callable] = None,
    bounds: Optional[Mapping[str, Tuple[float, float]]] = None,
    guess: Optional[Mapping[str, float]] = None,
    remove_outliers: bool = False,
    maxiter: int = int(5.0e6),
    contamination: Union[str, float] = "auto",
    n_jobs: int = 1,
    clean_outliers_kw: Optional[Mapping[str, Any]] = None,
    failure: Literal["raise", "warn"] = "raise",
    random_state: Optional[int] = None,
    **kwargs
) -> rv_continuous:
    dist = _dist_from_name(dist)
    if failure not in {"raise", "warn"}:
        raise ValueError("failure must be one of {'raise', 'warn'}")

    data = np.array(data)
    if bounds is None:
        if isinstance(dist, _continuous_distns.gengamma_gen):
            bounds = {
                "a": (0, 50),
                "c": (-50, 50),
            }
        elif isinstance(dist, _continuous_distns.skewnorm_gen):
            bounds = {"a": (-50, 50)}
        elif isinstance(dist, _continuous_distns.gamma_gen):
            bounds = {"a": (0, 50)}
        elif isinstance(dist, _continuous_distns.betaprime_gen):
            bounds = {
                "a": (0, 50),
                "b": (0, 50),
            }
        else:
            bounds = {
                "a": (-1000, 1000),
                "b": (-1000, 1000),
                "c": (-20, 20),
                "s": (0, 20),
            }
        bounds.update(
            {
                "loc": (-5000, 5000),
                "scale": (0, 5000),
            }
        )
    if guess is None:
        guess = {
            "loc": np.median(data),
        }

    if optimizer is None:
        def f(*args, **kwargs):
            return differential_evolution(
                *args,
                maxiter=kwargs.pop("maxiter", maxiter),
                seed=kwargs.pop("seed", random_state),
                **kwargs
            )

        optimizer = f

    if remove_outliers:
        if clean_outliers_kw is None:
            clean_outliers_kw = {}
        data = clean_outliers(
            data=data,
            contamination=contamination,
            n_jobs=n_jobs,
            random_state=random_state,
            **clean_outliers_kw
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        params = scipy.stats.fit(
            dist,
            data=data,
            optimizer=optimizer,
            bounds=bounds,
            guess=guess,
            **kwargs
        )
    if not params.success:
        message = f"Optimisation failed! {params.message}"
        if failure == "raise":
            raise RuntimeError(message)
        else:
            warnings.warn(message, RuntimeWarning)
    return dist(*(params.params))


def ll(data: ArrayLike, dist: rv_continuous) -> NDArray:
    data = np.array(data)
    return np.log(dist.pdf(data))


def create_classifier(
    data: _PathOrDataFrame,
    n_jobs: int = 1,
    random_state: Optional[int] = None,
    svc_kw: Optional[Mapping[str, Any]] = None,
    baggingpuclassifier_kw: Optional[Mapping[str, Any]] = None,
    predictor: str = "erosion (m)",
    label: str = "label",
    fit: bool = True,
    verbose: bool = False,
) -> BaggingPuClassifier:
    data = load_data(data, verbose=verbose)

    if svc_kw is None:
        svc_kw = {}
    tmp = _SVC_KW.copy()
    tmp.update(**svc_kw)
    svc_kw = tmp
    svc_kw["random_state"] = svc_kw.pop("random_state", random_state)

    if baggingpuclassifier_kw is None:
        baggingpuclassifier_kw = {}
    tmp = _BAGGINGPUCLASSIFIER_KW.copy()
    tmp.update(**baggingpuclassifier_kw)
    baggingpuclassifier_kw = tmp
    del tmp
    baggingpuclassifier_kw["n_jobs"] = baggingpuclassifier_kw.pop(
        "n_jobs",
        n_jobs,
    )
    baggingpuclassifier_kw["random_state"] = baggingpuclassifier_kw.pop(
        "random_state",
        random_state,
    )

    data = data[data[label].isin({"positive", "unlabelled", "unlabeled"})].copy()
    data[label] = data[label].replace(
        {
            "positive": 1,
            "unlabelled": 0,
            "unlabeled": 0,
        }
    )

    x = np.reshape(data[predictor], (-1, 1))
    y = np.array(data[label])

    scaler = RobustScaler()
    model_base = SVC(**svc_kw)
    pipeline = make_pipeline(scaler, model_base)
    model_pu = BaggingPuClassifier(
        clone(pipeline),
        **baggingpuclassifier_kw,
    )
    if fit:
        model_pu.fit(x, y)
    return model_pu


def bootstrap_resample(
    data: _PathOrDataFrame,
    size=1000,
    random_state: Optional[int] = None,
    stratify_on: Optional[str] = "label",
    reindex=False,
    verbose: bool = False,
) -> pd.DataFrame:
    data = load_data(data, verbose=verbose)
    rng = np.random.default_rng(random_state)
    if stratify_on is None:
        indices = rng.choice(data.index, size=size, replace=True)
        data = data.loc[indices, :]
        if reindex:
            data = data.reset_index(drop=True)
    else:
        to_concat = []
        for _, df_label in data.groupby(stratify_on):
            indices = rng.choice(
                df_label.index,
                size=size,
                replace=True,
            )
            to_concat.append(data.loc[indices, :])
        data = pd.concat(to_concat, ignore_index=reindex)
    return data


def _dist_from_name(dist: Union[str, rv_continuous]) -> rv_continuous:
    if isinstance(dist, rv_continuous):
        return dist
    dist_str = str(dist)
    if dist_str in vars(scipy.stats).keys():
        out = vars(scipy.stats)[dist_str]
        if not isinstance(out, rv_continuous):
            raise ValueError(f"Unrecognised distribution: {dist}")
        return out
    raise ValueError(f"Unrecognised distribution: {dist}")
