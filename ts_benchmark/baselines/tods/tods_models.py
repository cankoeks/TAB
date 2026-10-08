# -*- coding: utf-8 -*-
"""
TODS baselines implemented directly on top of pyod.

The original implementation wrapped these detectors through the TODS / d3m
primitive stack (kept unchanged in ``third_party``), which only installs on
Python 3.8. Every TODS detector used by the benchmark is a thin wrapper around
a pyod model (or the d3m-free sliding-window PCA in ``pyod_core``): with the
TODS hyperparameters the d3m layer only converts the input array to a d3m
DataFrame and back. The adapters below therefore call the same estimators
directly, with the same effective constructor arguments, and keep the model
names and output conventions, so ``--model-name "tods.hbosski"`` keeps working.

``autoencoderski`` and ``lstmodetectorski`` are not available here: they relied
on the TensorFlow/Keras-based detectors of the d3m stack. Use the legacy
Python 3.8 environment (requirements-legacy-py38.txt) to run them.
"""
import logging

import numpy as np
import pandas as pd
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM

from ts_benchmark.baselines.sklearn_compat import kmeans_sklearn024
from ts_benchmark.baselines.tods.pyod_core.PCA import PCA as WindowPCA

logger = logging.getLogger(__name__)


# Effective constructor arguments of the estimators built by the original d3m
# primitives when no hyperparameters are given (TAB's scripts always pass '{}').
# Dumped in the legacy Python 3.8 environment as ``primitive._clf.get_params()``
# of each primitive built with ``Hyperparams.defaults()``. Several of
# them differ from current pyod defaults (e.g. HBOS tol=0.1, PCA whiten=True).
_LEGACY_DEFAULTS = {
    "IsolationForestSKI": {
        "behaviour": "new",
        "bootstrap": False,
        "contamination": 0.1,
        "max_features": 1.0,
        "max_samples": "auto",
        "n_estimators": 100,
        "n_jobs": 1,
        "random_state": None,
        "verbose": 0,
    },
    "KNNSKI": {
        "algorithm": "auto",
        "contamination": 0.1,
        "leaf_size": 30,
        "method": "largest",
        "metric": "minkowski",
        "metric_params": None,
        "n_jobs": 1,
        "n_neighbors": 5,
        "p": 2,
        "radius": 1.0,
    },
    "LOFSKI": {
        "algorithm": "auto",
        "contamination": 0.1,
        "leaf_size": 30,
        "metric": "minkowski",
        "metric_params": None,
        "n_jobs": 1,
        "n_neighbors": 20,
        "novelty": True,
        "p": 2,
    },
    "OCSVMSKI": {
        "cache_size": 200,
        "coef0": 0.0,
        "contamination": 0.1,
        "degree": 3,
        "gamma": "auto",
        "kernel": "rbf",
        "max_iter": -1,
        "nu": 0.5,
        "shrinking": True,
        "tol": 0.001,
        "verbose": False,
    },
    "HBOSSKI": {
        "alpha": 0.1,
        "contamination": 0.1,
        "n_bins": 10,
        "tol": 0.1,
    },
    "LODASKI": {
        "contamination": 0.1,
        "n_bins": 10,
        "n_random_cuts": 100,
    },
    "PCAODetectorSKI": {
        "contamination": 0.1,
        "copy": True,
        "iterated_power": "auto",
        "n_components": None,
        "n_selected_components": None,
        "random_state": None,
        "standardization": True,
        "step_size": 1,
        "svd_solver": "auto",
        "tol": 0.0,
        "weighted": True,
        "whiten": True,
        "window_size": 10,
    },
    "COFSKI": {
        "contamination": 0.1,
        "method": "fast",
        "n_neighbors": 5,
    },
    "CBLOFSKI": {
        "alpha": 0.9,
        "beta": 5,
        "check_estimator": False,
        "contamination": 0.1,
        "n_clusters": 8,
        "n_jobs": None,
        "random_state": None,
        "use_weights": False,
    },
}


def _cblof_args(args: dict) -> dict:
    """
    Add the clustering estimator CBLOF used in the original environment.

    pyod builds ``KMeans(n_clusters, random_state)`` with scikit-learn's defaults,
    which changed after scikit-learn 0.24 (n_init, algorithm, k-means++ sampling)
    and give a different clustering for the same seed. Pass an estimator with the
    old behaviour explicitly.
    """
    if args.get("clustering_estimator") is None:
        args = dict(args)
        args["clustering_estimator"] = kmeans_sklearn024(
            n_clusters=args["n_clusters"], random_state=args["random_state"]
        )
    return args


# [exported name (kept from the original TODS wrappers), model class, required params]
TODS_MODELS = [
    ["IsolationForestSKI", IForest, {}],
    ["KNNSKI", KNN, {}],
    ["LOFSKI", LOF, {}],
    ["OCSVMSKI", OCSVM, {}],
    ["HBOSSKI", HBOS, {}],
    ["LODASKI", LODA, {}],
    ["PCAODetectorSKI", WindowPCA, {}],
    ["COFSKI", COF, {}],
    ["CBLOFSKI", CBLOF, {}],
]


class TodsModelAdapter:
    """
    The Tods model adapter class is used to adapt models in the Tods framework to meet the requirements of prediction strategies.
    """

    def __init__(
        self,
        model_name: str,
        model_class: object,
        model_args: dict,
    ):
        """
        Initialize the Tods model adapter object.

        :param model_name: Model name.
        :param model_class: Tods model class.
        :param model_args: Model initialization parameters.
        """
        self.model = None
        self.model_class = model_class
        self.model_args = model_args
        self.model_name = model_name

    def build_model(self) -> object:
        """
        Build the underlying estimator with the historical TODS defaults,
        overridden by the user-supplied hyperparameters.
        """
        args = {**_LEGACY_DEFAULTS[self.model_name], **self.model_args}
        if self.model_name == "CBLOFSKI":
            args = _cblof_args(args)
        return self.model_class(**args)

    def detect_fit(self, series: pd.DataFrame, label: pd.DataFrame) -> object:
        """
        Fit a suitable Tods model on time series data.

        :param series: Time series data.
        :param label: Label data.
        :return: The fitted model object.
        """

        self.model = self.build_model()
        X = series.values
        self.model.fit(X)

        return self.model

    def detect_score(self, train: pd.DataFrame) -> np.ndarray:
        """
        Calculate anomaly scores using an adapted Tods model.

        :param train: Training data used to calculate scores.
        :return: Anomaly score array.
        """
        X = train.values
        prediction_score = self.model.decision_function(X)
        if isinstance(prediction_score, tuple):
            # collective (window-based) detectors return (scores, left_inds, right_inds)
            prediction_score = prediction_score[0]
        prediction_score = np.asarray(prediction_score).reshape(-1)

        return prediction_score, prediction_score

    def detect_label(self, train: pd.DataFrame) -> np.ndarray:
        """
        Use an adapted Tods model for anomaly detection and generate labels.

        :param train: Training data used for anomaly detection.
        :return: Anomaly label array.
        """
        X = train.values
        prediction_labels = self.model.predict(X)
        if isinstance(prediction_labels, tuple):
            # collective (window-based) detectors return (labels, left_inds, right_inds)
            prediction_labels = prediction_labels[0]
        prediction_labels = np.asarray(prediction_labels).reshape(-1)

        return prediction_labels, prediction_labels

    def __repr__(self):
        """
        Returns a string representation of the model name.
        """
        return self.model_name


def generate_model_factory(
    model_name: str,
    model_class: object,
    required_args: dict,
) -> object:
    """
    Generate model factory information for creating Tods model adapters.

    :param model_name: Model name.
    :param model_class: Tods model class.
    :param required_args: Required parameters for model initialization.
    :return: A dictionary containing the model factory and required parameters.
    """

    def model_factory(**kwargs) -> object:
        """
        Model factory, used to create Tods model adapter objects.
        :param kwargs: Model initialization parameters.
        :return: Tods model adapter object.
        """
        return TodsModelAdapter(
            model_name,
            model_class,
            kwargs,
        )

    return {"model_factory": model_factory, "required_hyper_params": required_args}


# Generate model factories for each model class and required parameters in TODS-MODELS and add them to global variables
for model_name, model_class, required_args in TODS_MODELS:
    globals()[model_name.lower()] = generate_model_factory(
        model_name, model_class, required_args
    )


def _generate_unavailable_factory(model_name: str) -> dict:
    def model_factory(**kwargs) -> object:
        raise NotImplementedError(
            f"tods.{model_name.lower()} is not available on Python 3.12: it relied on "
            "the TensorFlow/Keras detectors of the d3m stack. Run it in the legacy "
            "Python 3.8 environment (requirements-legacy-py38.txt) instead."
        )

    return {"model_factory": model_factory, "required_hyper_params": {}}


autoencoderski = _generate_unavailable_factory("AutoEncoderSKI")
lstmodetectorski = _generate_unavailable_factory("LSTMODetectorSKI")
