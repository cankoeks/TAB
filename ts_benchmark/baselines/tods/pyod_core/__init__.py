# -*- coding: utf-8 -*-
# Vendored from TODS (https://github.com/datamllab/tods, Apache License 2.0),
# tods/detection_algorithm/core, via ts_benchmark/baselines/tods/third_party.
# Kept d3m-free so the sliding-window PCA detector runs without the d3m stack.
# Changes: np.float -> float (removed in numpy 1.24) and an attribute-based
# check_is_fitted (scikit-learn>=1.6 rejects non-estimators); PCA.py also restores
# the scikit-learn<1.5 orientation of the principal components.
