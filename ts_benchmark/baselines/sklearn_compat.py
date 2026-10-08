# -*- coding: utf-8 -*-
"""
Reproduce scikit-learn 0.24 defaults that TAB's baselines relied on.

TAB's published results were produced with scikit-learn 0.24 (pinned by the d3m
stack). Newer scikit-learn releases changed some defaults in ways that alter
the results of seeded models even with the same random seed. The helpers here
restore the old behaviour explicitly.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.extmath import row_norms


def kmeans_plusplus_sklearn024(X, n_clusters, random_state):
    """
    k-means++ initialisation exactly as in scikit-learn 0.24 (``_kmeans_plusplus``).

    scikit-learn>=1.3 draws the first center with ``random_state.choice`` (to
    support sample weights) instead of ``random_state.randint``, which yields a
    different initialisation for the same random state.

    :param X: Data, already centered by KMeans.fit.
    :param n_clusters: Number of centers.
    :param random_state: A numpy RandomState.
    :return: Initial centers of shape (n_clusters, n_features).
    """
    n_samples, n_features = X.shape
    x_squared_norms = row_norms(X, squared=True)
    centers = np.empty((n_clusters, n_features), dtype=X.dtype)
    n_local_trials = 2 + int(np.log(n_clusters))

    center_id = random_state.randint(n_samples)
    centers[0] = X[center_id]

    closest_dist_sq = euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms, squared=True
    )
    current_pot = closest_dist_sq.sum()

    for c in range(1, n_clusters):
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(
            np.cumsum(closest_dist_sq, dtype=np.float64), rand_vals
        )
        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

        distance_to_candidates = euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True
        )
        np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        centers[c] = X[best_candidate]

    return centers


def kmeans_sklearn024(n_clusters, random_state=None, **kwargs):
    """
    A KMeans estimator that behaves like scikit-learn 0.24's ``KMeans(n_clusters)``.

    Restores the old defaults ``n_init=10`` (now 'auto', i.e. a single run),
    ``algorithm='elkan'`` for more than one cluster (the old 'auto') and the old
    k-means++ initialisation. The 10 runs share one random state, as before.

    :param n_clusters: Number of clusters.
    :param random_state: Seed, RandomState or None (the global numpy state).
    :param kwargs: Further KMeans parameters.
    :return: An unfitted KMeans estimator.
    """
    params = {
        "init": kmeans_plusplus_sklearn024,
        "n_init": 10,
        "algorithm": "lloyd" if n_clusters == 1 else "elkan",
    }
    params.update(kwargs)
    return KMeans(n_clusters=n_clusters, random_state=random_state, **params)
