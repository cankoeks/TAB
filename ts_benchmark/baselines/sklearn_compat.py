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
from sklearn.decomposition import PCA
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


def orient_components_u_based(pca, X):
    """
    Orient fitted PCA components with scikit-learn 0.24's sign convention.

    scikit-learn<1.5 used ``svd_flip(U, Vt)``: the largest absolute entry of each
    column of U is positive. 1.5 switched to ``u_based_decision=False``, which can
    flip components and with them every projection onto them.

    :param pca: A fitted sklearn PCA.
    :param X: The data the PCA was fitted on.
    :return: The same PCA, with re-oriented ``components_``.
    """
    projections = (np.asarray(X, dtype=float) - pca.mean_) @ pca.components_.T
    max_abs_rows = np.argmax(np.abs(projections), axis=0)
    signs = np.sign(projections[max_abs_rows, range(projections.shape[1])])
    signs[signs == 0] = 1
    pca.components_ *= signs[:, np.newaxis]
    return pca


def fit_pca_sklearn024(X, n_components, random_state=None):
    """
    Fit ``PCA(n_components)`` as scikit-learn 0.24 did with ``svd_solver='auto'``.

    scikit-learn 1.5 changed the 'auto' solver policy (it now prefers
    'covariance_eigh' for tall data) and the sign convention. This picks the solver
    0.24 would have used for this data ('full' or 'randomized') and restores the
    old component orientation.

    :param X: Data of shape (n_samples, n_features).
    :param n_components: Number of components (int).
    :param random_state: Seed, RandomState or None (the global numpy state).
    :return: A fitted PCA.
    """
    X = np.asarray(X)
    if max(X.shape) <= 500:
        solver = "full"
    elif 1 <= n_components < 0.8 * min(X.shape):
        solver = "randomized"
    else:
        solver = "full"
    pca = PCA(n_components=n_components, svd_solver=solver, random_state=random_state)
    pca.fit(X)
    return orient_components_u_based(pca, X)
