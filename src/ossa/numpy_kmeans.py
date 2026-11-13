"""NumPy implementation of Lloyd k-means with k-means++ initialization."""

from __future__ import annotations

import numpy as np


def _kmeans_pp_init(X: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ initialisation, returning K row-vectors (centers)."""
    n = X.shape[0]
    centers = np.empty((K, X.shape[1]), dtype=X.dtype)
    first = int(rng.integers(0, n))
    centers[0] = X[first]
    closest_sq = np.sum((X - centers[0]) ** 2, axis=1)
    for k in range(1, K):
        total = closest_sq.sum()
        if total <= 0:
            idx = int(rng.integers(0, n))
        else:
            probs = closest_sq / total
            idx = int(rng.choice(n, p=probs))
        centers[k] = X[idx]
        new_d = np.sum((X - centers[k]) ** 2, axis=1)
        closest_sq = np.minimum(closest_sq, new_d)
    return centers


def _lloyd(
    X: np.ndarray, centers: np.ndarray, max_iter: int, tol: float
) -> tuple[np.ndarray, np.ndarray, float]:
    K = centers.shape[0]
    for _ in range(max_iter):
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)  # (n, K)
        labels = np.argmin(d2, axis=1)
        new_centers = np.empty_like(centers)
        for k in range(K):
            sel = X[labels == k]
            if len(sel) == 0:
                # Re-seed empty cluster on point furthest from current centers.
                far = int(np.argmax(d2.min(axis=1)))
                new_centers[k] = X[far]
            else:
                new_centers[k] = sel.mean(axis=0)
        shift = float(np.linalg.norm(new_centers - centers))
        centers = new_centers
        if shift <= tol:
            break
    d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(d2, axis=1)
    inertia = float(d2[np.arange(X.shape[0]), labels].sum())
    return labels, centers, inertia


def kmeans(
    X: np.ndarray,
    K: int,
    *,
    n_init: int = 50,
    max_iter: int = 300,
    tol: float = 1e-6,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run KMeans with k-means++ init and ``n_init`` restarts.

    Returns ``(labels, centers, inertia)`` of the best (lowest inertia) restart.
    Deterministic given ``random_state``.
    """
    X = np.asarray(X, dtype=float)
    rng_master = np.random.default_rng(random_state)
    seeds = rng_master.integers(0, 2**31 - 1, size=n_init)
    best = None
    for s in seeds:
        rng = np.random.default_rng(int(s))
        c0 = _kmeans_pp_init(X, K, rng)
        labels, centers, inertia = _lloyd(X, c0, max_iter, tol)
        if best is None or inertia < best[2]:
            best = (labels, centers, inertia)
    return best  # type: ignore[return-value]
