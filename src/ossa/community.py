"""Spectral community discovery and diagnostics for training FC."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .numpy_kmeans import kmeans

EIGEN_FLOOR = 1e-12


# ---------------------------------------------------------------------------
# Spectral embedding
# ---------------------------------------------------------------------------


def _abs_sym(M: np.ndarray) -> np.ndarray:
    M = 0.5 * (M + M.T)
    M = np.abs(M)
    np.fill_diagonal(M, 0.0)
    return M


def normalized_laplacian(W: np.ndarray) -> np.ndarray:
    """L_sym = I - D^{-1/2} W D^{-1/2}, with W >= 0 and zero diagonal."""
    W = _abs_sym(W)
    deg = W.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, EIGEN_FLOOR)), 0.0)
    L_sym = np.eye(W.shape[0]) - (d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :])
    return 0.5 * (L_sym + L_sym.T)


def spectral_embedding(A_bar: np.ndarray, Q: int) -> np.ndarray:
    """Top-Q eigenvectors of |A_bar| (FC-style affinity), row-normalised.

    The top eigenvectors of the absolute, symmetric affinity are row-normalized
    before k-means, which clusters their directions on the unit sphere.
    """
    M = _abs_sym(A_bar)
    w, V = np.linalg.eigh(M)
    U = V[:, -Q:]  # top-Q (eigh returns ascending)
    norms = np.linalg.norm(U, axis=1, keepdims=True) + 1e-12
    return U / norms


def silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette coefficient (numpy-only)."""
    n = X.shape[0]
    K = int(labels.max()) + 1
    if K < 2:
        return 0.0
    # Pairwise distances (Euclidean) for the small p=68 case.
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    sil = np.zeros(n, dtype=float)
    for i in range(n):
        own = np.where(labels == labels[i])[0]
        own = own[own != i]
        if len(own) == 0:
            sil[i] = 0.0
            continue
        a = D[i, own].mean()
        b = np.inf
        for k in range(K):
            if k == labels[i]:
                continue
            other = np.where(labels == k)[0]
            if len(other) == 0:
                continue
            b = min(b, D[i, other].mean())
        sil[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(sil.mean())


def newman_girvan_modularity(W: np.ndarray, labels: np.ndarray) -> float:
    """Q(c) = (1/2m) sum_{ij} (W_ij - k_i k_j / 2m) delta(c_i, c_j)."""
    W = _abs_sym(W)
    m2 = W.sum()  # 2m
    if m2 <= 0:
        return 0.0
    deg = W.sum(axis=1)
    Q = 0.0
    K = int(labels.max()) + 1
    for c in range(K):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        sub = W[np.ix_(idx, idx)]
        e_c = sub.sum() / m2
        a_c = deg[idx].sum() / m2
        Q += e_c - a_c**2
    return float(Q)


# ---------------------------------------------------------------------------
# Diagnostics for Q
# ---------------------------------------------------------------------------


def diagnostic_curves(A_bar: np.ndarray, Qs: List[int], *, kmeans_seed: int = 0) -> Dict[str, list]:
    """Compute eigengap, modularity, silhouette over candidate Q values."""
    M = _abs_sym(A_bar)
    L = normalized_laplacian(A_bar)
    eigs_L = np.linalg.eigvalsh(L)
    eigs_L = np.sort(eigs_L)
    out: Dict[str, list] = {
        "Q": list(Qs),
        "eigengap": [],
        "modularity": [],
        "silhouette": [],
    }
    for Q in Qs:
        # Eigengap: gap between the Q-th and (Q+1)-th smallest eigenvalues
        if Q < len(eigs_L):
            gap = float(eigs_L[Q] - eigs_L[Q - 1])
        else:
            gap = 0.0
        out["eigengap"].append(gap)
        # Spectral embedding + KMeans for modularity & silhouette
        U = spectral_embedding(A_bar, Q)
        labels, _, _ = kmeans(U, Q, n_init=50, random_state=kmeans_seed)
        out["modularity"].append(newman_girvan_modularity(M, labels))
        out["silhouette"].append(silhouette(U, labels))
    return out


# ---------------------------------------------------------------------------
# Final clustering
# ---------------------------------------------------------------------------


def cluster(A_bar: np.ndarray, Q: int, *, kmeans_seed: int = 0, n_init: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Spectral + KMeans community partition.

    Returns ``(labels, embedding)``. Labels are in 0..Q-1.
    """
    U = spectral_embedding(A_bar, Q)
    labels, _, _ = kmeans(U, Q, n_init=n_init, random_state=kmeans_seed)
    return labels, U


def cluster_sizes(labels: np.ndarray) -> List[int]:
    return [int(np.sum(labels == q)) for q in range(int(labels.max()) + 1)]
