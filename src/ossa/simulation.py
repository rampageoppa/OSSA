"""Planted-support generative models for the OSSA simulations.

Reproduces the generative model described in Section 5.1 of the paper:
    B_i = C_i + xi^B_i + tau^B_i + sigma_B * E^B_i
    A_i = W* C_i W*^T + xi^A_i + tau^A_i + sigma_A * E^A_i

with a low-rank planted block C_i on V_star, rank-4 background xi, decoy
blocks tau in disjoint nodes (per-modality only), and rotation strength alpha
that interpolates W_star between identity and a random orthogonal matrix.

The K-clique generator generalizes by summing K low-rank blocks on K
(possibly overlapping) supports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sym(X: np.ndarray) -> np.ndarray:
    return 0.5 * (X + X.T)


def random_orthogonal(p: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly random orthogonal matrix in O(p) via QR of Gaussian."""
    G = rng.standard_normal((p, p))
    Q, R = np.linalg.qr(G)
    # Sign-fix so the construction is uniform on O(p).
    sgn = np.sign(np.diag(R))
    sgn[sgn == 0] = 1.0
    return Q * sgn[None, :]


def interp_orthogonal(
    p: int, alpha: float, rng: np.random.Generator
) -> np.ndarray:
    """Interpolate between I_p and a random orthogonal matrix at strength alpha.

    Returns I_p when alpha=0 and a fresh random orthogonal matrix when alpha=1.
    For 0<alpha<1 we take W_alpha = polar(alpha * Q + (1-alpha) * I), which
    smoothly interpolates while staying on O(p).
    """
    if alpha <= 0.0:
        return np.eye(p)
    Q = random_orthogonal(p, rng)
    if alpha >= 1.0:
        return Q
    M = alpha * Q + (1.0 - alpha) * np.eye(p)
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    return U @ Vt


def _low_rank_block(
    p: int,
    support: Sequence[int],
    rank: int,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """A symmetric rank-`rank` block, supported only on the index set ``support``.

    Returns a p x p symmetric matrix whose nonzero entries lie inside
    support x support.
    """
    s = list(support)
    k = len(s)
    if k == 0 or rank == 0:
        return np.zeros((p, p))
    # Random (k x rank) factor scaled so that ||UU^T||_F ~ scale * sqrt(k * rank)
    U = rng.standard_normal((k, rank)) * (scale / max(np.sqrt(rank), 1.0))
    block = U @ U.T  # symmetric PSD
    # Random sign mask to introduce both signs (common in synthetic FC/SC).
    signs = rng.choice([-1.0, 1.0], size=(k, k))
    signs = 0.5 * (signs + signs.T)
    block = block * signs
    # Embed.
    out = np.zeros((p, p))
    idx = np.array(s, dtype=int)
    out[np.ix_(idx, idx)] = block
    return _sym(out)


def _gaussian_sym(p: int, rng: np.random.Generator) -> np.ndarray:
    G = rng.standard_normal((p, p))
    return _sym(G)


# ---------------------------------------------------------------------------
# Single-clique generator
# ---------------------------------------------------------------------------


@dataclass
class SingleCliqueConfig:
    p: int = 40
    n: int = 30
    k: int = 10
    alpha: float = 1.0          # rotation strength
    sigma_B: float = 0.4
    sigma_A: float = 0.4
    signal_scale: float = 1.5
    bg_scale_B: float = 0.6
    bg_scale_A: float = 0.6
    bg_rank: int = 4
    decoy_size: int = 12
    decoy_scale_B: float = 1.5
    decoy_scale_A: float = 1.5
    signal_rank: int = 3
    decoy_rank: int = 3


@dataclass
class SingleCliqueDataset:
    A: np.ndarray  # (n, p, p)  FC
    B: np.ndarray  # (n, p, p)  SC
    V_star: List[int]
    W_star: np.ndarray
    decoy_B_nodes: List[int]
    decoy_A_nodes: List[int]
    cfg: SingleCliqueConfig


def generate_single_clique(
    cfg: SingleCliqueConfig, seed: int
) -> SingleCliqueDataset:
    rng = np.random.default_rng(seed)
    p = cfg.p

    # Planted support and W*
    V_star = sorted(rng.choice(p, size=cfg.k, replace=False).tolist())
    W_star = interp_orthogonal(p, cfg.alpha, rng)

    # Decoys disjoint from V_star
    avail = sorted(set(range(p)) - set(V_star))
    if len(avail) < 2 * cfg.decoy_size:
        raise ValueError("Not enough nodes for two disjoint decoy sets.")
    rng.shuffle(avail)
    decoy_B_nodes = sorted(avail[: cfg.decoy_size])
    decoy_A_nodes = sorted(avail[cfg.decoy_size : 2 * cfg.decoy_size])

    A = np.zeros((cfg.n, p, p))
    B = np.zeros((cfg.n, p, p))
    for i in range(cfg.n):
        # Signal block on V_star, shared between modalities (after W* mixing)
        C = _low_rank_block(
            p, V_star, cfg.signal_rank, cfg.signal_scale, rng
        )

        # Backgrounds: independent rank-bg_rank symmetric matrices
        xi_B = _low_rank_block(
            p, list(range(p)), cfg.bg_rank, cfg.bg_scale_B, rng
        )
        xi_A = _low_rank_block(
            p, list(range(p)), cfg.bg_rank, cfg.bg_scale_A, rng
        )

        # Decoys, modality-specific (only one modality has the decoy block)
        tau_B = _low_rank_block(
            p, decoy_B_nodes, cfg.decoy_rank, cfg.decoy_scale_B, rng
        )
        tau_A = _low_rank_block(
            p, decoy_A_nodes, cfg.decoy_rank, cfg.decoy_scale_A, rng
        )

        # Noise
        E_B = _gaussian_sym(p, rng) * cfg.sigma_B
        E_A = _gaussian_sym(p, rng) * cfg.sigma_A

        Bi = C + xi_B + tau_B + E_B
        Ai = W_star @ C @ W_star.T + xi_A + tau_A + E_A
        # Zero diagonal so degree baselines see only off-diagonal entries.
        np.fill_diagonal(Bi, 0.0)
        np.fill_diagonal(Ai, 0.0)
        B[i] = _sym(Bi)
        A[i] = _sym(Ai)

    return SingleCliqueDataset(
        A=A,
        B=B,
        V_star=V_star,
        W_star=W_star,
        decoy_B_nodes=decoy_B_nodes,
        decoy_A_nodes=decoy_A_nodes,
        cfg=cfg,
    )


# ---------------------------------------------------------------------------
# Multi-clique generator
# ---------------------------------------------------------------------------


@dataclass
class MultiCliqueConfig:
    p: int = 50
    n: int = 30
    K: int = 2
    kappa: int = 10
    overlap: int = 0
    alpha: float = 0.6
    sigma_B: float = 0.4
    sigma_A: float = 0.4
    signal_scale: float = 1.5
    bg_scale_B: float = 0.4
    bg_scale_A: float = 0.4
    bg_rank: int = 4
    signal_rank: int = 3


@dataclass
class MultiCliqueDataset:
    A: np.ndarray
    B: np.ndarray
    V_stars: List[List[int]]
    W_star: np.ndarray
    cfg: MultiCliqueConfig


def _sample_overlapping_supports(
    p: int, K: int, kappa: int, overlap: int, rng: np.random.Generator
) -> List[List[int]]:
    """Sample K supports, each of size kappa, such that consecutive supports
    share exactly ``overlap`` nodes and otherwise are disjoint.

    When overlap=0 supports are pairwise disjoint.
    """
    if overlap > kappa:
        raise ValueError("overlap must be <= kappa.")
    if K * kappa - (K - 1) * overlap > p:
        raise ValueError("not enough nodes for the requested K, kappa, overlap.")

    nodes = rng.permutation(p).tolist()
    supports: List[List[int]] = []
    cursor = 0
    for ell in range(K):
        if ell == 0:
            supp = nodes[cursor : cursor + kappa]
            cursor += kappa
        else:
            shared = supports[-1][-overlap:] if overlap > 0 else []
            new_part = nodes[cursor : cursor + (kappa - overlap)]
            cursor += (kappa - overlap)
            supp = shared + new_part
        supports.append(sorted(supp))
    return supports


def generate_multi_clique(
    cfg: MultiCliqueConfig, seed: int
) -> MultiCliqueDataset:
    rng = np.random.default_rng(seed)
    p = cfg.p

    V_stars = _sample_overlapping_supports(
        p=p, K=cfg.K, kappa=cfg.kappa, overlap=cfg.overlap, rng=rng
    )
    W_star = interp_orthogonal(p, cfg.alpha, rng)

    A = np.zeros((cfg.n, p, p))
    B = np.zeros((cfg.n, p, p))
    for i in range(cfg.n):
        C_total = np.zeros((p, p))
        for V in V_stars:
            C_total = C_total + _low_rank_block(
                p, V, cfg.signal_rank, cfg.signal_scale, rng
            )
        xi_B = _low_rank_block(p, list(range(p)), cfg.bg_rank, cfg.bg_scale_B, rng)
        xi_A = _low_rank_block(p, list(range(p)), cfg.bg_rank, cfg.bg_scale_A, rng)
        E_B = _gaussian_sym(p, rng) * cfg.sigma_B
        E_A = _gaussian_sym(p, rng) * cfg.sigma_A

        Bi = C_total + xi_B + E_B
        Ai = W_star @ C_total @ W_star.T + xi_A + E_A
        np.fill_diagonal(Bi, 0.0)
        np.fill_diagonal(Ai, 0.0)
        B[i] = _sym(Bi)
        A[i] = _sym(Ai)

    return MultiCliqueDataset(
        A=A, B=B, V_stars=V_stars, W_star=W_star, cfg=cfg
    )
