"""Fit and evaluate OSSA in a single functional community."""

from collections.abc import Sequence
from numbers import Integral

import numpy as np

from .community import cluster
from .optimizer import fit_sc_fc_alignment, objective_J_k, symmetrize_stack


def _connectivity_stack(matrices: np.ndarray) -> np.ndarray:
    stack = np.asarray(matrices, dtype=float)
    if stack.ndim != 3 or not stack.size or stack.shape[1] != stack.shape[2]:
        raise ValueError("Connectivity matrices must have nonempty shape (n, p, p).")
    if not np.isfinite(stack).all():
        raise ValueError("Connectivity matrices must contain only finite values.")
    return symmetrize_stack(stack)


def _paired_stacks(
    fc: np.ndarray, sc: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    fc, sc = _connectivity_stack(fc), _connectivity_stack(sc)
    if fc.shape != sc.shape:
        raise ValueError("FC and SC must contain matching subjects and nodes.")
    return fc, sc


def discover_communities(
    fc: np.ndarray, n_communities: int, *, seed: int = 0
) -> np.ndarray:
    """Cluster the mean absolute training FC matrix into functional communities."""
    fc = _connectivity_stack(fc)
    if not isinstance(n_communities, Integral) or not 1 <= n_communities <= fc.shape[1]:
        raise ValueError("n_communities must be an integer between 1 and the number of nodes.")
    labels, _ = cluster(np.abs(fc).mean(axis=0), n_communities, kmeans_seed=seed)
    return labels


def fit_ossa(
    fc: np.ndarray,
    sc: np.ndarray,
    k: int,
    *,
    n_supports: int = 1,
    overlap_penalty: float = 0.0,
    n_restarts: int = 8,
    seed: int = 0,
    update_rotation: bool = True,
    initial_rotation: np.ndarray | None = None,
    max_outer: int = 80,
    max_iter_w: int = 200,
    max_iter_s: int = 200,
) -> dict:
    """Fit OSSA to paired training FC and SC arrays of shape (n, p, p).

    Each fitted support contains k nodes. Set n_supports > 1 to use the
    extension with a shared rotation and a pairwise overlap penalty. Restarts
    vary the initial membership weights and are ranked by training objective.
    The result includes the rotation W, soft weights, and zero-based supports
    under hard_cliques. With update_rotation=False, the rotation is fixed to
    the identity by default or to the orthogonalized initial_rotation.
    """
    fc, sc = _paired_stacks(fc, sc)
    for name, value in (
        ("k", k), ("n_supports", n_supports), ("n_restarts", n_restarts),
        ("max_outer", max_outer), ("max_iter_w", max_iter_w), ("max_iter_s", max_iter_s),
    ):
        if not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if k > fc.shape[1]:
        raise ValueError("k cannot exceed the number of nodes.")
    if not np.isfinite(overlap_penalty) or overlap_penalty < 0:
        raise ValueError("overlap_penalty must be finite and nonnegative.")
    if initial_rotation is not None and not np.isfinite(initial_rotation).all():
        raise ValueError("initial_rotation must contain only finite values.")

    best = None
    for restart in range(n_restarts):
        restart_seed = seed + restart
        result = fit_sc_fc_alignment(
            fc, sc, k, K_cliques=n_supports, lambda_overlap=overlap_penalty,
            init="random", random_state=restart_seed, W_init=initial_rotation,
            update_W=update_rotation, max_outer=max_outer, max_iter_W=max_iter_w,
            max_iter_s=max_iter_s, eta_W=0.05, eta_s=0.05,
        )
        final_objective = result["objective_history"][-1]
        if not np.isfinite(final_objective):
            raise FloatingPointError("The OSSA training objective is not finite.")
        if best is None or final_objective > best["final_objective"]:
            result["final_objective"] = final_objective
            result["restart_seed"] = restart_seed
            best = result
    return best


def score_alignment(
    fc: np.ndarray,
    sc: np.ndarray,
    rotation: np.ndarray,
    supports: Sequence[Sequence[int]],
) -> float:
    """Return mean alignment on held-out pairs with the supplied hard supports.

    The score uses the training-fitted rotation and excludes the overlap
    penalty. For multiple supports, their alignment contributions are summed.
    """
    fc, sc = _paired_stacks(fc, sc)
    n, p, _ = fc.shape
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (p, p) or not np.isfinite(rotation).all():
        raise ValueError("rotation must be a finite (p, p) matrix.")
    masks = np.zeros((len(supports), p))
    if not len(masks):
        raise ValueError("Provide at least one nonempty support.")
    for mask, support in zip(masks, supports):
        nodes = np.asarray(support)
        if (
            nodes.ndim != 1 or not nodes.size or nodes.dtype.kind not in "iu"
            or (nodes < 0).any() or (nodes >= p).any()
            or np.unique(nodes).size != nodes.size
        ):
            raise ValueError("Each support must contain unique integer node indices in [0, p).")
        mask[nodes] = 1.0
    return objective_J_k(fc, sc, rotation, masks) / n
