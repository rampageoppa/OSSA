from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def symmetrize_stack(X: np.ndarray) -> np.ndarray:
    """
    Symmetrize a stack of matrices: X <- (X + X^T)/2 along the last two axes.
    """
    if X.ndim == 2:
        return 0.5 * (X + X.T)
    return 0.5 * (X + np.swapaxes(X, -1, -2))


def project_to_orthogonal(W: np.ndarray, enforce_det_pos: bool = False) -> np.ndarray:
    """
    Section 3 / eq. (projO): Frobenius projection onto O(p) via SVD.
    """
    U, _, Vt = np.linalg.svd(W, full_matrices=False)
    Q = U @ Vt
    if enforce_det_pos and np.linalg.det(Q) < 0:
        U[:, -1] *= -1.0
        Q = U @ Vt
    return Q


def project_s_box_l1(
    s: np.ndarray,
    k: float,
    *,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> np.ndarray:
    """
    Section 4 / Section 9 projection onto {0 <= s <= 1, 1^T s <= k}.

    The tex specifies:
      1) clip to [0, 1]
      2) if the clipped sum exceeds k, apply a threshold projection onto the
         nonnegative simplex (implemented here by bisection).
    """
    s = np.asarray(s, dtype=float)
    clipped = np.clip(s, 0.0, 1.0)
    p = clipped.size

    if k >= p:
        return clipped
    if clipped.sum() <= k:
        return clipped

    lo = 0.0
    hi = float(np.max(clipped))
    for _ in range(max_iter):
        tau = 0.5 * (lo + hi)
        projected = np.maximum(clipped - tau, 0.0)
        if projected.sum() > k:
            lo = tau
        else:
            hi = tau
        if hi - lo <= tol:
            break

    projected = np.maximum(clipped - hi, 0.0)
    return np.clip(projected, 0.0, 1.0)


def _coerce_soft_memberships(
    soft_memberships: Sequence[np.ndarray] | np.ndarray,
) -> np.ndarray:
    arr = np.asarray(soft_memberships, dtype=float)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim != 2:
        raise ValueError(
            "soft memberships must have shape (p,) or (K_cliques, p). "
            f"Got shape={arr.shape}."
        )
    return arr


def project_soft_memberships_collection(
    soft_memberships: Sequence[np.ndarray] | np.ndarray,
    clique_size: float,
) -> np.ndarray:
    soft = _coerce_soft_memberships(soft_memberships).copy()
    for ell in range(soft.shape[0]):
        soft[ell] = project_s_box_l1(soft[ell], float(clique_size))
    return soft


def hard_cliques_from_soft(
    soft_memberships: Sequence[np.ndarray] | np.ndarray,
    clique_size: int,
) -> List[np.ndarray]:
    soft = _coerce_soft_memberships(soft_memberships)
    if clique_size < 1:
        raise ValueError(f"clique_size must be >= 1. Got {clique_size}.")
    return [np.argsort(-soft[ell])[:clique_size] for ell in range(soft.shape[0])]


def overlap_penalty_soft(
    soft_memberships: Sequence[np.ndarray] | np.ndarray,
    lambda_overlap: float,
) -> float:
    soft = _coerce_soft_memberships(soft_memberships)
    if soft.shape[0] <= 1 or lambda_overlap == 0.0:
        return 0.0

    penalty = 0.0
    for ell in range(soft.shape[0]):
        for ell2 in range(ell + 1, soft.shape[0]):
            penalty += float(np.dot(soft[ell], soft[ell2]))
    return float(lambda_overlap * penalty)


def overlap_penalty_hard(hard_cliques: Sequence[np.ndarray], lambda_overlap: float) -> float:
    if len(hard_cliques) <= 1 or lambda_overlap == 0.0:
        return 0.0

    penalty = 0.0
    hard_sets = [set(map(int, clique.tolist())) for clique in hard_cliques]
    for ell in range(len(hard_sets)):
        for ell2 in range(ell + 1, len(hard_sets)):
            penalty += float(len(hard_sets[ell] & hard_sets[ell2]))
    return float(lambda_overlap * penalty)


def compute_aggregated_masked_matrices(
    B: np.ndarray,
    soft_memberships: Sequence[np.ndarray] | np.ndarray,
) -> np.ndarray:
    """
    Section 9 / eq. (kc-Mi): M_i = sum_ell S^(ell) B_i S^(ell).
    """
    soft = _coerce_soft_memberships(soft_memberships)
    M = np.zeros_like(B, dtype=float)
    for s in soft:
        M += B * s[None, :, None] * s[None, None, :]
    return M


def alignment_objective_from_aggregated_M(
    A: np.ndarray,
    W: np.ndarray,
    aggregated_M: np.ndarray,
) -> float:
    J = 0.0
    for i in range(A.shape[0]):
        Q_i = W.T @ A[i] @ W
        J += float(np.sum(Q_i * aggregated_M[i]))
    return float(J)


def compute_d_matrix(A: np.ndarray, B: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Section 4 / eq. (D) and Section 9 fixed-W reduction.
    """
    _, p, _ = A.shape
    D = np.zeros((p, p), dtype=float)
    for i in range(A.shape[0]):
        Q_i = W.T @ A[i] @ W
        D += Q_i * B[i]
    return 0.5 * (D + D.T)


def fixed_w_objective_from_d(
    D: np.ndarray,
    soft_memberships: Sequence[np.ndarray] | np.ndarray,
    lambda_overlap: float = 0.0,
) -> float:
    """
    Section 9 / eq. (kc-fixedW).
    """
    soft = _coerce_soft_memberships(soft_memberships)
    value = 0.0
    for s in soft:
        value += float(s.T @ D @ s)
    value -= overlap_penalty_soft(soft, lambda_overlap)
    return float(value)


def objective_J_k(
    A: np.ndarray,
    B: np.ndarray,
    W: np.ndarray,
    soft_memberships: Sequence[np.ndarray] | np.ndarray,
    *,
    lambda_overlap: float = 0.0,
) -> float:
    """
    Section 9 / eq. (kc-obj).
    """
    aggregated_M = compute_aggregated_masked_matrices(B, soft_memberships)
    value = alignment_objective_from_aggregated_M(A, W, aggregated_M)
    value -= overlap_penalty_soft(soft_memberships, lambda_overlap)
    return float(value)


def objective_J(A: np.ndarray, B: np.ndarray, W: np.ndarray, s: np.ndarray) -> float:
    """
    Backward-compatible single-clique wrapper for eq. (obj).
    """
    return objective_J_k(A, B, W, s, lambda_overlap=0.0)


def gradient_W_k(A: np.ndarray, W: np.ndarray, aggregated_M: np.ndarray) -> np.ndarray:
    """
    Section 9 / eq. (kc-gradW): grad_W = 2 * sum_i A_i W M_i.
    """
    grad = np.zeros_like(W)
    for i in range(A.shape[0]):
        grad += A[i] @ W @ aggregated_M[i]
    return 2.0 * grad


def gradient_s_k(
    D: np.ndarray,
    soft_memberships: Sequence[np.ndarray] | np.ndarray,
    clique_index: int,
    lambda_overlap: float,
) -> np.ndarray:
    """
    Section 9 / eq. (kc-grads) using the latest Gauss-Seidel iterate state.
    """
    soft = _coerce_soft_memberships(soft_memberships)
    if clique_index < 0 or clique_index >= soft.shape[0]:
        raise IndexError(
            f"clique_index={clique_index} out of range for K_cliques={soft.shape[0]}."
        )

    other_sum = np.sum(soft, axis=0) - soft[clique_index]
    return 2.0 * (D @ soft[clique_index]) - float(lambda_overlap) * other_sum


def optimize_W_k(
    A: np.ndarray,
    B: np.ndarray,
    W: np.ndarray,
    soft_memberships: Sequence[np.ndarray] | np.ndarray,
    *,
    lambda_overlap: float = 0.0,
    eta: float = 1e-2,
    max_iter: int = 200,
    tol_improve: float = 1e-10,
    min_eta: float = 1e-12,
    enforce_det_pos: bool = False,
    return_info: bool = False,
) -> Tuple[np.ndarray, float, float] | Tuple[np.ndarray, float, float, Dict[str, Any]]:
    """
    Section 9 W-step with acceptance based on the full K-clique objective.
    """
    soft = _coerce_soft_memberships(soft_memberships)
    aggregated_M = compute_aggregated_masked_matrices(B, soft)
    penalty = overlap_penalty_soft(soft, lambda_overlap)

    J = alignment_objective_from_aggregated_M(A, W, aggregated_M) - penalty
    accepted_steps = 0
    backtracks = 0
    iterations_run = 0
    stopped_reason = "max_iter_reached"
    for _ in range(max_iter):
        iterations_run += 1
        grad = gradient_W_k(A, W, aggregated_M)

        eta_try = eta
        improved = False
        while eta_try >= min_eta:
            W_tilde = W + eta_try * grad
            W_new = project_to_orthogonal(W_tilde, enforce_det_pos=enforce_det_pos)
            J_new = alignment_objective_from_aggregated_M(A, W_new, aggregated_M) - penalty
            if J_new >= J + tol_improve:
                W, J = W_new, J_new
                eta = eta_try
                improved = True
                accepted_steps += 1
                break
            backtracks += 1
            eta_try *= 0.5

        if not improved:
            stopped_reason = "no_accepted_step"
            break

    info = {
        "accepted_steps": int(accepted_steps),
        "backtracks": int(backtracks),
        "iterations_run": int(iterations_run),
        "stopped_reason": str(stopped_reason),
        "converged": bool(stopped_reason == "no_accepted_step"),
    }
    if return_info:
        return W, float(eta), float(J), info
    return W, float(eta), float(J)


def optimize_W(
    A: np.ndarray,
    B: np.ndarray,
    W: np.ndarray,
    s: np.ndarray,
    eta: float = 1e-2,
    max_iter: int = 200,
    tol_improve: float = 1e-10,
    min_eta: float = 1e-12,
    enforce_det_pos: bool = False,
) -> Tuple[np.ndarray, float, float]:
    """
    Backward-compatible single-clique W-step wrapper.
    """
    W_out, eta_out, J_out = optimize_W_k(
        A,
        B,
        W,
        s,
        lambda_overlap=0.0,
        eta=eta,
        max_iter=max_iter,
        tol_improve=tol_improve,
        min_eta=min_eta,
        enforce_det_pos=enforce_det_pos,
    )
    return W_out, eta_out, J_out


def optimize_soft_memberships_gauss_seidel(
    A: np.ndarray,
    B: np.ndarray,
    W: np.ndarray,
    soft_memberships: Sequence[np.ndarray] | np.ndarray,
    clique_size: float,
    *,
    lambda_overlap: float = 0.0,
    eta: float = 1e-2,
    max_iter: int = 200,
    tol_improve: float = 1e-10,
    min_eta: float = 1e-12,
) -> Tuple[np.ndarray, float, float, Dict[str, Any]]:
    """
    Section 9 Gauss-Seidel s-step with projected gradient ascent.
    """
    soft = _coerce_soft_memberships(soft_memberships).copy()
    D = compute_d_matrix(A, B, W)
    J = fixed_w_objective_from_d(D, soft, lambda_overlap=lambda_overlap)

    accepted_total = 0
    backtracks_total = 0
    clique_logs: List[Dict[str, Any]] = []

    for ell in range(soft.shape[0]):
        accepted_this_clique = 0
        backtracks_this_clique = 0
        iterations_this_clique = 0
        start_obj = float(J)
        start_vec = soft[ell].copy()
        stopped_reason = "max_iter_reached"

        for _ in range(max_iter):
            iterations_this_clique += 1
            grad = gradient_s_k(D, soft, ell, lambda_overlap)
            s_old = soft[ell].copy()

            eta_try = eta
            improved = False
            while eta_try >= min_eta:
                s_tilde = s_old + eta_try * grad
                s_new = project_s_box_l1(s_tilde, float(clique_size))

                soft[ell] = s_new
                J_new = fixed_w_objective_from_d(D, soft, lambda_overlap=lambda_overlap)
                if J_new >= J + tol_improve:
                    J = float(J_new)
                    eta = eta_try
                    improved = True
                    accepted_this_clique += 1
                    accepted_total += 1
                    break

                soft[ell] = s_old
                backtracks_this_clique += 1
                backtracks_total += 1
                eta_try *= 0.5

            if not improved:
                soft[ell] = s_old
                stopped_reason = "no_accepted_step"
                break

        clique_logs.append(
            {
                "clique_index": int(ell),
                "accepted_steps": int(accepted_this_clique),
                "backtracks": int(backtracks_this_clique),
                "iterations_run": int(iterations_this_clique),
                "stopped_reason": str(stopped_reason),
                "converged": bool(stopped_reason == "no_accepted_step"),
                "delta_l2": float(np.linalg.norm(soft[ell] - start_vec)),
                "objective_gain": float(J - start_obj),
            }
        )

    return soft, float(eta), float(J), {
        "D": D,
        "accepted_total": int(accepted_total),
        "backtracks_total": int(backtracks_total),
        "clique_logs": clique_logs,
    }


def optimize_s(
    A: np.ndarray,
    B: np.ndarray,
    W: np.ndarray,
    s: np.ndarray,
    k: float,
    eta: float = 1e-2,
    max_iter: int = 200,
    tol_improve: float = 1e-10,
    min_eta: float = 1e-12,
) -> Tuple[np.ndarray, float, float]:
    """
    Backward-compatible single-clique s-step wrapper.
    """
    soft, eta_out, J, _ = optimize_soft_memberships_gauss_seidel(
        A,
        B,
        W,
        s,
        clique_size=k,
        lambda_overlap=0.0,
        eta=eta,
        max_iter=max_iter,
        tol_improve=tol_improve,
        min_eta=min_eta,
    )
    return soft[0], eta_out, J


def _initialize_soft_memberships(
    p: int,
    K_cliques: int,
    clique_size: int,
    init: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if init == "uniform":
        soft = np.full((K_cliques, p), min(1.0, float(clique_size) / float(p)), dtype=float)
        for ell in range(K_cliques):
            soft[ell] = project_s_box_l1(soft[ell], float(clique_size))
        return soft

    if init == "random":
        soft = np.zeros((K_cliques, p), dtype=float)
        for ell in range(K_cliques):
            soft[ell] = project_s_box_l1(rng.random(p), float(clique_size))
        return soft

    if init == "binary":
        soft = np.zeros((K_cliques, p), dtype=float)
        if K_cliques * clique_size <= p:
            perm = rng.permutation(p)
            for ell in range(K_cliques):
                idx = perm[ell * clique_size : (ell + 1) * clique_size]
                soft[ell, idx] = 1.0
        else:
            for ell in range(K_cliques):
                idx = rng.choice(p, size=clique_size, replace=False)
                soft[ell, idx] = 1.0
        return soft

    raise ValueError("init must be one of {'uniform', 'random', 'binary'}.")


def fit_sc_fc_alignment(
    A: np.ndarray,
    B: np.ndarray,
    k: int,
    *,
    K_cliques: int = 1,
    clique_size: Optional[int] = None,
    lambda_overlap: float = 0.0,
    max_outer: int = 50,
    eta_W: float = 1e-2,
    eta_s: float = 1e-2,
    max_iter_W: int = 200,
    max_iter_s: int = 200,
    tol_outer: float = 1e-8,
    tol_params: Optional[float] = None,
    tol_inner: float = 1e-10,
    min_eta: float = 1e-12,
    init: str = "uniform",
    random_state: Optional[int] = None,
    enforce_det_pos: bool = False,
    W_init: Optional[np.ndarray] = None,
    soft_memberships_init: Optional[Sequence[np.ndarray] | np.ndarray] = None,
    update_W: bool = True,
) -> Dict[str, Any]:
    """
    Sections 9--11 implementation entry point.

    Backward compatibility:
      - When K_cliques=1 and lambda_overlap=0, this reduces to the original
        single-clique objective and keeps the old `s` / `V_topk` return fields.
      - The positional argument `k` is preserved; by default clique_size = k.
    """
    if A.shape != B.shape or A.ndim != 3 or A.shape[1] != A.shape[2]:
        raise ValueError("A and B must have the same shape (n, p, p).")

    A = symmetrize_stack(A.astype(float, copy=True))
    B = symmetrize_stack(B.astype(float, copy=True))

    _, p, _ = A.shape
    if clique_size is None:
        clique_size = int(k)
    if not (1 <= clique_size <= p):
        raise ValueError(
            f"clique_size must be in [1, p]. Got clique_size={clique_size}, p={p}."
        )
    if K_cliques < 1:
        raise ValueError(f"K_cliques must be >= 1. Got {K_cliques}.")
    if lambda_overlap < 0.0:
        raise ValueError(f"lambda_overlap must be >= 0. Got {lambda_overlap}.")

    rng = np.random.default_rng(random_state)
    if W_init is None:
        W = np.eye(p, dtype=float)
    else:
        W = np.asarray(W_init, dtype=float).copy()
        if W.shape != (p, p):
            raise ValueError(f"W_init must have shape {(p, p)}. Got {W.shape}.")
        W = project_to_orthogonal(W, enforce_det_pos=enforce_det_pos)

    if soft_memberships_init is None:
        soft_memberships = _initialize_soft_memberships(
            p=p,
            K_cliques=int(K_cliques),
            clique_size=int(clique_size),
            init=init,
            rng=rng,
        )
    else:
        soft_memberships = project_soft_memberships_collection(
            soft_memberships_init,
            float(clique_size),
        )
        if soft_memberships.shape != (int(K_cliques), p):
            raise ValueError(
                "soft_memberships_init must have shape "
                f"({int(K_cliques)}, {p}). Got {soft_memberships.shape}."
            )

    if tol_params is None:
        tol_params = tol_outer

    history: List[float] = []
    iteration_logs: List[Dict[str, Any]] = []
    converged = False
    convergence_reason = "max_outer_reached"

    J_prev = objective_J_k(A, B, W, soft_memberships, lambda_overlap=lambda_overlap)
    history.append(float(J_prev))

    for outer_iter in range(max_outer):
        W_prev = W.copy()
        soft_prev = soft_memberships.copy()

        if update_W:
            W, eta_W, J_after_W, w_info = optimize_W_k(
                A,
                B,
                W,
                soft_memberships,
                lambda_overlap=lambda_overlap,
                eta=eta_W,
                max_iter=max_iter_W,
                tol_improve=tol_inner,
                min_eta=min_eta,
                enforce_det_pos=enforce_det_pos,
                return_info=True,
            )
        else:
            J_after_W = objective_J_k(A, B, W, soft_memberships, lambda_overlap=lambda_overlap)
            w_info = {
                "accepted_steps": 0,
                "backtracks": 0,
                "iterations_run": 0,
                "stopped_reason": "fixed_W",
                "converged": True,
            }
        soft_memberships, eta_s, J_after_s, sweep_info = optimize_soft_memberships_gauss_seidel(
            A,
            B,
            W,
            soft_memberships,
            clique_size=float(clique_size),
            lambda_overlap=lambda_overlap,
            eta=eta_s,
            max_iter=max_iter_s,
            tol_improve=tol_inner,
            min_eta=min_eta,
        )

        J = objective_J_k(A, B, W, soft_memberships, lambda_overlap=lambda_overlap)
        history.append(float(J))

        delta_W = float(np.linalg.norm(W - W_prev, ord="fro"))
        delta_s_total = float(
            np.sum([np.linalg.norm(soft_memberships[ell] - soft_prev[ell]) for ell in range(K_cliques)])
        )
        delta_objective = float(J - J_prev)

        iteration_logs.append(
            {
                "outer_iter": int(outer_iter + 1),
                "objective_before": float(J_prev),
                "objective_after_W": float(J_after_W),
                "objective_after_s": float(J_after_s),
                "objective_final": float(J),
                "delta_objective": delta_objective,
                "delta_W_fro": delta_W,
                "delta_s_total_l2": delta_s_total,
                "update_W": bool(update_W),
                "eta_W": float(eta_W),
                "eta_s": float(eta_s),
                "accepted_W_steps": int(w_info["accepted_steps"]),
                "w_backtracks": int(w_info["backtracks"]),
                "w_iterations_run": int(w_info["iterations_run"]),
                "w_stopped_reason": str(w_info["stopped_reason"]),
                "accepted_s_steps": int(sweep_info["accepted_total"]),
                "s_backtracks_total": int(sweep_info["backtracks_total"]),
                "sweep_cliques": sweep_info["clique_logs"],
            }
        )

        obj_small = abs(delta_objective) <= tol_outer * max(1.0, abs(J_prev))
        param_small = (delta_W + delta_s_total) <= float(tol_params)
        if obj_small and param_small:
            converged = True
            convergence_reason = "outer_tolerance"
            break
        J_prev = float(J)

    hard_cliques = hard_cliques_from_soft(soft_memberships, int(clique_size))
    s_compat = soft_memberships[0].copy() if K_cliques == 1 else None
    v_compat = hard_cliques[0].copy() if K_cliques == 1 else None

    return {
        "W": W,
        "s": s_compat,
        "s_all": soft_memberships,
        "soft_memberships": soft_memberships,
        "V_topk": v_compat,
        "V_topk_all": hard_cliques,
        "hard_cliques": hard_cliques,
        "history": history,
        "objective_history": history,
        "iteration_logs": iteration_logs,
        "K_cliques": int(K_cliques),
        "clique_size": int(clique_size),
        "lambda_overlap": float(lambda_overlap),
        "converged": bool(converged),
        "convergence_reason": str(convergence_reason),
        "outer_iterations": int(len(iteration_logs)),
        "update_W": bool(update_W),
    }


def extract_cluster_submatrices(X: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """
    X: (n, P, P) full matrices
    idx: cluster node indices
    returns: (n, p, p) submatrices
    """
    idx = np.asarray(idx, dtype=int)
    return X[:, idx][:, :, idx]
