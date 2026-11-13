"""Run a small planted-support simulation without reading or writing data files."""

import argparse

from .fit import fit_ossa, score_alignment
from .simulation import (
    MultiCliqueConfig,
    SingleCliqueConfig,
    generate_multi_clique,
    generate_single_clique,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("single", "multi"), default="single")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    k = 3
    settings = dict(
        p=12, n=20, alpha=0.25, sigma_A=0.05, sigma_B=0.05,
        bg_scale_A=0.03, bg_scale_B=0.03, signal_scale=2.0,
    )
    if args.mode == "single":
        sample = generate_single_clique(
            SingleCliqueConfig(k=k, decoy_size=2, decoy_scale_A=0.1,
                               decoy_scale_B=0.1, **settings),
            seed=args.seed,
        )
        planted = [sample.V_star]
        penalty = 0.0
    else:
        sample = generate_multi_clique(
            MultiCliqueConfig(K=2, kappa=k, **settings), seed=args.seed
        )
        planted = sample.V_stars
        penalty = 25.0

    split = 15
    result = fit_ossa(
        sample.A[:split], sample.B[:split], k,
        n_supports=len(planted), overlap_penalty=penalty,
        n_restarts=4, seed=args.seed, max_outer=20, max_iter_w=30, max_iter_s=30,
    )
    estimated = [sorted(nodes.tolist()) for nodes in result["hard_cliques"]]
    held_out = score_alignment(
        sample.A[split:], sample.B[split:], result["W"], estimated
    )
    print(f"Planted supports:   {planted}")
    print(f"Estimated supports: {estimated}")
    print(f"Training objective: {result['final_objective']:.4f}")
    print(f"Held-out alignment: {held_out:.4f}")


if __name__ == "__main__":
    main()
