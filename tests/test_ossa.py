import unittest

import numpy as np

from ossa import discover_communities, fit_ossa, score_alignment
from ossa.simulation import (
    MultiCliqueConfig,
    SingleCliqueConfig,
    generate_multi_clique,
    generate_single_clique,
)


class OSSATests(unittest.TestCase):
    fit_options = dict(
        n_restarts=3, seed=7, max_outer=20, max_iter_w=30, max_iter_s=30
    )
    simulation_options = dict(
        n=20, p=12, alpha=0.25, sigma_A=0.05, sigma_B=0.05,
        bg_scale_A=0.03, bg_scale_B=0.03, signal_scale=2.0,
    )

    def test_recovers_single_planted_support(self):
        sample = generate_single_clique(
            SingleCliqueConfig(
                k=3, decoy_size=2, decoy_scale_A=0.1, decoy_scale_B=0.1,
                **self.simulation_options,
            ),
            seed=7,
        )
        result = fit_ossa(sample.A[:15], sample.B[:15], 3, **self.fit_options)
        support = result["hard_cliques"][0]
        self.assertEqual(set(support), set(sample.V_star))
        np.testing.assert_allclose(result["W"].T @ result["W"], np.eye(12), atol=1e-10)
        self.assertTrue(np.all(np.diff(result["objective_history"]) >= -1e-8))
        weights = result["soft_memberships"]
        self.assertTrue(np.all((weights >= 0) & (weights <= 1)))
        self.assertTrue(np.all(weights.sum(axis=1) <= 3 + 1e-10))

        aligned_fc = result["W"].T @ sample.A[15:] @ result["W"]
        ix = np.ix_(np.arange(5), support, support)
        expected = np.sum(aligned_fc[ix] * sample.B[15:][ix]) / 5
        observed = score_alignment(
            sample.A[15:], sample.B[15:], result["W"], result["hard_cliques"]
        )
        self.assertGreater(observed, 0)
        self.assertAlmostEqual(observed, expected)

    def test_recovers_multiple_planted_supports(self):
        sample = generate_multi_clique(
            MultiCliqueConfig(K=2, kappa=3, **self.simulation_options), seed=7
        )
        result = fit_ossa(
            sample.A[:15], sample.B[:15], 3, n_supports=2,
            overlap_penalty=25.0, **self.fit_options,
        )
        self.assertEqual(
            {frozenset(support) for support in result["hard_cliques"]},
            {frozenset(support) for support in sample.V_stars},
        )

    def test_fixed_rotation_is_identity_by_default(self):
        matrices = np.array([[[0, 1, 0], [1, 0, 0], [0, 0, 0]]])
        result = fit_ossa(
            matrices, matrices, 2, update_rotation=False, **self.fit_options
        )
        np.testing.assert_array_equal(result["W"], np.eye(3))
        self.assertEqual(set(result["hard_cliques"][0]), {0, 1})

    def test_recovers_two_functional_communities(self):
        affinity = np.full((6, 6), 0.1)
        affinity[:3, :3] = 1
        affinity[3:, 3:] = 1
        np.fill_diagonal(affinity, 0)
        labels = discover_communities(np.stack([affinity, -affinity]), 2, seed=7)
        np.testing.assert_array_equal(labels[:3], np.repeat(labels[0], 3))
        np.testing.assert_array_equal(labels[3:], np.repeat(labels[3], 3))
        self.assertNotEqual(labels[0], labels[3])

    def test_rejects_mismatched_or_invalid_matrices(self):
        fc = np.zeros((2, 3, 3))
        with self.assertRaisesRegex(ValueError, "matching subjects and nodes"):
            fit_ossa(fc, fc[:1], 2)
        with self.assertRaisesRegex(ValueError, "number of nodes"):
            fit_ossa(fc, fc, 4)
        fc[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            fit_ossa(fc, fc, 2)


if __name__ == "__main__":
    unittest.main()
