# Orthogonal Sparse Subgraph Alignment (OSSA)

Official Python implementation for *Orthogonal Sparse Subgraph Alignment for Structure-Function Coupling in Brain Networks* (NeurIPS 2026).

[Paper](https://arxiv.org/abs/2611.14663) · [Project page](https://rampageoppa.github.io/projects/ossa/) · [Citation](#citation)

OSSA identifies small sets of structurally connected brain regions whose structural connectivity aligns with functional connectivity across subjects. Within each functional community, the method jointly fits an orthogonal alignment and a sparse vector of node weights by projected block coordinate ascent. The largest weights define the selected nodes. The extension for multiple supports shares the alignment across supports and penalizes their overlap.

## Installation and quick start

Use Python 3.10 or newer. NumPy is the only runtime dependency.

```bash
git clone https://github.com/rampageoppa/OSSA.git
cd OSSA
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m ossa.demo
python -m ossa.demo --mode multi
python -m unittest discover -s tests -v
```

The demos generate paired matrices in memory, fit on a training subset, and print the planted and estimated supports and the held-out alignment. They are small examples of the methodology. The paper reports the full simulation and cohort experiments.

## Fit to your own matrices

Pass paired FC and SC arrays of shape `(n_subjects, n_nodes, n_nodes)` in the same subject and node order. The implementation symmetrizes each matrix. Use a separate training set for community discovery, rotation fitting, and support selection.

```python
import numpy as np

from ossa import discover_communities, fit_ossa, score_alignment

# fc_train, sc_train, fc_test, and sc_test are NumPy arrays with matching node order.
labels = discover_communities(fc_train, n_communities=3, seed=7)
nodes = np.flatnonzero(labels == 0)
k = 3
if len(nodes) < k:
    raise ValueError("The selected community contains fewer than k nodes.")

train_fc = fc_train[:, nodes][:, :, nodes]
train_sc = sc_train[:, nodes][:, :, nodes]
test_fc = fc_test[:, nodes][:, :, nodes]
test_sc = sc_test[:, nodes][:, :, nodes]

fit = fit_ossa(train_fc, train_sc, k=k, seed=7)
supports = fit["hard_cliques"]  # Indices within the selected community.
selected_nodes = [nodes[support] for support in supports]
held_out = score_alignment(test_fc, test_sc, fit["W"], supports)
```

If you already have functional communities, you can supply your own node subsets. For multiple supports in one community, call `fit_ossa(..., n_supports=2, overlap_penalty=25.0)`. Select the overlap penalty and support size on validation data when comparing configurations. The held-out score averages the objective over subjects using the training-fitted rotation and hardened supports, without the overlap penalty.

## Repository layout

- `src/ossa/optimizer.py` implements the projected block coordinate ascent optimizer used for the fits.
- `src/ossa/community.py` implements spectral community discovery from training FC.
- `src/ossa/fit.py` provides restart selection and held-out scoring.
- `src/ossa/simulation.py` contains the single and multiple planted-support generators.
- `src/ossa/demo.py` runs the small synthetic examples.
- `tests/` checks synthetic recovery, fitting and scoring, and orthogonality.

No participant data, cohort files, or experiment outputs are distributed. The examples and tests require no downloads beyond the software dependencies.

## Citation

```bibtex
@inproceedings{gao2026orthogonal,
  title         = {Orthogonal Sparse Subgraph Alignment for
                   Structure-Function Coupling in Brain Networks},
  author        = {Gao, Haonan and Ham, Daeyoung and Zhang, Yifei and
                   Tian, Xinyuan and Ding, Shengxian and Zhao, Yize and Li, Tianxi},
  booktitle     = {40th Conference on Neural Information Processing Systems},
  year          = {2026},
  eprint        = {2611.14663},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ME}
}
```
