<div align="center">

# Orthogonal Sparse Subgraph Alignment<br>for Structure-Function Coupling in Brain Networks

**Official Python implementation · NeurIPS 2026**

[![NeurIPS 2026 Poster](https://img.shields.io/badge/NeurIPS_2026-Poster-5545A0)](https://rampageoppa.github.io/projects/ossa/)
[![arXiv:2611.14663](https://img.shields.io/badge/arXiv-2611.14663-B31B1B?logo=arxiv)](https://arxiv.org/abs/2611.14663)

[**🌐 Project page**](https://rampageoppa.github.io/projects/ossa/) &nbsp;·&nbsp; [**📄 Paper**](https://arxiv.org/abs/2611.14663) &nbsp;·&nbsp; [**🧪 Demo source**](src/ossa/demo.py) &nbsp;·&nbsp; [**📌 BibTeX**](#citation)

</div>

---

## 🧩 Overview

**OSSA** identifies compact sets of brain regions whose structural connectivity (SC) aligns with functional connectivity (FC) across subjects.

### 🎯 Single-core training objective

For a functional community with `p` nodes, let `Aᵢ` and `Bᵢ` be subject `i`'s FC and SC matrices, and let `S = diag(s)`. The optimizer solves the continuous relaxation in [Section 2.1 of the paper](https://arxiv.org/abs/2611.14663).

$$
\max_{W,s}\ J(W,s) = \sum_{i=1}^{n} \mathrm{tr}\bigl[(W^{\top} A_i W)(S B_i S)\bigr]
$$

$$
W^{\top}W = I_p,\qquad s \in [0,1]^p,\qquad \sum_{j=1}^{p} s_j \leq \kappa .
$$

The objective scores how well the rotated FC aligns with the structural connectivity weighted by the selected nodes. [The optimizer](src/ossa/optimizer.py) alternates between updating **the orthogonal rotation** `W` and updating **the node weights** `s`. The `κ` largest weights define the final support.

The **multiple-support extension** shares one rotation across subnetworks and penalizes overlap between their node weights.

## 📁 Repository structure

| Path | Contents |
| :--- | :--- |
| [`optimizer.py`](src/ossa/optimizer.py) | Projected block coordinate ascent for one or multiple supports |
| [`community.py`](src/ossa/community.py) | Spectral community discovery from training FC |
| [`fit.py`](src/ossa/fit.py) | Fit across restarts and score on held-out pairs |
| [`simulation.py`](src/ossa/simulation.py) | Synthetic generators with known supports |
| [`demo.py`](src/ossa/demo.py) | Runnable single- and multiple-support examples |
| [`tests/`](tests/) | Support recovery, orthogonality, and input validation |

## 🚀 Getting started

**Requirements:** Python 3.10+ and NumPy.

```bash
git clone https://github.com/rampageoppa/OSSA.git
cd OSSA
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Run either synthetic example, then run the tests.

```bash
python -m ossa.demo
python -m ossa.demo --mode multi
python -m unittest discover -s tests -v
```

Each demo prints the planted and recovered supports, the training objective, and the held-out alignment score.

## 🛠️ Fit to your own matrices

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

**Key parameters**

| Argument | Meaning |
| :--- | :--- |
| `k` | Number of nodes selected per support |
| `n_supports` | Number of supports fitted in one community. Defaults to one. |
| `overlap_penalty` | Penalty on overlap between supports. Tune on validation data. |
| `n_restarts` | Number of initializations. The best fit is selected by training objective. |

You can supply your own community partition by passing the corresponding node subsets. The held-out score averages alignment across test subjects using the training-fitted rotation and selected supports.

## 🗄️ Data availability

The demos generate synthetic inputs in memory. Participant data, cohort files, and experiment outputs are excluded from this repository. See the [paper](https://arxiv.org/abs/2611.14663) for the full simulation and cohort analyses.

<a id="citation"></a>

## ✒️ Citation

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

## 📜 License

Released under the [MIT License](LICENSE).
