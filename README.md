# Optimization-for-RL-692-UTennessee-
Repository for the course project: Optimization for RL (IE692)
# Policy Mirror Descent for Simplified Blackjack, GridWorld, and Quantum Error Correction

**IE 692: Optimization for Reinforcement Learning, University of Tennessee, Knoxville (Fall 2026)**
Team: Chac, Jishnu Mahmud, Raiyan Basher, Raisa Mashtura

This repository holds our course project. We study stochastic policy mirror descent (SPMD) with an
on-policy Monte Carlo critic on two small MDPs, a simplified Blackjack game and a GridWorld, and test
three enhancements to it. As a bonus, we move the same machinery onto a quantum problem: decoding
Google's real surface-code hardware data, posed as a GridWorld.

---

## What we did

### Baseline
- **SPMD (KL policy mirror descent):** π<sub>k+1</sub>(a|s) ∝ π<sub>k</sub>(a|s) · exp(−η Q̂<sub>k</sub>(s,a)).
- **Critic: truncated on-policy Monte Carlo (TOMC)** with the dynamic exploration time of Ju & Lan
  (arXiv:2512.06244). A batch ends once every pair with π<sub>k</sub>(a|s) ≥ 0.025 has been visited and
  followed by T = 100 costs. Pairs below the threshold get the pessimistic value 1/(1−γ) = 10.

### Enhancements
| | Enhancement | What it changes |
|---|---|---|
| **E1** | Expected TD evaluation | The critic. It replaces Monte Carlo returns with the target c + γ Σ<sub>a′</sub> π(a′\|s′) Q̂(s′,a′). |
| **E2** | Damped momentum in SPMD | The actor. It adds a damped momentum term to the KL update. |
| **E3** | All-pair exploration with count-based optimism | Data collection. Every (s,a) pair must be visited (no threshold, no pessimistic fallback), and the update uses Q̂ − b with a UCB-style bonus b(s,a) = β √(2 ln(2+N(s)) / (1+N(s,a))). |

### GridWorld protocol (shared by the baseline and E1–E3)
- γ = 0.9, minimum history T = 100, budget **15M observations**, **5 seeds**.
- η is tuned with **10% of the budget** on the first seed, over {0.1, 1, 2, 10}. The baseline picks η = 1.
- Each learned policy is evaluated **exactly** with the Bellman equation. We report the optimality gap
  at state 0 and the gap averaged over all 100 states, as mean ± 95% CI over seeds. The model is
  used for evaluation only.

**GridWorld results (15M observations, 5 seeds)**

| Method | Gap at state 0 | Mean gap, all states |
|---|---|---|
| Baseline: SPMD + TOMC | 0.0553 ± 0.0147 | 0.1003 ± 0.0139 |
| E1: Expected TD (tuned) | ≈ 0.0000 | 0.0058 |
| E2: damped momentum (λ = 0.5) | 0.0450 (within noise of the baseline) | — |
| E3 control: all pairs, β = 0 | 0.0862 ± 0.0272 | 0.1524 ± 0.0143 |
| E3: all pairs + bonus (η = 1, β = 1) | 0.0649 ± 0.0142 | 0.1384 ± 0.0091 |

E1 gives the largest improvement. In E3, the bonus improves on the β = 0 control for every seed, but
it does not beat the baseline: without the threshold, each batch has to wait for its rarest pair, so
far fewer policy updates fit in the budget.

### Bonus: surface-code decoding as a GridWorld (real quantum hardware data)
- **Problem.** A quantum error-correcting code shows only which parity checks fired (the
  *syndrome*). A decoder must guess a correction. We pose this as a single-agent GridWorld:
  - **State:** the 8-bit detector pattern (256 states).
  - **Action:** flip one of 23 error edges.
  - **Transition:** deterministic, s′ = s ⊕ mask(a).
  - **Cost:** the edge's log-likelihood ratio log((1−p)/p), normalised.
  - **Goal:** the empty syndrome.
  - The minimum-cost path is the most likely error, so the optimal policy is a maximum-likelihood
    matching decoder.
- **Data.** Google's 2022 Sycamore experiment, distance-3 surface code, 1 round, Z basis
  (`surface_code_bZ_d3_r01_center_3_5`): **50,000 real shots**.
- **Method.** The same Expected TD-PMD algorithm as GridWorld E1 (SPMD + E1). It uses two-fold
  cross-fitting (train on even shots, test on odd, and the reverse), so every shot is decoded by a
  policy that never saw it. Edge costs come from Google's own `pij_from_even_for_odd.dem` /
  `pij_from_odd_for_even.dem` error models.
- **Checks.** Exact value iteration serves as the reference, and its decisions agree with Google's
  PyMatching predictions on 100% of shots.

**Logical error rate on all 50,000 real shots (s.e. ≈ 0.0006)**

| Decoder | Logical errors | Rate |
|---|---|---|
| No decoding | — | 0.0850 |
| Google: PyMatching / correlated matching / tensor network | 819 | 0.0164 |
| Google: belief matching | 821 | 0.0164 |
| Exact optimum, γ = 0.9 (the RL target) | 821 | 0.0164 |
| **Ours: Expected TD-PMD, final greedy policy** | **821** | **0.0164** |

The learned tabular decoder matches its exact target and Google's production decoders to within two
shots.

---

## Repository contents

| File | What it is |
|---|---|
| `blackjack_*.ipynb` | Simplified Blackjack: baseline and enhancements. Each notebook is self-contained. |
| `gridworld_results.py` | Shared helper module for the GridWorld notebooks (seeds, exact evaluation, `save_runs`/`load_runs`, CIs, plot style). |
| `gridworld_final_baseline.ipynb` | GridWorld baseline (SPMD + TOMC), with η tuning. Writes `results/gridworld_baseline.npz`. |
| `gridworld_enhancement1.ipynb` | E1: Expected TD critic. |
| `gridworld_enhancement2.ipynb` | E2: damped momentum. |
| `gridworld_enhancement3.ipynb` | E3: all-pair exploration + count-based optimism. |
| `qec_enhancement1_spmd_e1.ipynb` | Bonus: SPMD + E1 surface-code decoder on Google's hardware data. |
| `results/`, `figures/` | Created by the GridWorld notebooks (saved runs as `.npz`, figures as `.png`). |

---

## How to run

### Setup
Python 3.9 or newer.
```bash
pip install numpy pandas matplotlib scipy numba jupyter
# optional, enables one extra cross-check in the QEC notebook:
pip install stim pymatching
```

### 1. Blackjack
Each Blackjack notebook is self-contained. Open it and run all cells.

### 2. GridWorld (order matters)
Keep `gridworld_results.py` in the same folder as the notebooks.
1. Run `gridworld_final_baseline.ipynb` **first**. It creates `results/gridworld_baseline.npz`, which
   the enhancement notebooks read for γ, T, the budget, the tuned η, and the baseline curves.
2. Then run `gridworld_enhancement1.ipynb`, `gridworld_enhancement2.ipynb`, and
   `gridworld_enhancement3.ipynb`, in any order.

On Colab, upload `gridworld_results.py`, plus `results/gridworld_baseline.npz` inside a `results/`
folder, before running an enhancement notebook. E3 installs Numba if it is missing; its sampler
compiles on first use. Because of Numba, E3's wall-clock times are not comparable with the other
notebooks.

### 3. Quantum bonus (QEC decoding)
1. We have used `google_qec3v5_experiment_data.zip` (315 MB, CC-BY 4.0) from
   [Zenodo record 6804040](https://zenodo.org/records/6804040) as provided in this repository.
4. Run `qec_enhancement1_spmd_e1.ipynb` with the dataset file on the same root folder. It takes about 2–3 minutes and ~0.5 GB of
   memory. Outputs go to `qec_enhancement1_B50_K400000_outputs/`.

To decode a different patch or basis, change `EXPERIMENT` in the settings cell (for example,
`surface_code_bX_d3_r01_center_3_5`).

---

## References
- C. Ju and G. Lan, *Auto-exploration for online reinforcement learning*, arXiv:2512.06244.
- P. Auer, N. Cesa-Bianchi, P. Fischer, *Finite-time analysis of the multiarmed bandit problem*,
  Machine Learning 47 (2002).
- P. Andreasson, J. Johansson, S. Liljestrand, M. Granath, *Quantum error correction for the toric
  code using deep reinforcement learning*, Quantum 3, 183 (2019), arXiv:1811.12338.
- Google Quantum AI, *Suppressing quantum errors by scaling a surface code logical qubit*,
  Nature 614, 676 (2023). Data: Zenodo 6804040.
