# save as gridworld_results.py

"""Shared helpers for the GridWorld notebooks.

Every notebook saves its runs in the same format, so any notebook can load the
baseline (or another enhancement) and plot it on the same axes.

One run is a dictionary with
    seed      random seed of the run
    samples   [n] cumulative training observations used to obtain policy i
    values    [n, n_states] exact values V^{pi_i}(s) of policy i (Bellman solve)
    estimate  [n] the algorithm's own estimate of V^{pi_i}(report state); NaN if none
    seconds   wall-clock learning time, without the Bellman evaluation
plus any extra arrays a notebook wants to keep (for example visit counts).
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
SEEDS = (0, 1, 2, 3, 4)          # the same seeds in every notebook

# Two-sided 95% Student-t critical values, indexed by the number of seeds.
T_CRITICAL = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
              7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}

# Colors: the baseline is always neutral gray; enhancements take these in order.
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e0"
BASELINE_COLOR = "#6f6e69"
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")


# ---------------------------------------------------------------- exact values
def exact_values(policy, P, c, gamma):
    """Solve (I - gamma P_pi) V = c_pi for one policy. c is [state] or [state, action]."""
    P_pi = np.einsum("sa,san->sn", policy, P)
    c_pi = c if c.ndim == 1 else np.sum(policy * c, axis=1)
    return np.linalg.solve(np.eye(len(policy)) - gamma * P_pi, c_pi)


def optimal_values(P, c, gamma, tolerance=1e-13):
    """Value iteration for the minimum discounted cost V*. Used only as a reference line."""
    c_sa = c[:, None] if c.ndim == 1 else c
    values = np.zeros(P.shape[0])
    while True:
        new_values = (c_sa + gamma * P @ values).min(axis=1)
        if np.max(np.abs(new_values - values)) < tolerance:
            return new_values
        values = new_values


# ------------------------------------------------------------------ save / load
def save_runs(name, runs, **settings):
    """Write a list of runs and the experiment settings to results/<name>.npz."""
    RESULTS_DIR.mkdir(exist_ok=True)
    arrays = {f"setting_{key}": value for key, value in settings.items()}
    arrays["seeds"] = np.array([run["seed"] for run in runs])
    for run in runs:
        for key, value in run.items():
            if key != "seed":
                arrays[f"seed{run['seed']}_{key}"] = value
    path = RESULTS_DIR / f"{name}.npz"
    np.savez_compressed(path, **arrays)
    return path


def load_runs(name):
    """Return (runs, settings) saved by save_runs."""
    with np.load(RESULTS_DIR / f"{name}.npz", allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    settings = {key[len("setting_"):]: value[()] if value.ndim == 0 else value
                for key, value in data.items() if key.startswith("setting_")}
    runs = []
    for seed in data["seeds"]:
        prefix = f"seed{seed}_"
        run = {key[len(prefix):]: value[()] if value.ndim == 0 else value
               for key, value in data.items() if key.startswith(prefix)}
        runs.append({"seed": int(seed), **run})
    return runs, settings


# ------------------------------------------------------------- seed statistics
def sample_grid(max_samples, n_points=200, first=1_000):
    """Log-spaced observation counts on which runs with different batch lengths are compared."""
    return np.unique(np.round(np.geomspace(first, max_samples, n_points)).astype(np.int64))


def on_grid(run, grid, metric):
    """Value of metric for the latest policy obtained with at most `grid` observations.

    metric maps values [n, n_states] to one number per policy. A policy stays in
    use until the next update, so the curve is a step function of the samples.
    """
    index = np.searchsorted(run["samples"], grid, side="right") - 1
    return metric(run["values"])[index]


def mean_ci(curves):
    """Mean over seeds (axis 0) and the half-width of the 95% t confidence interval."""
    curves = np.asarray(curves, dtype=float)
    n_seeds = len(curves)
    half_width = T_CRITICAL[n_seeds] * curves.std(axis=0, ddof=1) / np.sqrt(n_seeds)
    return curves.mean(axis=0), half_width


def format_mean_ci(values, digits=4):
    mean, half_width = mean_ci(np.asarray(values)[:, None])
    return f"{mean[0]:.{digits}f} ± {half_width[0]:.{digits}f}"


# ------------------------------------------------------------------------ plots
def apply_plot_style():
    """One quiet style for every figure: recessive grid and axes, readable on a slide."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 12, "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelsize": 12,
        "text.color": INK, "axes.labelcolor": MUTED, "axes.edgecolor": GRID,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
        "legend.frameon": False, "legend.fontsize": 11,
        "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def plot_band(ax, grid, curves, color, label, linestyle="-"):
    """Draw the seed mean as a line and its 95% confidence interval as a light band."""
    mean, half_width = mean_ci(curves)
    ax.fill_between(grid, mean - half_width, mean + half_width,
                    color=color, alpha=0.18, linewidth=0)
    ax.plot(grid, mean, color=color, linestyle=linestyle, label=label)
    return mean


def save_figure(fig, name):
    """Save a PNG for the slides and a PDF for the report."""
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png")
    fig.savefig(FIGURES_DIR / f"{name}.pdf")
