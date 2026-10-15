"""Visualizations for the railway cooperative-game prototype.

All functions save a PNG into results/figures and return the path, so they
can be called independently from the experiment driver or from tests.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.milp import MILPResult
from src.network import STATIONS, Segment, Service

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SERVICE_COLORS = {
    "Shatabdi_12027": "#9467bd",
    "Lalbagh_12607": "#1f77b4",
    "Mail_12657": "#d62728",
    "Passenger_56261": "#2ca02c",
    "MEMU_43401": "#ff7f0e",
}


def _station_positions(segments: List[Segment]) -> Dict[str, float]:
    pos = {STATIONS[0]: 0.0}
    cum = 0.0
    for seg in segments:
        cum += seg.length_km
        pos[seg.to_station] = cum
    return pos


def plot_network(services: Dict[str, Service], segments: List[Segment]) -> str:
    pos = _station_positions(segments)
    fig, ax = plt.subplots(figsize=(10, 2 + 0.6 * len(services)))

    for x in pos.values():
        ax.axvline(x, color="#cccccc", linewidth=1, zorder=0)
    for st, x in pos.items():
        ax.plot(x, 0, "o", color="black", zorder=3, markersize=8)
        ax.text(x, 0.15, st, ha="center", fontsize=10, fontweight="bold")

    for i, (name, svc) in enumerate(services.items()):
        y = -(i + 1) * 0.6
        xs = [pos[s] for s in svc.stops]
        color = SERVICE_COLORS.get(name, None)
        ax.plot(xs, [y] * len(xs), "-", color=color, linewidth=3, zorder=1)
        for x in xs:
            ax.plot(x, y, "o", color=color, markersize=6, zorder=2)
        ax.text(max(xs) + 5, y, f"{name} (cap {int(svc.capacity)}/train)", va="center", fontsize=9, color=color)

    ax.set_ylim(-(len(services) + 1) * 0.6, 0.6)
    ax.set_xlabel("Position along corridor (km)")
    ax.set_yticks([])
    ax.set_title("Chennai Central - Bangalore City Corridor and Candidate Services")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "network_services.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_segment_utilization(result: MILPResult, services: Dict[str, Service], segments: List[Segment]) -> str:
    labels, utilizations, colors = [], [], []
    for (s, seg_idx), load in sorted(result.segment_loads.items()):
        freq = result.frequencies[s]
        cap_total = services[s].capacity * freq
        util = (load / cap_total) if cap_total > 0 else 0.0
        seg = segments[seg_idx]
        labels.append(f"{s}\n{seg.from_station}-{seg.to_station}")
        utilizations.append(util * 100)
        colors.append(SERVICE_COLORS.get(s, "#888888"))

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(labels)), 5))
    bars = ax.bar(labels, utilizations, color=colors)
    ax.axhline(100, color="red", linestyle="--", linewidth=1, label="Capacity limit")
    ax.set_ylabel("Segment utilization (%)")
    ax.set_title("Segment Utilization under the Globally Optimal Plan")
    ax.set_ylim(0, max(110, max(utilizations) + 10 if utilizations else 110))
    plt.xticks(rotation=45, ha="right", fontsize=8)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "segment_utilization.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_coalition_values(v: Dict[Tuple[str, ...], float]) -> str:
    items = sorted(v.items(), key=lambda kv: (len(kv[0]), kv[1]))
    labels = ["{" + ",".join(c) + "}" if c else "{}" for c, _ in items]
    values = [val for _, val in items]
    colors = plt.cm.viridis(np.linspace(0, 1, len(items)))

    fig, ax = plt.subplots(figsize=(max(10, 0.35 * len(items)), 6))
    ax.bar(range(len(items)), values, color=colors)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("v(S): net system benefit")
    ax.set_title("Characteristic Function Value for All 32 Coalitions")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "coalition_values.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_shapley_values(phi: Dict[str, float]) -> str:
    names = list(phi.keys())
    values = [phi[n] for n in names]
    colors = [SERVICE_COLORS.get(n, "#888888") for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Shapley value (fair share of v(N))")
    ax.set_title("Shapley Value Attribution per Service")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "shapley_values.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_pairwise_complementarity(players: List[str], comp: Dict[Tuple[str, str], float]) -> str:
    n = len(players)
    matrix = np.full((n, n), np.nan)
    idx = {p: i for i, p in enumerate(players)}
    for (i, j), val in comp.items():
        matrix[idx[i], idx[j]] = val
        matrix[idx[j], idx[i]] = val

    fig, ax = plt.subplots(figsize=(7, 6))
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap="RdBu_r", vmin=-np.nanmax(np.abs(matrix)), vmax=np.nanmax(np.abs(matrix)))
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(players, rotation=45, ha="right")
    ax.set_yticklabels(players)
    for i in range(n):
        for j in range(n):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Complementarity c_ij")
    ax.set_title("Pairwise Service Complementarity")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "pairwise_complementarity.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_allocation_comparison(players: List[str], shapley: Dict[str, float],
                                least_core_alloc: Dict[str, float], lp_dual_alloc: Dict[str, float]) -> str:
    x = np.arange(len(players))
    width = 0.27

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, [shapley[p] for p in players], width, label="Shapley value", color="#1f77b4")
    ax.bar(x, [least_core_alloc[p] for p in players], width, label="Least-core allocation", color="#ff7f0e")
    ax.bar(x + width, [lp_dual_alloc[p] for p in players], width, label="LP-dual allocation", color="#2ca02c")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(players, rotation=20, ha="right")
    ax.set_ylabel("Allocated value (₹/day)")
    ax.set_title("Shapley vs. Least-Core vs. LP-Dual Allocation")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "allocation_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_strategy_comparison(comparison: Dict[str, Dict]) -> str:
    labels = list(comparison.keys())
    values = [comparison[k]["value"] for k in labels]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=["#ff7f0e", "#2ca02c", "#1f77b4"])
    for bar, label in zip(bars, labels):
        services_str = ",".join(comparison[label]["services"]) or "(none)"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Net system benefit")
    ax.set_title("Independent vs Greedy vs Globally Optimal Selection")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "strategy_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
