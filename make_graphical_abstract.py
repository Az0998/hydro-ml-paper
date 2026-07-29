# -*- coding: utf-8 -*-
"""Graphical abstract for JHRS (matplotlib only; no generative AI imagery)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).parent
OUT = ROOT / "paper" / "JHRS_graphical_abstract.png"
RES = ROOT / "results" / "enhanced" / "enhanced_summary.json"


def main():
    data = json.loads(RES.read_text(encoding="utf-8")) if RES.exists() else {}
    m1 = data.get("horizons", {}).get("1", {}).get("metrics", {})
    models = ["Persistence", "XGBoost", "LSTM", "LSTM-Attention"]
    nse = [m1.get(m, {}).get("NSE", np.nan) for m in models]
    abl = data.get("ablation", {}).get("1", {})
    full = abl.get("full_upstream", {}).get("NSE", np.nan)
    loc = abl.get("local_only", {}).get("NSE", np.nan)

    fig = plt.figure(figsize=(10.5, 4.2), dpi=300)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.35,
                          left=0.05, right=0.98, top=0.82, bottom=0.18)

    # Panel A: network sketch
    ax0 = fig.add_subplot(gs[0])
    ax0.set_xlim(0, 10)
    ax0.set_ylim(0, 10)
    ax0.axis("off")
    ax0.set_title("A. Potomac gauge network", fontsize=10, fontweight="bold", loc="left")
    # simplified river path
    ax0.plot([1.5, 3.5, 5.5, 8.2], [7.5, 6.2, 4.8, 2.5], color="#1f4e79", lw=3, zorder=1)
    ax0.plot([2.2, 3.5], [8.6, 6.2], color="#2e75b6", lw=2.5, zorder=1)
    sites = [
        (2.2, 8.6, "Millville\n(upstream)"),
        (3.5, 6.2, "Harpers Ferry\n(upstream)"),
        (8.2, 2.5, "Washington, D.C.\n(target)"),
    ]
    for x, y, lab in sites:
        ax0.scatter([x], [y], s=90, c="#c00000", zorder=2)
        ax0.text(x + 0.35, y, lab, fontsize=7.5, va="center")
    ax0.text(5.0, 0.6, "Predict Q at D.C. for lead times 1 / 3 / 7 days",
             ha="center", fontsize=8, style="italic", color="#333333")

    # Panel B: 1-day NSE bars
    ax1 = fig.add_subplot(gs[1])
    labels = ["Pers.", "XGB", "LSTM", "Attn"]
    colors = ["#9e9e9e", "#4c78a8", "#f58518", "#54a24b"]
    bars = ax1.bar(labels, nse, color=colors, edgecolor="black", linewidth=0.6)
    ax1.set_ylim(0.7, 1.0)
    ax1.set_ylabel("NSE (1-day test)", fontsize=9)
    ax1.set_title("B. Short-range skill", fontsize=10, fontweight="bold", loc="left")
    ax1.tick_params(labelsize=8)
    ax1.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, nse):
        if np.isfinite(v):
            ax1.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=7.5)

    # Panel C: ablation
    ax2 = fig.add_subplot(gs[2])
    ax2.bar(["Full\nupstream", "Local\nonly"], [full, loc],
            color=["#54a24b", "#e45756"], edgecolor="black", linewidth=0.6)
    ax2.set_ylim(0.85, 0.96)
    ax2.set_ylabel("NSE (1-day Attn)", fontsize=9)
    ax2.set_title("C. Value of upstream gauges", fontsize=10, fontweight="bold", loc="left")
    ax2.tick_params(labelsize=8)
    ax2.grid(axis="y", alpha=0.3)
    for i, v in enumerate([full, loc]):
        if np.isfinite(v):
            ax2.text(i, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        "Upstream gauges improve 1-day Potomac forecasts; weekly skill remains limited",
        fontsize=11, fontweight="bold", y=0.96,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
