# -*- coding: utf-8 -*-
"""Graphical abstract for HSJ submission (matplotlib only)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
OUT = ROOT / "paper" / "HSJ_graphical_abstract.png"
BASIN = ROOT / "results" / "tier2" / "basin_compare.csv"
QPF = ROOT / "results" / "tier2" / "qpf_ladder.csv"


def main():
    b = pd.read_csv(BASIN)
    q = pd.read_csv(QPF)
    fig = plt.figure(figsize=(11, 4.2), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.0, 1.1], wspace=0.32,
                          left=0.05, right=0.98, top=0.80, bottom=0.16)

    ax0 = fig.add_subplot(gs[0])
    ax0.set_xlim(0, 10)
    ax0.set_ylim(0, 10)
    ax0.axis("off")
    ax0.set_title("A. Two mid-Atlantic outlets", fontsize=10, fontweight="bold", loc="left")
    ax0.text(5, 7.2, "Potomac → Washington, D.C.", ha="center", fontsize=9)
    ax0.text(5, 5.8, "James → Richmond, VA", ha="center", fontsize=9)
    ax0.text(5, 3.8, "Same protocol:\nrouting baseline · ML · ablation\nQPF ladder · flood CSI",
             ha="center", fontsize=8, color="#333333",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#f3f6fa", edgecolor="#ccd"))

    ax1 = fig.add_subplot(gs[1])
    sub = b[b["horizon"] == 1]
    x = np.arange(len(sub))
    w = 0.25
    ax1.bar(x - w, sub["routing_NSE"], w, label="Routing", color="#4C78A8", edgecolor="k", lw=0.4)
    ax1.bar(x, sub["xgboost_NSE"], w, label="XGB", color="#F58518", edgecolor="k", lw=0.4)
    ax1.bar(x + w, sub["lstm_attention_NSE"], w, label="Attn", color="#54A24B", edgecolor="k", lw=0.4)
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.title() for s in sub["basin"]])
    ax1.set_ylim(0.7, 1.0)
    ax1.set_ylabel("1-day NSE")
    ax1.set_title("B. Protocol transfer", fontsize=10, fontweight="bold", loc="left")
    ax1.legend(fontsize=7, ncol=3, loc="lower right")
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = fig.add_subplot(gs[2])
    x = np.arange(len(q))
    w = 0.2
    for i, (col, lab, c) in enumerate([
        ("obs_NSE", "Obs", "#9E9E9E"),
        ("qpf_NSE", "GFS QPF", "#F58518"),
        ("oracle_NSE", "Oracle", "#E45756"),
    ]):
        ax2.bar(x + (i - 1) * w, q[col], w, label=lab, color=c, edgecolor="k", lw=0.4)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{h}d" for h in q["horizon"]])
    ax2.set_ylabel("NSE (2024)")
    ax2.set_ylim(-0.1, 1.05)
    ax2.set_title("C. Real QPF vs oracle", fontsize=10, fontweight="bold", loc="left")
    ax2.legend(fontsize=7)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.axhline(0, color="k", lw=0.6)

    fig.suptitle(
        "Information value of gauges & precip foresight for mid-Atlantic streamflow forecasts",
        fontsize=11, fontweight="bold", y=0.96,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
