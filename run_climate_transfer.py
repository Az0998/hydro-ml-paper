# -*- coding: utf-8 -*-
"""
Multi-climate protocol transfer: humid / snowmelt / semi-arid.
Answers: under what conditions is upstream information valuable?
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_tier2_upgrade import run_basin

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RES = ROOT / "results" / "climate_transfer"
FIG = ROOT / "figures" / "climate_transfer"
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

BASINS = {
    "potomac": {"climate": "humid_mid_atlantic", "path": DATA / "potomac_daily.csv"},
    "james": {"climate": "humid_mid_atlantic", "path": DATA / "james_daily.csv"},
    "willamette": {"climate": "humid_pacific_nw", "path": DATA / "willamette_daily.csv"},
    "animas": {"climate": "snowmelt_rockies", "path": DATA / "animas_daily.csv"},
    "verde": {"climate": "semiarid_southwest", "path": DATA / "verde_daily.csv"},
}


def plot_ablation_by_climate(df: pd.DataFrame, path: Path):
    sub = df[df["horizon"] == 1].copy()
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(sub))
    colors = {
        "humid_mid_atlantic": "#4C78A8",
        "humid_pacific_nw": "#54A24B",
        "snowmelt_rockies": "#72B7B2",
        "semiarid_southwest": "#E45756",
    }
    c = [colors.get(v, "#999") for v in sub["climate"]]
    ax.bar(x, sub["ablation_delta"], color=c, edgecolor="k", lw=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n({c})" for b, c in zip(sub["basin"], sub["climate"])], fontsize=7)
    ax.set_ylabel("1-day ablation ΔNSE (full − local)")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_title("Upstream information value across climate settings")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_routing_margin(df: pd.DataFrame, path: Path):
    sub = df[df["horizon"] == 1].copy()
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(sub))
    w = 0.25
    ax.bar(x - w, sub["routing_NSE"], w, label="Routing LR", color="#4C78A8", edgecolor="k", lw=0.4)
    ax.bar(x, sub["xgboost_NSE"], w, label="XGBoost", color="#F58518", edgecolor="k", lw=0.4)
    ax.bar(x + w, sub["lstm_attention_NSE"], w, label="Attn", color="#54A24B", edgecolor="k", lw=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["basin"].str.title(), fontsize=8)
    ax.set_ylabel("1-day test NSE")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("Routing vs ML across climate-zone basins")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    all_rows = []
    summaries = {}
    for name, meta in BASINS.items():
        path = meta["path"]
        if not path.exists():
            print(f"SKIP {name}: missing {path.name}")
            continue
        print(f"\n===== {name} ({meta['climate']}) =====")
        try:
            summary, rows = run_basin(name, path)
            for r in rows:
                r["climate"] = meta["climate"]
            all_rows.extend(rows)
            summaries[name] = {"climate": meta["climate"], **summary}
        except Exception as e:
            print(f"FAILED {name}: {e}")

    if not all_rows:
        raise SystemExit("No basins completed.")

    df = pd.DataFrame(all_rows)
    df.to_csv(RES / "climate_basin_compare.csv", index=False)
    plot_ablation_by_climate(df, FIG / "fig_ablation_by_climate.png")
    plot_routing_margin(df, FIG / "fig_routing_ml_by_climate.png")

    # condition summary
    d1 = df[df["horizon"] == 1]
    cond = {
        "interpretation": [
            "Large ablation ΔNSE => upstream gauges carry material next-day information.",
            "Small ML-minus-routing => skill is mostly linear routing/memory.",
            "Compare climates to state when information value holds.",
        ],
        "by_basin_1d": {
            r["basin"]: {
                "climate": r["climate"],
                "ablation_delta": float(r["ablation_delta"]),
                "attn_minus_routing": float(r["attn_minus_routing"]),
                "routing_NSE": float(r["routing_NSE"]),
                "attn_NSE": float(r["lstm_attention_NSE"]),
            }
            for _, r in d1.iterrows()
        },
    }
    out = {"basins": summaries, "condition_summary": cond}
    (RES / "climate_transfer_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nWrote", RES / "climate_transfer_summary.json")
    print(d1[["basin", "climate", "routing_NSE", "lstm_attention_NSE", "ablation_delta"]].to_string(index=False))


if __name__ == "__main__":
    main()
