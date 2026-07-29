# -*- coding: utf-8 -*-
"""
HSJ upgrade experiments (respond to JHRS desk-reject on novelty):

1) Calibrated lagged-upstream linear routing baseline (physical/simple hydrologic reference)
2) Oracle future-precipitation ceiling for LSTM-Attention (information limit / QPF value)
3) Flood-threshold decision skill (POD / FAR / CSI) for operational relevance

Outputs -> results/hsj_upgrade/ and figures/hsj_upgrade/
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from metrics import compute_all
from run_enhanced_analysis import prepare_arrays, run_dl, run_xgb, set_seed
from run_experiment import (
    FEATURE_COLS, HORIZONS, SEQ_LEN, TARGET_COL, TRAIN_END, VAL_END,
    load_data, make_sequences, temporal_split, inverse_transform_y,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
FIG = ROOT / "figures" / "hsj_upgrade"
RES = ROOT / "results" / "hsj_upgrade"
FIG.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)


def contingency_scores(obs, sim, thr):
    obs = np.asarray(obs, float)
    sim = np.asarray(sim, float)
    o = obs >= thr
    s = sim >= thr
    hits = int(np.sum(o & s))
    misses = int(np.sum(o & ~s))
    false_alarms = int(np.sum(~o & s))
    correct_neg = int(np.sum(~o & ~s))
    pod = hits / (hits + misses + 1e-12)
    far = false_alarms / (hits + false_alarms + 1e-12)
    csi = hits / (hits + misses + false_alarms + 1e-12)
    return {
        "threshold_cfs": float(thr),
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_negatives": correct_neg,
        "POD": float(pod),
        "FAR": float(far),
        "CSI": float(csi),
        "n_events_obs": int(np.sum(o)),
    }


def lagged_upstream_baseline(df: pd.DataFrame, horizon: int):
    """
    Simple forecast: Q_d(t+h) ~ a0 + a1 Q_d(t) + a2 Q_u1(t) + a3 Q_u2(t)
    Optionally search lag L on upstream gauges (use Q_u(t-L)) with L in 0..3.
    Calibrate on train only; evaluate on test.
    """
    train, val, test = temporal_split(df)
    cal = pd.concat([train, val]).reset_index(drop=True)

    best = None
    for lag in range(0, 4):
        rows = []
        for i in range(lag, len(cal) - horizon):
            rows.append({
                "y": cal[TARGET_COL].iloc[i + horizon],
                "qd": cal[TARGET_COL].iloc[i],
                "qu1": cal["Q_01636500"].iloc[i - lag],
                "qu2": cal["Q_01638480"].iloc[i - lag],
            })
        dcal = pd.DataFrame(rows).dropna()
        X = dcal[["qd", "qu1", "qu2"]].values
        y = dcal["y"].values
        reg = LinearRegression().fit(X, y)
        pred = reg.predict(X)
        score = compute_all(y, pred)["NSE"]
        if best is None or score > best["cal_NSE"]:
            best = {"lag": lag, "coef": reg, "cal_NSE": float(score)}

    # build aligned test predictions with same indexing as DL pack
    # Use full series for index alignment
    lag = best["lag"]
    reg = best["coef"]
    full = df.reset_index(drop=True)
    y_list, p_list, date_list = [], [], []
    test_start = test["date"].min()
    for i in range(max(lag, SEQ_LEN), len(full) - horizon):
        dt = full["date"].iloc[i + horizon]
        if dt < test_start:
            continue
        x = np.array([[
            full[TARGET_COL].iloc[i],
            full["Q_01636500"].iloc[i - lag],
            full["Q_01638480"].iloc[i - lag],
        ]])
        y_list.append(full[TARGET_COL].iloc[i + horizon])
        p_list.append(float(reg.predict(x)[0]))
        date_list.append(dt)

    y_obs = np.array(y_list, float)
    y_pred = np.array(p_list, float)
    metrics = compute_all(y_obs, y_pred)
    return {
        "lag_days": lag,
        "coefficients": {
            "intercept": float(reg.intercept_),
            "Q_target": float(reg.coef_[0]),
            "Q_01636500": float(reg.coef_[1]),
            "Q_01638480": float(reg.coef_[2]),
        },
        "cal_NSE": best["cal_NSE"],
        "test_metrics": {k: float(v) for k, v in metrics.items()},
        "y_obs": y_obs,
        "y_pred": y_pred,
        "dates": np.array(date_list),
    }


def add_oracle_precip(df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, list[str]]:
    """Append perfect-foresight local precip leads 1..horizon (upper-bound experiment)."""
    out = df.copy()
    cols = list(FEATURE_COLS)
    for lead in range(1, horizon + 1):
        name = f"oracle_precip_lead{lead}"
        out[name] = out["precip_target"].shift(-lead)
        cols.append(name)
    out = out.dropna().reset_index(drop=True)
    return out, cols


def align_preds(dates_a, ya, pa, dates_b, yb, pb):
    """Inner-join two prediction series on dates."""
    da = pd.DataFrame({"date": pd.to_datetime(dates_a), "ya": ya, "pa": pa})
    db = pd.DataFrame({"date": pd.to_datetime(dates_b), "yb": yb, "pb": pb})
    m = da.merge(db, on="date", how="inner")
    return m["ya"].values, m["pa"].values, m["pb"].values, m["date"].values


def plot_routing_vs_ml(rows, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    hs = [r["horizon"] for r in rows]
    x = np.arange(len(hs))
    w = 0.2
    series = [
        ("Persistence", [r["persistence_NSE"] for r in rows], "#999999"),
        ("Lagged upstream LR", [r["routing_NSE"] for r in rows], "#4C78A8"),
        ("XGBoost", [r["xgb_NSE"] for r in rows], "#F58518"),
        ("LSTM-Attention", [r["attn_NSE"] for r in rows], "#54A24B"),
    ]
    for i, (lab, vals, c) in enumerate(series):
        ax.bar(x + (i - 1.5) * w, vals, w, label=lab, color=c, edgecolor="black", lw=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}d" for h in hs])
    ax.set_ylabel("Test NSE")
    ax.set_ylim(0, 1.05)
    ax.set_title("Physical/simple baseline vs ML (Potomac outlet)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_oracle_gain(rows, path):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    hs = [r["horizon"] for r in rows]
    x = np.arange(len(hs))
    w = 0.32
    obs_nse = [r["attn_obs_NSE"] for r in rows]
    ora_nse = [r["attn_oracle_NSE"] for r in rows]
    ax.bar(x - w / 2, obs_nse, w, label="Attention (obs only)", color="#54A24B", edgecolor="black", lw=0.4)
    ax.bar(x + w / 2, ora_nse, w, label="Attention + oracle precip", color="#E45756", edgecolor="black", lw=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}d" for h in hs])
    ax.set_ylabel("Test NSE")
    ax.set_ylim(0, 1.05)
    ax.set_title("Value of perfect-foresight precipitation (ceiling experiment)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_csi(rows, path):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    hs = [r["horizon"] for r in rows]
    x = np.arange(len(hs))
    w = 0.25
    for i, (key, lab, c) in enumerate([
        ("routing_CSI", "Lagged upstream LR", "#4C78A8"),
        ("xgb_CSI", "XGBoost", "#F58518"),
        ("attn_CSI", "LSTM-Attention", "#54A24B"),
    ]):
        ax.bar(x + (i - 1) * w, [r[key] for r in rows], w, label=lab, color=c, edgecolor="black", lw=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}d" for h in hs])
    ax.set_ylabel("CSI (Q ≥ P90 train)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Flood-threshold forecast skill (critical success index)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    df = load_data()
    train, _, _ = temporal_split(df)
    thr = float(np.nanpercentile(train[TARGET_COL].values, 90))

    comparison_rows = []
    oracle_rows = []
    flood_rows = []
    summary = {"threshold_P90_cfs": thr, "horizons": {}, "novelty_notes": [
        "Adds lagged-upstream linear routing as a hydrologic baseline beyond persistence.",
        "Adds oracle precipitation ceiling to quantify when QPF is required.",
        "Adds POD/FAR/CSI flood-threshold skill for operational relevance.",
    ]}

    for h in HORIZONS:
        print(f"\n=== Horizon {h}d ===")
        # --- routing baseline ---
        route = lagged_upstream_baseline(df, h)
        print(f"  routing lag={route['lag_days']}d test NSE={route['test_metrics']['NSE']:.3f}")

        # --- observation-driven ML (reuse prepare_arrays) ---
        pack = prepare_arrays(df, FEATURE_COLS, h)
        pers_m = compute_all(pack["y_obs"], pack["pers"])
        xgb_pred, _ = run_xgb(pack)
        xgb_m = compute_all(pack["y_obs"], xgb_pred)
        attn_pred, _ = run_dl(pack, "attn", seed=42, epochs=60)
        attn_m = compute_all(pack["y_obs"], attn_pred)
        print(f"  pers={pers_m['NSE']:.3f} xgb={xgb_m['NSE']:.3f} attn={attn_m['NSE']:.3f}")

        # align all predictors on the intersection of routing and ML dates
        base = pd.DataFrame({
            "date": pd.to_datetime(pack["dates"]),
            "y": pack["y_obs"],
            "pers": pack["pers"],
            "xgb": xgb_pred,
            "attn": attn_pred,
        })
        rt = pd.DataFrame({
            "date": pd.to_datetime(route["dates"]),
            "routing": route["y_pred"],
        })
        m = base.merge(rt, on="date", how="inner")
        yo = m["y"].values
        pr = m["routing"].values
        pa = m["attn"].values
        px = m["xgb"].values
        pp = m["pers"].values
        dts = m["date"].values
        route_m_al = compute_all(yo, pr)
        attn_m_al = compute_all(yo, pa)
        xgb_m_al = compute_all(yo, px)
        pers_m_al = compute_all(yo, pp)

        comparison_rows.append({
            "horizon": h,
            "persistence_NSE": pers_m_al["NSE"],
            "routing_NSE": route_m_al["NSE"],
            "xgb_NSE": xgb_m_al["NSE"],
            "attn_NSE": attn_m_al["NSE"],
            "attn_minus_routing": attn_m_al["NSE"] - route_m_al["NSE"],
        })

        # flood skill on aligned set
        flood = {
            "horizon": h,
            "routing": contingency_scores(yo, pr, thr),
            "xgb": contingency_scores(yo, px, thr),
            "attn": contingency_scores(yo, pa, thr),
            "pers": contingency_scores(yo, pp, thr),
        }
        flood_rows.append({
            "horizon": h,
            "routing_CSI": flood["routing"]["CSI"],
            "xgb_CSI": flood["xgb"]["CSI"],
            "attn_CSI": flood["attn"]["CSI"],
            "pers_CSI": flood["pers"]["CSI"],
            "attn_POD": flood["attn"]["POD"],
            "attn_FAR": flood["attn"]["FAR"],
        })

        # --- oracle precip ceiling ---
        df_ora, cols_ora = add_oracle_precip(df, h)
        pack_o = prepare_arrays(df_ora, cols_ora, h)
        attn_o_pred, _ = run_dl(pack_o, "attn", seed=42, epochs=60)
        # align oracle to obs-only attn dates
        y2, p_obs, p_ora, _ = align_preds(
            pack["dates"], pack["y_obs"], attn_pred,
            pack_o["dates"], pack_o["y_obs"], attn_o_pred,
        )
        m_obs = compute_all(y2, p_obs)
        m_ora = compute_all(y2, p_ora)
        print(f"  oracle attn NSE={m_ora['NSE']:.3f} (obs {m_obs['NSE']:.3f}, Δ={m_ora['NSE']-m_obs['NSE']:+.3f})")
        oracle_rows.append({
            "horizon": h,
            "attn_obs_NSE": m_obs["NSE"],
            "attn_oracle_NSE": m_ora["NSE"],
            "delta_NSE": m_ora["NSE"] - m_obs["NSE"],
        })

        summary["horizons"][str(h)] = {
            "routing": {
                "lag_days": route["lag_days"],
                "coefficients": route["coefficients"],
                "test_metrics": route["test_metrics"],
                "aligned_NSE": float(route_m_al["NSE"]),
            },
            "persistence_NSE": float(pers_m_al["NSE"]),
            "xgboost_NSE": float(xgb_m_al["NSE"]),
            "lstm_attention_NSE": float(attn_m_al["NSE"]),
            "attn_minus_routing_NSE": float(attn_m_al["NSE"] - route_m_al["NSE"]),
            "oracle_precip": {
                "obs_NSE": float(m_obs["NSE"]),
                "oracle_NSE": float(m_ora["NSE"]),
                "delta_NSE": float(m_ora["NSE"] - m_obs["NSE"]),
            },
            "flood_P90": flood,
        }

    plot_routing_vs_ml(comparison_rows, FIG / "fig_routing_vs_ml.png")
    plot_oracle_gain(oracle_rows, FIG / "fig_oracle_precip_ceiling.png")
    plot_csi(flood_rows, FIG / "fig_flood_csi.png")

    pd.DataFrame(comparison_rows).to_csv(RES / "routing_vs_ml.csv", index=False)
    pd.DataFrame(oracle_rows).to_csv(RES / "oracle_precip.csv", index=False)
    pd.DataFrame(flood_rows).to_csv(RES / "flood_csi.csv", index=False)
    (RES / "hsj_upgrade_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\nWrote", RES / "hsj_upgrade_summary.json")
    print("Figures in", FIG)


if __name__ == "__main__":
    main()
