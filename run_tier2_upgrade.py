# -*- coding: utf-8 -*-
"""
Tier-2 upgrade for HSJ:
1) Neighboring-basin replication on James River (same protocol)
2) Real GFS QPF (Open-Meteo previous runs) vs obs-only vs oracle on Potomac 2024 test

Outputs: results/tier2/ and figures/tier2/
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
from run_experiment import (
    DEVICE, SEQ_LEN, TARGET_COL,
    LSTMAttention, make_sequences, inverse_transform_y, train_dl_model, predict_dl,
    flatten_sequences,
)
from xgboost import XGBRegressor
import torch

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RES = ROOT / "results" / "tier2"
FIG = ROOT / "figures" / "tier2"
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# Chronological splits used in the original Potomac study
TRAIN_END = "2018-12-31"
VAL_END = "2020-12-31"
# For QPF experiment (QPF archive from 2024): retrain with later cutoffs
QPF_TRAIN_END = "2022-12-31"
QPF_VAL_END = "2023-12-31"
QPF_TEST_START = "2024-01-01"


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def feature_cols_for(df: pd.DataFrame) -> list[str]:
    cols = ["Q_target", "precip_target", "temp_target"]
    ups = sorted([c for c in df.columns if c.startswith("Q_") and c != "Q_target"])
    for q in ups:
        sid = q.replace("Q_", "")
        cols.append(q)
        p = f"precip_{sid}"
        if p in df.columns:
            cols.append(p)
    return cols


def temporal_split(df, train_end, val_end):
    train = df[df["date"] <= train_end].copy()
    val = df[(df["date"] > train_end) & (df["date"] <= val_end)].copy()
    test = df[df["date"] > val_end].copy()
    return train, val, test


def prepare(df, cols, horizon, train_end, val_end):
    train, val, test = temporal_split(df, train_end, val_end)
    if len(test) < SEQ_LEN + horizon + 30:
        raise RuntimeError(f"Test set too small: {len(test)}")
    feat_scaler = StandardScaler()
    y_scaler = StandardScaler()
    train_feat = feat_scaler.fit_transform(train[cols])
    val_feat = feat_scaler.transform(val[cols])
    y_scaler.fit(train[[TARGET_COL]])

    def build(feat_arr, target_series):
        ys = y_scaler.transform(target_series.values.reshape(-1, 1)).ravel()
        return make_sequences(feat_arr, ys, SEQ_LEN, horizon)

    X_tr, y_tr = build(train_feat, train[TARGET_COL])
    val_feat_ext = np.vstack([train_feat[-SEQ_LEN:], val_feat])
    val_y_ext = pd.concat([train[TARGET_COL].iloc[-SEQ_LEN:], val[TARGET_COL]]).reset_index(drop=True)
    X_va, y_va = build(val_feat_ext, val_y_ext)

    full_feat = np.vstack([train_feat, val_feat])
    full_y = pd.concat([train[TARGET_COL], val[TARGET_COL]]).reset_index(drop=True)
    X_tr_full, y_tr_full = build(full_feat, full_y)

    ext_df = pd.concat([val, test]).reset_index(drop=True)
    test_feat_ext = feat_scaler.transform(ext_df[cols].values)
    y_ext = y_scaler.transform(ext_df[TARGET_COL].values.reshape(-1, 1)).ravel()
    X_te, y_te_s = make_sequences(test_feat_ext, y_ext, SEQ_LEN, horizon)
    all_dates = ext_df["date"].iloc[SEQ_LEN + horizon - 1: SEQ_LEN + horizon - 1 + len(X_te)].values
    mask = all_dates >= test["date"].min()
    X_te, y_te_s, dates = X_te[mask], y_te_s[mask], all_dates[mask]
    y_obs = inverse_transform_y(y_scaler, y_te_s)
    ext_q = ext_df[TARGET_COL].values.astype(float)
    pers_idx = np.where(mask)[0] + SEQ_LEN + horizon - 1
    pers = ext_q[pers_idx - horizon]
    return {
        "X_tr": X_tr, "y_tr": y_tr, "X_va": X_va, "y_va": y_va,
        "X_tr_full": X_tr_full, "y_tr_full": y_tr_full,
        "X_te": X_te, "y_obs": y_obs, "dates": dates, "pers": pers,
        "y_scaler": y_scaler, "n_features": len(cols), "cols": cols,
        "train": train, "test": test,
    }


def run_models(pack, epochs=55):
    set_seed(42)
    # persistence
    pers_m = compute_all(pack["y_obs"], pack["pers"])
    # xgb
    xgb = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1,
    )
    xgb.fit(flatten_sequences(pack["X_tr_full"]), pack["y_tr_full"])
    xgb_pred = inverse_transform_y(pack["y_scaler"], xgb.predict(flatten_sequences(pack["X_te"])))
    xgb_m = compute_all(pack["y_obs"], xgb_pred)
    # attn
    model = LSTMAttention(pack["n_features"])
    model = train_dl_model(model, pack["X_tr"], pack["y_tr"], pack["X_va"], pack["y_va"], epochs=epochs)
    attn_pred = inverse_transform_y(pack["y_scaler"], predict_dl(model, pack["X_te"]))
    attn_m = compute_all(pack["y_obs"], attn_pred)
    return {
        "persistence": {k: float(v) for k, v in pers_m.items()},
        "xgboost": {k: float(v) for k, v in xgb_m.items()},
        "lstm_attention": {k: float(v) for k, v in attn_m.items()},
        "preds": {"pers": pack["pers"], "xgb": xgb_pred, "attn": attn_pred, "y": pack["y_obs"], "dates": pack["dates"]},
    }


def routing_baseline(df, horizon, train_end, val_end):
    train, val, test = temporal_split(df, train_end, val_end)
    cal = pd.concat([train, val]).reset_index(drop=True)
    ups = sorted([c for c in df.columns if c.startswith("Q_") and c != "Q_target"])
    if len(ups) < 2:
        ups = ups + ups  # pad
    u1, u2 = ups[0], ups[1 if len(ups) > 1 else 0]
    best = None
    for lag in range(0, 4):
        rows = []
        for i in range(lag, len(cal) - horizon):
            rows.append({
                "y": cal[TARGET_COL].iloc[i + horizon],
                "qd": cal[TARGET_COL].iloc[i],
                "qu1": cal[u1].iloc[i - lag],
                "qu2": cal[u2].iloc[i - lag],
            })
        dcal = pd.DataFrame(rows).dropna()
        reg = LinearRegression().fit(dcal[["qd", "qu1", "qu2"]], dcal["y"])
        nse = compute_all(dcal["y"].values, reg.predict(dcal[["qd", "qu1", "qu2"]]))["NSE"]
        if best is None or nse > best["cal_NSE"]:
            best = {"lag": lag, "reg": reg, "cal_NSE": float(nse), "u1": u1, "u2": u2}
    lag, reg, u1, u2 = best["lag"], best["reg"], best["u1"], best["u2"]
    full = df.reset_index(drop=True)
    y, p, dts = [], [], []
    t0 = test["date"].min()
    for i in range(max(lag, SEQ_LEN), len(full) - horizon):
        dt = full["date"].iloc[i + horizon]
        if dt < t0:
            continue
        x = np.array([[full[TARGET_COL].iloc[i], full[u1].iloc[i - lag], full[u2].iloc[i - lag]]])
        y.append(full[TARGET_COL].iloc[i + horizon])
        p.append(float(reg.predict(x)[0]))
        dts.append(dt)
    m = compute_all(np.array(y), np.array(p))
    return {"lag": lag, "metrics": {k: float(v) for k, v in m.items()}, "y": np.array(y), "p": np.array(p), "dates": np.array(dts)}


def ablation_local(df, cols, horizon, train_end, val_end, epochs=45):
    local = [c for c in cols if ("target" in c) or c == "Q_target"]
    # ensure precip/temp/Q local only
    local = [c for c in ["Q_target", "precip_target", "temp_target"] if c in df.columns]
    pack_full = prepare(df, cols, horizon, train_end, val_end)
    pack_loc = prepare(df, local, horizon, train_end, val_end)
    set_seed(42)
    m_full = LSTMAttention(pack_full["n_features"])
    m_full = train_dl_model(m_full, pack_full["X_tr"], pack_full["y_tr"], pack_full["X_va"], pack_full["y_va"], epochs=epochs)
    pred_f = inverse_transform_y(pack_full["y_scaler"], predict_dl(m_full, pack_full["X_te"]))
    m_loc = LSTMAttention(pack_loc["n_features"])
    m_loc = train_dl_model(m_loc, pack_loc["X_tr"], pack_loc["y_tr"], pack_loc["X_va"], pack_loc["y_va"], epochs=epochs)
    pred_l = inverse_transform_y(pack_loc["y_scaler"], predict_dl(m_loc, pack_loc["X_te"]))
    # align dates
    a = pd.DataFrame({"date": pd.to_datetime(pack_full["dates"]), "y": pack_full["y_obs"], "full": pred_f})
    b = pd.DataFrame({"date": pd.to_datetime(pack_loc["dates"]), "local": pred_l})
    m = a.merge(b, on="date")
    return {
        "full_NSE": float(compute_all(m["y"], m["full"])["NSE"]),
        "local_NSE": float(compute_all(m["y"], m["local"])["NSE"]),
        "delta_NSE": float(compute_all(m["y"], m["full"])["NSE"] - compute_all(m["y"], m["local"])["NSE"]),
    }


def add_future_precip(df, horizon, source: str):
    """
    source:
      oracle  — precip_target(t+lead)
      persist — precip_target(t) for all leads
      qpf     — GFS previous-run precip at valid time t+lead when available;
                persistence fill before 2024 (QPF archive starts 2024)
    """
    out = df.copy()
    extra = []
    for lead in range(1, horizon + 1):
        name = f"foresight_precip_lead{lead}"
        if source == "oracle":
            out[name] = out["precip_target"].shift(-lead)
        elif source == "persist":
            out[name] = out["precip_target"]
        elif source == "qpf":
            col = f"qpf_lead{lead}" if f"qpf_lead{lead}" in out.columns else (
                "qpf_lead7" if lead >= 7 and "qpf_lead7" in out.columns else (
                    "qpf_lead3" if lead >= 3 and "qpf_lead3" in out.columns else "qpf_lead1"
                )
            )
            if col not in out.columns:
                raise KeyError("QPF columns missing")
            q = out[col].shift(-lead)
            out[name] = q.fillna(out["precip_target"])
        else:
            raise ValueError(source)
        extra.append(name)
    need = ["Q_target", "precip_target"] + extra
    out = out.dropna(subset=need).reset_index(drop=True)
    return out, extra


def run_basin(name: str, path: Path, horizons=(1, 3, 7)):
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    # Keep original test window comparable (through 2023); 2024 reserved for QPF study
    df = df[df["date"] <= "2023-12-31"].reset_index(drop=True)
    cols = feature_cols_for(df)
    out = {"basin": name, "n_days": len(df), "features": cols, "horizons": {}}
    rows = []
    for h in horizons:
        print(f"\n[{name}] horizon={h}d")
        pack = prepare(df, cols, h, TRAIN_END, VAL_END)
        models = run_models(pack)
        route = routing_baseline(df, h, TRAIN_END, VAL_END)
        abl = ablation_local(df, cols, h, TRAIN_END, VAL_END)
        # align routing NSE to model dates
        base = pd.DataFrame({"date": pd.to_datetime(pack["dates"]), "y": pack["y_obs"],
                             "attn": models["preds"]["attn"], "xgb": models["preds"]["xgb"],
                             "pers": models["preds"]["pers"]})
        rt = pd.DataFrame({"date": pd.to_datetime(route["dates"]), "routing": route["p"]})
        m = base.merge(rt, on="date", how="inner")
        route_nse = float(compute_all(m["y"], m["routing"])["NSE"])
        attn_nse = float(compute_all(m["y"], m["attn"])["NSE"])
        rec = {
            "persistence_NSE": float(compute_all(m["y"], m["pers"])["NSE"]),
            "routing_NSE": route_nse,
            "xgboost_NSE": float(compute_all(m["y"], m["xgb"])["NSE"]),
            "lstm_attention_NSE": attn_nse,
            "attn_minus_routing": attn_nse - route_nse,
            "ablation": abl,
            "routing_lag": route["lag"],
        }
        out["horizons"][str(h)] = rec
        rows.append({"basin": name, "horizon": h, **{k: rec[k] for k in
                     ["persistence_NSE", "routing_NSE", "xgboost_NSE", "lstm_attention_NSE", "attn_minus_routing"]},
                     "ablation_delta": abl["delta_NSE"]})
        print(f"  pers={rec['persistence_NSE']:.3f} route={route_nse:.3f} xgb={rec['xgboost_NSE']:.3f} "
              f"attn={attn_nse:.3f} ablΔ={abl['delta_NSE']:.3f}")
    return out, rows


def run_qpf_experiment(horizons=(1, 3, 7)):
    path = DATA / "potomac_daily_with_qpf.csv"
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df = df[df["date"] <= "2024-12-31"].copy()
    base_cols = feature_cols_for(df)
    summary = {
        "train_end": QPF_TRAIN_END,
        "val_end": QPF_VAL_END,
        "test": "2024",
        "qpf_fill_note": "Pre-2024 foresight uses persistence fill; 2024 test uses real GFS previous-run QPF.",
        "horizons": {},
    }
    rows = []
    for h in horizons:
        print(f"\n[QPF] horizon={h}d")
        configs = {}
        for source in ["obs", "persist", "qpf", "oracle"]:
            if source == "obs":
                dsub = df.dropna(subset=base_cols).reset_index(drop=True)
                cols = base_cols
            else:
                dsub, extra = add_future_precip(df, h, source)
                cols = base_cols + extra
            pack = prepare(dsub, cols, h, QPF_TRAIN_END, QPF_VAL_END)
            mask = pd.to_datetime(pack["dates"]) >= pd.Timestamp(QPF_TEST_START)
            if mask.sum() < 50:
                print(f"  skip {source}: few 2024 samples {mask.sum()}")
                continue
            pack2 = dict(pack)
            pack2["X_te"] = pack["X_te"][mask]
            pack2["y_obs"] = pack["y_obs"][mask]
            pack2["dates"] = pack["dates"][mask]
            pack2["pers"] = pack["pers"][mask]
            set_seed(42)
            model = LSTMAttention(pack2["n_features"])
            model = train_dl_model(model, pack["X_tr"], pack["y_tr"], pack["X_va"], pack["y_va"], epochs=55)
            pred = inverse_transform_y(pack["y_scaler"], predict_dl(model, pack2["X_te"]))
            met = compute_all(pack2["y_obs"], pred)
            configs[source] = {k: float(v) for k, v in met.items()}
            print(f"  {source}: NSE={met['NSE']:.3f} n={int(mask.sum())}")
        summary["horizons"][str(h)] = configs
        rows.append({
            "horizon": h,
            "obs_NSE": configs.get("obs", {}).get("NSE"),
            "persist_NSE": configs.get("persist", {}).get("NSE"),
            "qpf_NSE": configs.get("qpf", {}).get("NSE"),
            "oracle_NSE": configs.get("oracle", {}).get("NSE"),
        })
    return summary, rows


def plot_basin_compare(rows, path):
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, h in zip(axes, [1, 3, 7]):
        sub = df[df["horizon"] == h]
        x = np.arange(len(sub))
        w = 0.2
        for i, (col, lab, c) in enumerate([
            ("routing_NSE", "Routing", "#4C78A8"),
            ("xgboost_NSE", "XGB", "#F58518"),
            ("lstm_attention_NSE", "Attn", "#54A24B"),
        ]):
            ax.bar(x + (i - 1) * w, sub[col], w, label=lab, color=c, edgecolor="k", lw=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["basin"].str.title())
        ax.set_title(f"{h}-day NSE")
        ax.grid(True, axis="y", alpha=0.3)
        if h == 1:
            ax.legend(fontsize=8)
            ax.set_ylabel("Test NSE")
    fig.suptitle("Protocol transfer: Potomac vs James River", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_qpf_ladder(rows, path):
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    x = np.arange(len(df))
    w = 0.2
    series = [
        ("obs_NSE", "Obs only", "#9E9E9E"),
        ("persist_NSE", "Persist precip", "#4C78A8"),
        ("qpf_NSE", "GFS QPF", "#F58518"),
        ("oracle_NSE", "Oracle precip", "#E45756"),
    ]
    for i, (col, lab, c) in enumerate(series):
        ax.bar(x + (i - 1.5) * w, df[col], w, label=lab, color=c, edgecolor="k", lw=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}d" for h in df["horizon"]])
    ax.set_ylabel("Test NSE (2024)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Potomac: observation-only vs real GFS QPF vs oracle ceiling")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    pot_path = DATA / "potomac_daily.csv"
    james_path = DATA / "james_daily.csv"
    if not pot_path.exists() or not james_path.exists() or not (DATA / "potomac_daily_with_qpf.csv").exists():
        raise SystemExit("Run download_extended.py first.")

    all_rows = []
    pot_sum, pot_rows = run_basin("potomac", pot_path)
    james_sum, james_rows = run_basin("james", james_path)
    all_rows.extend(pot_rows)
    all_rows.extend(james_rows)

    qpf_sum, qpf_rows = run_qpf_experiment()

    plot_basin_compare(all_rows, FIG / "fig_potomac_james_transfer.png")
    plot_qpf_ladder(qpf_rows, FIG / "fig_real_qpf_ladder.png")

    out = {
        "potomac": pot_sum,
        "james": james_sum,
        "qpf_2024": qpf_sum,
        "notes": [
            "James River replication uses identical chronological split and model protocol.",
            "Real QPF from Open-Meteo GFS previous-runs (hourly precip aggregated to daily).",
            "QPF experiment uses train<=2022 / val 2023 / test 2024 because QPF archive starts 2024.",
        ],
    }
    (RES / "tier2_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    pd.DataFrame(all_rows).to_csv(RES / "basin_compare.csv", index=False)
    pd.DataFrame(qpf_rows).to_csv(RES / "qpf_ladder.csv", index=False)
    print("\nWrote", RES / "tier2_summary.json")


if __name__ == "__main__":
    main()
