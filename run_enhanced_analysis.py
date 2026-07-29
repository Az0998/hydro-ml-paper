# -*- coding: utf-8 -*-
"""
Enhanced analysis for JHRS manuscript:
- Re-evaluate Persistence / XGBoost / LSTM / LSTM-Attention at 1/3/7-day horizons
- 3 random seeds for DL uncertainty bands
- Upstream ablation
- Stratified / seasonal / peak-event diagnostics
- Permutation importance on LSTM-Attention (1-day)
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
import torch
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from metrics import compute_all
from metrics_ext import compute_extended
from run_experiment import (
    DEVICE, FEATURE_COLS, HORIZONS, SEQ_LEN, TARGET_COL, TRAIN_END, VAL_END,
    LSTMAttention, LSTMModel, load_data, make_sequences, temporal_split,
    inverse_transform_y, train_dl_model, flatten_sequences, predict_dl,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
FIG = ROOT / "figures" / "enhanced"
RES = ROOT / "results" / "enhanced"
SEEDS = [42, 123, 7]


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_arrays(df, feature_cols, horizon):
    train, val, test = temporal_split(df)
    feat_scaler = StandardScaler()
    y_scaler = StandardScaler()
    train_feat = feat_scaler.fit_transform(train[feature_cols])
    val_feat = feat_scaler.transform(val[feature_cols])
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
    test_feat_ext = feat_scaler.transform(ext_df[feature_cols].values)
    y_ext = y_scaler.transform(ext_df[TARGET_COL].values.reshape(-1, 1)).ravel()
    X_te, y_te_s = make_sequences(test_feat_ext, y_ext, SEQ_LEN, horizon)
    all_dates = ext_df["date"].iloc[SEQ_LEN + horizon - 1: SEQ_LEN + horizon - 1 + len(X_te)].values
    mask = all_dates >= test["date"].min()
    X_te, y_te_s, dates = X_te[mask], y_te_s[mask], all_dates[mask]
    y_obs = inverse_transform_y(y_scaler, y_te_s)

    # persistence
    ext_q = ext_df[TARGET_COL].values.astype(float)
    pers_idx = np.where(mask)[0] + SEQ_LEN + horizon - 1
    pers = ext_q[pers_idx - horizon]

    return {
        "X_tr": X_tr, "y_tr": y_tr, "X_va": X_va, "y_va": y_va,
        "X_tr_full": X_tr_full, "y_tr_full": y_tr_full,
        "X_te": X_te, "y_obs": y_obs, "dates": dates, "pers": pers,
        "y_scaler": y_scaler, "n_features": len(feature_cols),
        "feature_cols": feature_cols,
    }


def run_xgb(pack):
    model = XGBRegressor(
        n_estimators=350, max_depth=6, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1,
    )
    model.fit(flatten_sequences(pack["X_tr_full"]), pack["y_tr_full"])
    pred = inverse_transform_y(pack["y_scaler"], model.predict(flatten_sequences(pack["X_te"])))
    return pred, model


def run_dl(pack, kind: str, seed: int, epochs=70):
    set_seed(seed)
    n_f = pack["n_features"]
    model = LSTMAttention(n_f) if kind == "attn" else LSTMModel(n_f)
    model = train_dl_model(model, pack["X_tr"], pack["y_tr"], pack["X_va"], pack["y_va"], epochs=epochs)
    pred = inverse_transform_y(pack["y_scaler"], predict_dl(model, pack["X_te"]))
    return pred, model


def permutation_importance(pack, model, n_repeats=3):
    """Feature-group permutation on test set (sequence-level shuffle within feature)."""
    X = pack["X_te"].copy()
    y = pack["y_obs"]
    base_pred = inverse_transform_y(pack["y_scaler"], predict_dl(model, X))
    base_nse = compute_all(y, base_pred)["NSE"]
    cols = pack["feature_cols"]
    scores = {}
    rng = np.random.default_rng(0)
    for j, name in enumerate(cols):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            # permute time axis independently per sample for feature j
            for i in range(Xp.shape[0]):
                Xp[i, :, j] = rng.permutation(Xp[i, :, j])
            pred = inverse_transform_y(pack["y_scaler"], predict_dl(model, Xp))
            drops.append(base_nse - compute_all(y, pred)["NSE"])
        scores[name] = float(np.mean(drops))
    return {"baseline_NSE": base_nse, "delta_NSE": scores}


def plot_seasonal(metrics_by_model, horizon, path):
    seasons = ["DJF", "MAM", "JJA", "SON"]
    models = list(metrics_by_model.keys())
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(seasons))
    w = 0.18
    colors = {"Persistence": "#999", "XGBoost": "#E76F51", "LSTM": "#40916C", "LSTM-Attention": "#1B4332"}
    for i, m in enumerate(models):
        vals = [metrics_by_model[m]["seasonal"].get(s, np.nan) for s in seasons]
        ax.bar(x + i * w, vals, w, label=m, color=colors.get(m, "#333"))
    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels(seasons)
    ax.set_ylabel("NSE")
    ax.set_ylim(-0.2, 1.05)
    ax.set_title(f"Seasonal NSE — {horizon}-day ahead")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close()


def plot_importance(imp, path):
    items = sorted(imp["delta_NSE"].items(), key=lambda kv: kv[1], reverse=True)
    names = [k.replace("Q_", "Q ").replace("precip_", "P ").replace("temp_", "T ") for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(names[::-1], vals[::-1], color="#1B4332")
    ax.set_xlabel("Drop in NSE after permutation")
    ax.set_title("Permutation importance (LSTM-Attention, 1-day)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close()


def plot_uncertainty(dates, obs, preds_seeds, path, title):
    arr = np.vstack(preds_seeds)
    mean = arr.mean(axis=0)
    lo, hi = arr.min(axis=0), arr.max(axis=0)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.fill_between(dates, lo, hi, color="#2D6A4F", alpha=0.25, label="min–max (3 seeds)")
    ax.plot(dates, obs, "k-", lw=1.0, label="Observed")
    ax.plot(dates, mean, color="#1B4332", lw=1.1, label="Ensemble mean")
    ax.set_title(title)
    ax.set_ylabel("Discharge (cfs)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close()


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)

    df = load_data().dropna().reset_index(drop=True)
    print(f"Loaded {len(df)} days")

    # feature sets for ablation
    local_cols = ["Q_target", "precip_target", "temp_target"]
    full_cols = FEATURE_COLS

    summary = {"horizons": {}, "ablation": {}, "importance_1d": None, "meta": {
        "basin": "Potomac River at Washington, DC (USGS-01646500)",
        "upstream": ["01636500", "01638480"],
        "period": "2000-01-01 to 2023-12-31",
        "split": "train≤2018 / val 2019–2020 / test 2021–2023",
    }}

    for h in HORIZONS:
        print(f"\n=== Horizon {h}d ===")
        pack = prepare_arrays(df, full_cols, h)
        preds = {"Persistence": pack["pers"]}
        metrics = {}

        xgb_pred, _ = run_xgb(pack)
        preds["XGBoost"] = xgb_pred

        lstm_seeds, attn_seeds = [], []
        for seed in SEEDS:
            print(f"  seed {seed}: LSTM / Attention")
            p_l, _ = run_dl(pack, "lstm", seed)
            p_a, model_a = run_dl(pack, "attn", seed)
            lstm_seeds.append(p_l)
            attn_seeds.append(p_a)
            if seed == SEEDS[0] and h == 1:
                imp = permutation_importance(pack, model_a)
                summary["importance_1d"] = imp
                plot_importance(imp, FIG / "fig_permutation_importance_1d.png")

        preds["LSTM"] = np.mean(lstm_seeds, axis=0)
        preds["LSTM-Attention"] = np.mean(attn_seeds, axis=0)
        preds["LSTM_seeds"] = lstm_seeds
        preds["Attn_seeds"] = attn_seeds

        for name in ["Persistence", "XGBoost", "LSTM", "LSTM-Attention"]:
            ext = compute_extended(pack["y_obs"], preds[name], pack["dates"])
            metrics[name] = ext
            print(f"  {name:16s} NSE={ext['NSE']:.3f} KGE={ext['KGE']:.3f} highNSE={ext['NSE_high']:.3f}")

        summary["horizons"][str(h)] = {
            "metrics": {},
            "peak_events_attn": metrics["LSTM-Attention"]["peaks"]["events"],
        }
        for k, v in metrics.items():
            entry = {kk: vv for kk, vv in v.items() if kk not in ("seasonal", "peaks")}
            entry["seasonal"] = v["seasonal"]
            entry["peaks_summary"] = {
                "mean_abs_peak_rel_err_pct": v["peaks"]["mean_abs_peak_rel_err_pct"],
                "mean_abs_timing_days": v["peaks"]["mean_abs_timing_days"],
            }
            summary["horizons"][str(h)]["metrics"][k] = entry

        plot_seasonal({m: metrics[m] for m in metrics}, h, FIG / f"fig_seasonal_nse_{h}d.png")
        plot_uncertainty(
            pack["dates"], pack["y_obs"], attn_seeds,
            FIG / f"fig_uncertainty_attn_{h}d.png",
            f"LSTM-Attention ensemble (3 seeds), {h}-day ahead",
        )

        # hydrograph zoom: last 365 test days
        n = min(400, len(pack["y_obs"]))
        fig, ax = plt.subplots(figsize=(11, 3.8))
        ax.plot(pack["dates"][-n:], pack["y_obs"][-n:], "k-", lw=1.0, label="Obs")
        ax.plot(pack["dates"][-n:], preds["LSTM-Attention"][-n:], color="#1B4332", lw=1.0, label="LSTM-Attn")
        ax.plot(pack["dates"][-n:], preds["XGBoost"][-n:], color="#E76F51", lw=0.9, alpha=0.85, label="XGBoost")
        ax.set_title(f"Test-period hydrograph excerpt ({h}-day ahead)")
        ax.set_ylabel("cfs")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / f"fig_hydro_excerpt_{h}d.png", dpi=300)
        plt.close()

        # ablation at this horizon
        pack_loc = prepare_arrays(df, local_cols, h)
        attn_full = preds["LSTM-Attention"]
        attn_loc_seeds = [run_dl(pack_loc, "attn", s, epochs=60)[0] for s in SEEDS]
        attn_loc = np.mean(attn_loc_seeds, axis=0)
        summary["ablation"][str(h)] = {
            "full_upstream": compute_all(pack["y_obs"], attn_full),
            "local_only": compute_all(pack_loc["y_obs"], attn_loc),
        }
        print(f"  Ablation NSE full={summary['ablation'][str(h)]['full_upstream']['NSE']:.3f} "
              f"local={summary['ablation'][str(h)]['local_only']['NSE']:.3f}")

    # Save JSON (convert numpy)
    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return float(o)
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    (RES / "enhanced_summary.json").write_text(json.dumps(_clean(summary), indent=2), encoding="utf-8")

    # CSV metrics table
    rows = []
    for h, block in summary["horizons"].items():
        for model, m in block["metrics"].items():
            rows.append({
                "horizon_d": int(h), "model": model,
                "NSE": m["NSE"], "KGE": m["KGE"], "RMSE_cfs": m["RMSE"], "MAE_cfs": m["MAE"],
                "PBIAS_pct": m["PBIAS"], "NSE_low": m.get("NSE_low"), "NSE_high": m.get("NSE_high"),
                "peak_abs_rel_err_pct": m.get("peaks_summary", {}).get("mean_abs_peak_rel_err_pct"),
                "peak_timing_abs_days": m.get("peaks_summary", {}).get("mean_abs_timing_days"),
            })
    pd.DataFrame(rows).to_csv(RES / "metrics_extended.csv", index=False)
    print(f"\nSaved results to {RES}")


if __name__ == "__main__":
    main()
