# -*- coding: utf-8 -*-
"""
SHAP interpretability for Potomac 1-day models (XGBoost + optional Attention surrogate).
Cross-checks existing permutation importance.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from metrics import compute_all
from run_experiment import (
    FEATURE_COLS, SEQ_LEN, TARGET_COL, TRAIN_END, VAL_END,
    load_data, make_sequences, flatten_sequences, inverse_transform_y, temporal_split,
)
from run_enhanced_analysis import prepare_arrays

ROOT = Path(__file__).parent
RES = ROOT / "results" / "shap"
FIG = ROOT / "figures" / "shap"
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def main():
    df = load_data()
    # clip to 2023 for consistency with climate protocol
    if "date" in df.columns:
        df = df[df["date"] <= "2023-12-31"].reset_index(drop=True)

    pack = prepare_arrays(df, FEATURE_COLS, horizon=1)
    model = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1,
    )
    Xtr = flatten_sequences(pack["X_tr_full"])
    Xte = flatten_sequences(pack["X_te"])
    model.fit(Xtr, pack["y_tr_full"])
    pred = inverse_transform_y(pack["y_scaler"], model.predict(Xte))
    print("XGB NSE", compute_all(pack["y_obs"], pred)["NSE"])

    # SHAP on a sample of test flattened windows (speed)
    rng = np.random.default_rng(0)
    n = min(400, len(Xte))
    idx = rng.choice(len(Xte), size=n, replace=False)
    Xs = Xte[idx]

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xs)

    # Aggregate |SHAP| back to original feature channels (seq_len blocks)
    n_feat = len(FEATURE_COLS)
    # flattened layout: [t0_f0, t0_f1, ... t0_fn, t1_f0, ...]
    abs_mean = np.abs(sv).mean(axis=0)
    channel = np.zeros(n_feat)
    for t in range(SEQ_LEN):
        channel += abs_mean[t * n_feat:(t + 1) * n_feat]
    channel /= SEQ_LEN
    ranking = sorted(zip(FEATURE_COLS, channel.tolist()), key=lambda x: -x[1])
    print("SHAP channel ranking:")
    for k, v in ranking:
        print(f"  {k}: {v:.4f}")

    # bar plot
    names = [k for k, _ in ranking]
    vals = [v for _, v in ranking]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.barh(names[::-1], vals[::-1], color="#4C78A8", edgecolor="k", lw=0.4)
    ax.set_xlabel("Mean |SHAP| (aggregated over lag window)")
    ax.set_title("Potomac 1-day XGBoost — SHAP channel importance")
    fig.tight_layout()
    fig.savefig(FIG / "fig_shap_xgb_1d.png", dpi=200)
    plt.close(fig)

    # beeswarm-like summary via shap plotting API
    # rebuild feature names for flattened dims is huge; plot channel-only is enough for paper

    out = {
        "model": "XGBoost",
        "horizon": 1,
        "n_shap_samples": int(n),
        "channel_mean_abs_shap": {k: float(v) for k, v in ranking},
        "test_NSE": float(compute_all(pack["y_obs"], pred)["NSE"]),
        "note": "Channel scores average |SHAP| across the 30-day flattened lags.",
    }
    (RES / "shap_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Wrote", RES / "shap_summary.json")


if __name__ == "__main__":
    main()
