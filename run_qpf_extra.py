# -*- coding: utf-8 -*-
"""Extra QPF verification + mid-2024 pure-QPF streamflow test."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from metrics import compute_all, rmse
from run_experiment import (
    SEQ_LEN, TARGET_COL, LSTMAttention, make_sequences,
    inverse_transform_y, train_dl_model, predict_dl,
)
from run_tier2_upgrade import feature_cols_for, add_future_precip, set_seed, RES

ROOT = Path(__file__).parent


def midyear_one(df, horizon, cols, foresight=False):
    dsub = df[df["date"].dt.year == 2024].dropna(subset=cols).reset_index(drop=True)
    train = dsub[dsub["date"] <= "2024-06-30"]
    test = dsub[dsub["date"] > "2024-06-30"]
    val = train.iloc[-60:].copy()
    train2 = train.iloc[:-60].copy()
    if len(train2) < SEQ_LEN + horizon + 20 or len(test) < 40:
        return None

    feat_scaler = StandardScaler()
    y_scaler = StandardScaler()
    tr_f = feat_scaler.fit_transform(train2[cols])
    y_scaler.fit(train2[[TARGET_COL]])
    va_f = feat_scaler.transform(val[cols])

    def build(feat, series):
        ys = y_scaler.transform(series.values.reshape(-1, 1)).ravel()
        return make_sequences(feat, ys, SEQ_LEN, horizon)

    Xtr, ytr = build(tr_f, train2[TARGET_COL])
    va_ext = np.vstack([tr_f[-SEQ_LEN:], va_f])
    vy = pd.concat([train2[TARGET_COL].iloc[-SEQ_LEN:], val[TARGET_COL]]).reset_index(drop=True)
    Xva, yva = build(va_ext, vy)

    ctx = pd.concat([train, test]).reset_index(drop=True)
    ctx_f = feat_scaler.transform(ctx[cols])
    ctx_y = y_scaler.transform(ctx[TARGET_COL].values.reshape(-1, 1)).ravel()
    Xall, yall = make_sequences(ctx_f, ctx_y, SEQ_LEN, horizon)
    dates = ctx["date"].iloc[SEQ_LEN + horizon - 1: SEQ_LEN + horizon - 1 + len(Xall)].values
    mask = dates > np.datetime64("2024-06-30")
    Xte = Xall[mask]
    yobs = inverse_transform_y(y_scaler, yall[mask])
    set_seed(42)
    model = train_dl_model(LSTMAttention(len(cols)), Xtr, ytr, Xva, yva, epochs=40)
    pred = inverse_transform_y(y_scaler, predict_dl(model, Xte))
    met = compute_all(yobs, pred)
    return {k: float(v) for k, v in met.items()} | {"n_test": int(mask.sum()), "foresight": foresight}


def main():
    df = pd.read_csv(ROOT / "data" / "potomac_daily_with_qpf.csv", parse_dates=["date"]).sort_values("date")
    d24 = df[df["date"].dt.year == 2024].copy()
    ver = {}
    for lead, col in [(1, "qpf_lead1"), (3, "qpf_lead3"), (7, "qpf_lead7")]:
        a = d24[["precip_target", col]].dropna()
        r = float(np.corrcoef(a["precip_target"], a[col])[0, 1])
        ver[str(lead)] = {
            "corr": r,
            "rmse_mm": float(rmse(a["precip_target"].values, a[col].values)),
            "bias_mm": float(np.mean(a[col] - a["precip_target"])),
            "n": int(len(a)),
        }
        print("QPF verify", lead, ver[str(lead)])

    base_cols = feature_cols_for(df)
    mid = {}
    for h in [1, 3, 7]:
        d_q, extra = add_future_precip(df, h, "qpf")
        # restrict to 2024 rows where original qpf was present (not only fill)
        d_q = d_q[d_q["date"].dt.year == 2024].reset_index(drop=True)
        mq = midyear_one(d_q, h, base_cols + extra, foresight=True)
        mo = midyear_one(df, h, base_cols, foresight=False)
        if mq and mo:
            mid[str(h)] = {
                "obs_NSE": mo["NSE"],
                "qpf_NSE": mq["NSE"],
                "delta": mq["NSE"] - mo["NSE"],
                "n_test": mq["n_test"],
            }
            print(f"midyear {h}d obs={mo['NSE']:.3f} qpf={mq['NSE']:.3f} d={mq['NSE']-mo['NSE']:+.3f}")

    path = RES / "tier2_summary.json"
    s = json.loads(path.read_text(encoding="utf-8"))
    s["qpf_extra"] = {"precip_verification_2024": ver, "midyear_2024": mid}
    path.write_text(json.dumps(s, indent=2), encoding="utf-8")
    print("updated", path)


if __name__ == "__main__":
    main()
