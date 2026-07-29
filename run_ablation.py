# -*- coding: utf-8 -*-
"""Ablation: effect of upstream station inputs on forecast skill."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from metrics import compute_all
from run_experiment import (
    DEVICE, FEATURE_COLS, HORIZONS, SEQ_LEN, TARGET_COL,
    LSTMAttention, load_data, make_sequences, temporal_split,
    inverse_transform_y, train_dl_model,
)

ROOT = Path(__file__).parent
RES_DIR = ROOT / "results"


def run_ablation_horizon(df, feature_cols, horizon):
    train, val, test = temporal_split(df)
    feat_scaler = StandardScaler()
    y_scaler = StandardScaler()

    train_feat = feat_scaler.fit_transform(train[feature_cols])
    val_feat = feat_scaler.transform(val[feature_cols])
    y_scaler.fit(train[[TARGET_COL]])

    def build_split(feat_arr, target_series):
        y_scaled = y_scaler.transform(target_series.values.reshape(-1, 1)).ravel()
        return make_sequences(feat_arr, y_scaled, SEQ_LEN, horizon)

    X_tr, y_tr = build_split(train_feat, train[TARGET_COL])
    val_feat_ext = np.vstack([train_feat[-SEQ_LEN:], val_feat])
    val_y_ext = pd.concat([train[TARGET_COL].iloc[-SEQ_LEN:], val[TARGET_COL]]).reset_index(drop=True)
    X_va, y_va = build_split(val_feat_ext, val_y_ext)

    ext_df = pd.concat([val, test]).reset_index(drop=True)
    test_feat_ext = feat_scaler.transform(ext_df[feature_cols].values)
    y_ext_scaled = y_scaler.transform(ext_df[TARGET_COL].values.reshape(-1, 1)).ravel()
    X_te, y_te_scaled = make_sequences(test_feat_ext, y_ext_scaled, SEQ_LEN, horizon)

    all_target_dates = ext_df["date"].iloc[SEQ_LEN + horizon - 1 : SEQ_LEN + horizon - 1 + len(X_te)].values
    test_mask = all_target_dates >= test["date"].min()
    X_te = X_te[test_mask]
    y_te_scaled = y_te_scaled[test_mask]
    y_obs = inverse_transform_y(y_scaler, y_te_scaled)

    model = train_dl_model(LSTMAttention(len(feature_cols)), X_tr, y_tr, X_va, y_va, epochs=60)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_te).to(DEVICE)).cpu().numpy()
    pred = inverse_transform_y(y_scaler, pred)
    return compute_all(y_obs, pred)


def main():
    df = load_data()
    # Add rolling precipitation features
    for col in ["precip_target", "precip_01636500", "precip_01638480"]:
        df[f"{col}_7d"] = df[col].rolling(7, min_periods=1).sum()
        df[f"{col}_30d"] = df[col].rolling(30, min_periods=1).sum()
    df = df.dropna().reset_index(drop=True)

    full_features = FEATURE_COLS + [c for c in df.columns if c.endswith("_7d") or c.endswith("_30d")]
    local_features = ["Q_target", "precip_target", "temp_target",
                      "precip_target_7d", "precip_target_30d"]

    results = {"full_upstream": {}, "local_only": {}}
    for h in HORIZONS:
        print(f"Ablation horizon {h}d...")
        results["full_upstream"][str(h)] = run_ablation_horizon(df, full_features, h)
        results["local_only"][str(h)] = run_ablation_horizon(df, local_features, h)
        print(f"  Full:    NSE={results['full_upstream'][str(h)]['NSE']:.3f}")
        print(f"  Local:   NSE={results['local_only'][str(h)]['NSE']:.3f}")

    RES_DIR.mkdir(exist_ok=True)
    (RES_DIR / "ablation_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {RES_DIR / 'ablation_results.json'}")


if __name__ == "__main__":
    main()
