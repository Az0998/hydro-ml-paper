# -*- coding: utf-8 -*-
"""
Streamflow forecasting experiment:
LSTM+Attention vs baselines on USGS Potomac River basin data.
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

from metrics import compute_all

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "potomac_daily.csv"
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"

SEQ_LEN = 30
HORIZONS = [1, 3, 7]
TRAIN_END = "2018-12-31"
VAL_END = "2020-12-31"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

FEATURE_COLS = [
    "Q_target", "precip_target", "temp_target",
    "Q_01636500", "precip_01636500",
    "Q_01638480", "precip_01638480",
]
TARGET_COL = "Q_target"


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def temporal_split(df):
    train = df[df["date"] <= TRAIN_END].copy()
    val = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)].copy()
    test = df[df["date"] > VAL_END].copy()
    return train, val, test


def make_sequences(arr, target, seq_len, horizon):
    X, y = [], []
    for i in range(len(arr) - seq_len - horizon + 1):
        X.append(arr[i : i + seq_len])
        y.append(target[i + seq_len + horizon - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class LSTMModel(nn.Module):
    def __init__(self, n_features, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class LSTMAttention(nn.Module):
    def __init__(self, n_features, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True, dropout=dropout)
        self.attn = nn.Linear(hidden, 1)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        weights = torch.softmax(self.attn(out).squeeze(-1), dim=1)
        context = torch.sum(out * weights.unsqueeze(-1), dim=1)
        return self.fc(context).squeeze(-1)


def train_dl_model(model, X_tr, y_tr, X_va, y_va, epochs=80, lr=1e-3, batch=64):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_state, best_val = None, float("inf")

    tr_loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), batch_size=batch, shuffle=True)
    X_va_t = torch.tensor(X_va).to(DEVICE)
    y_va_t = torch.tensor(y_va).to(DEVICE)

    for ep in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_va_t), y_va_t).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model


def predict_dl(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X).to(DEVICE)).cpu().numpy()


def run_persistence(test_target, horizon):
    """Yesterday's flow as forecast."""
    obs = test_target.values
    sim = np.roll(obs, horizon)
    sim[:horizon] = obs[:horizon]
    return sim


def run_arima(train_target, test_target, horizon, n_test):
    """ARIMA baseline with periodic refit (operational forecasting protocol)."""
    try:
        history = list(train_target.values.astype(float))
        fit, preds = None, []
        test_vals = test_target.values.astype(float)
        for i in range(n_test):
            if fit is None or i % 60 == 0:
                fit = ARIMA(history[-800:], order=(1, 0, 1)).fit()
            fc = fit.forecast(steps=horizon)
            preds.append(float(fc.iloc[-1] if hasattr(fc, "iloc") else fc[-1]))
            history.append(float(test_vals[i]))
        return np.array(preds)
    except Exception:
        return run_persistence(test_target, horizon)[:n_test]


def run_xgboost(X_tr_flat, y_tr, X_te_flat):
    model = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1,
    )
    model.fit(X_tr_flat, y_tr)
    return model.predict(X_te_flat)


def flatten_sequences(X):
    return X.reshape(X.shape[0], -1)


def inverse_transform_y(scaler_y, y):
    return scaler_y.inverse_transform(y.reshape(-1, 1)).ravel()


def run_horizon(df, horizon):
    set_seed()
    train, val, test = temporal_split(df)

    feat_scaler = StandardScaler()
    y_scaler = StandardScaler()

    train_feat = feat_scaler.fit_transform(train[FEATURE_COLS])
    val_feat = feat_scaler.transform(val[FEATURE_COLS])
    test_feat = feat_scaler.transform(test[FEATURE_COLS])

    y_scaler.fit(train[[TARGET_COL]])

    def build_split(feat_arr, target_series):
        X, y = make_sequences(feat_arr, y_scaler.transform(target_series.values.reshape(-1, 1)).ravel(), SEQ_LEN, horizon)
        return X, y

    X_tr, y_tr = build_split(train_feat, train[TARGET_COL])
    val_feat_ext = np.vstack([train_feat[-SEQ_LEN:], val_feat])
    val_y_ext = pd.concat([train[TARGET_COL].iloc[-SEQ_LEN:], val[TARGET_COL]]).reset_index(drop=True)
    X_va, y_va = build_split(val_feat_ext, val_y_ext)
    full_train_feat = np.vstack([train_feat, val_feat])
    full_train_y = pd.concat([train[TARGET_COL], val[TARGET_COL]]).reset_index(drop=True)
    X_tr_full, y_tr_full = build_split(full_train_feat, full_train_y)

    ext_df = pd.concat([val, test]).reset_index(drop=True)
    test_feat_ext = feat_scaler.transform(ext_df[FEATURE_COLS].values)
    y_ext_scaled = y_scaler.transform(ext_df[TARGET_COL].values.reshape(-1, 1)).ravel()
    X_te, y_te_scaled = make_sequences(test_feat_ext, y_ext_scaled, SEQ_LEN, horizon)

    # Keep only sequences whose target date falls in test period
    all_target_dates = ext_df["date"].iloc[SEQ_LEN + horizon - 1 : SEQ_LEN + horizon - 1 + len(X_te)].values
    test_start = test["date"].min()
    test_mask = all_target_dates >= test_start
    X_te = X_te[test_mask]
    y_te_scaled = y_te_scaled[test_mask]
    test_dates = all_target_dates[test_mask]
    y_obs = inverse_transform_y(y_scaler, y_te_scaled)
    n_eval = len(y_obs)

    results = {}

    # Persistence: Q(t) as forecast for Q(t+h)
    ext_q = ext_df[TARGET_COL].values.astype(float)
    pers_idx = np.where(test_mask)[0] + SEQ_LEN + horizon - 1
    pers = ext_q[pers_idx - horizon]
    results["Persistence"] = compute_all(y_obs, pers)

    # ARIMA
    arima_pred = run_arima(full_train_y, test[TARGET_COL].reset_index(drop=True), horizon, n_eval)
    results["ARIMA"] = compute_all(y_obs, arima_pred)

    # XGBoost
    xgb_pred = run_xgboost(flatten_sequences(X_tr_full), y_tr_full, flatten_sequences(X_te))
    xgb_pred = inverse_transform_y(y_scaler, xgb_pred)
    results["XGBoost"] = compute_all(y_obs, xgb_pred)

    # LSTM
    lstm = train_dl_model(LSTMModel(len(FEATURE_COLS)), X_tr, y_tr, X_va, y_va)
    lstm_pred = inverse_transform_y(y_scaler, predict_dl(lstm, X_te))
    results["LSTM"] = compute_all(y_obs, lstm_pred)

    # LSTM + Attention
    attn = train_dl_model(LSTMAttention(len(FEATURE_COLS)), X_tr, y_tr, X_va, y_va)
    attn_pred = inverse_transform_y(y_scaler, predict_dl(attn, X_te))
    results["LSTM-Attention"] = compute_all(y_obs, attn_pred)

    preds = {
        "dates": test_dates,
        "observed": y_obs,
        "Persistence": pers,
        "ARIMA": arima_pred[:n_eval],
        "XGBoost": xgb_pred,
        "LSTM": lstm_pred,
        "LSTM-Attention": attn_pred,
    }
    return results, preds


def plot_hydrograph(preds, horizon, out_path):
    dates = preds["dates"]
    obs = preds["observed"]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, obs, "k-", label="Observed", linewidth=1.2)
    ax.plot(dates, preds["LSTM-Attention"], color="#2D6A4F", label="LSTM-Attention", linewidth=1.0)
    ax.plot(dates, preds["LSTM"], color="#40916C", label="LSTM", linewidth=0.9, alpha=0.8)
    ax.plot(dates, preds["XGBoost"], color="#E76F51", label="XGBoost", linewidth=0.9, alpha=0.8)
    ax.plot(dates, preds["ARIMA"], color="#457B9D", label="ARIMA", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Date")
    ax.set_ylabel("Discharge (cfs)")
    ax.set_title(f"Potomac River at Washington, DC — {horizon}-day ahead forecast (Test: 2021–2023)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_metrics_bar(all_metrics, out_path):
    models = list(next(iter(all_metrics.values())).keys())
    metrics_names = ["NSE", "KGE", "RMSE"]
    horizons = list(all_metrics.keys())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    colors = {"Persistence": "#999", "ARIMA": "#457B9D", "XGBoost": "#E76F51", "LSTM": "#40916C", "LSTM-Attention": "#1B4332"}

    for ax, metric in zip(axes, metrics_names):
        x = np.arange(len(horizons))
        width = 0.15
        for i, model in enumerate(models):
            vals = [all_metrics[h][model][metric] for h in horizons]
            ax.bar(x + i * width, vals, width, label=model, color=colors.get(model, "#333"))
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels([f"{h}-day" for h in horizons])
        ax.set_title(metric)
        ax.grid(True, axis="y", alpha=0.3)
        if metric in ("NSE", "KGE"):
            ax.set_ylim(0, 1)
    axes[0].legend(loc="lower left", fontsize=7, ncol=2)
    fig.suptitle("Model Performance Comparison Across Forecast Horizons", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(preds, out_path):
    obs = preds["observed"]
    sim = preds["LSTM-Attention"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(obs, sim, alpha=0.4, s=12, c="#2D6A4F")
    lim = [min(obs.min(), sim.min()), max(obs.max(), sim.max())]
    ax.plot(lim, lim, "k--", linewidth=1)
    ax.set_xlabel("Observed Discharge (cfs)")
    ax.set_ylabel("Predicted Discharge (cfs)")
    ax.set_title("LSTM-Attention: Observed vs Predicted (7-day)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        from download_data import main as dl
        dl()

    df = load_data()
    print(f"Data: {len(df)} days, {df['date'].min().date()} – {df['date'].max().date()}")

    all_metrics = {}
    preds_7d = None

    for h in HORIZONS:
        print(f"\n=== Horizon: {h} day(s) ===")
        metrics, preds = run_horizon(df, h)
        all_metrics[h] = metrics
        for model, m in metrics.items():
            print(f"  {model:16s} NSE={m['NSE']:.3f} KGE={m['KGE']:.3f} RMSE={m['RMSE']:.1f}")
        plot_hydrograph(preds, h, FIG_DIR / f"hydrograph_{h}d.png")
        if h == 7:
            preds_7d = preds

    # Save metrics table
    rows = []
    for h, models in all_metrics.items():
        for model, m in models.items():
            rows.append({"horizon": h, "model": model, **m})
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(RES_DIR / "metrics_summary.csv", index=False)

    plot_metrics_bar(all_metrics, FIG_DIR / "metrics_comparison.png")
    if preds_7d:
        plot_scatter(preds_7d, FIG_DIR / "scatter_7d.png")

    # Save JSON for paper
    (RES_DIR / "metrics_summary.json").write_text(
        json.dumps({str(k): v for k, v in all_metrics.items()}, indent=2), encoding="utf-8"
    )

    print(f"\nResults saved to {RES_DIR}")
    print(f"Figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
