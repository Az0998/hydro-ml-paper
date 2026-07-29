# -*- coding: utf-8 -*-
"""Extended hydrological skill metrics for manuscript diagnostics."""

from __future__ import annotations

import numpy as np

from metrics import compute_all, nse, kge, rmse, mae, pbias


def stratified_nse(obs, sim, q_low=None, q_high=None):
    """NSE on low-flow (<q_low) and high-flow (>q_high) subsets (percentile thresholds)."""
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    if q_low is None:
        q_low = np.nanpercentile(obs, 30)
    if q_high is None:
        q_high = np.nanpercentile(obs, 70)
    low = obs <= q_low
    high = obs >= q_high
    mid = (~low) & (~high)
    out = {
        "NSE_low": nse(obs[low], sim[low]) if low.sum() > 10 else float("nan"),
        "NSE_mid": nse(obs[mid], sim[mid]) if mid.sum() > 10 else float("nan"),
        "NSE_high": nse(obs[high], sim[high]) if high.sum() > 10 else float("nan"),
        "q_low": float(q_low),
        "q_high": float(q_high),
    }
    return out


def seasonal_nse(obs, sim, dates):
    """Seasonal NSE: DJF/MAM/JJA/SON."""
    import pandas as pd
    d = pd.to_datetime(dates)
    months = np.array(d.month)
    seasons = {
        "DJF": (months == 12) | (months <= 2),
        "MAM": (months >= 3) & (months <= 5),
        "JJA": (months >= 6) & (months <= 8),
        "SON": (months >= 9) & (months <= 11),
    }
    return {k: nse(obs[m], sim[m]) if m.sum() > 20 else float("nan") for k, m in seasons.items()}


def peak_event_errors(obs, sim, dates, n_events=8, window=5):
    """
    Identify top n_events peaks in observations; report relative peak error and timing error (days).
    Timing: argmax in ±window around observed peak.
    """
    import pandas as pd
    obs = np.asarray(obs, float)
    sim = np.asarray(sim, float)
    d = pd.to_datetime(dates)
    # non-overlapping peak picking
    order = np.argsort(obs)[::-1]
    used = np.zeros(len(obs), dtype=bool)
    events = []
    for idx in order:
        if used[idx]:
            continue
        lo, hi = max(0, idx - window), min(len(obs), idx + window + 1)
        if used[lo:hi].any():
            continue
        used[lo:hi] = True
        local = slice(lo, hi)
        sim_peak_i = lo + int(np.nanargmax(sim[local]))
        peak_obs = obs[idx]
        peak_sim = sim[sim_peak_i]
        rel_err = (peak_sim - peak_obs) / (peak_obs + 1e-6) * 100
        timing = int(sim_peak_i - idx)
        events.append({
            "date": str(pd.Timestamp(d[idx]).date()),
            "Q_obs_peak": float(peak_obs),
            "Q_sim_peak": float(peak_sim),
            "peak_rel_err_pct": float(rel_err),
            "timing_error_days": timing,
        })
        if len(events) >= n_events:
            break
    if not events:
        return {"mean_abs_peak_rel_err_pct": float("nan"), "mean_abs_timing_days": float("nan"), "events": []}
    return {
        "mean_abs_peak_rel_err_pct": float(np.mean([abs(e["peak_rel_err_pct"]) for e in events])),
        "mean_abs_timing_days": float(np.mean([abs(e["timing_error_days"]) for e in events])),
        "events": events,
    }


def compute_extended(obs, sim, dates=None) -> dict:
    base = compute_all(obs, sim)
    base.update(stratified_nse(obs, sim))
    if dates is not None:
        base["seasonal"] = seasonal_nse(obs, sim, dates)
        base["peaks"] = peak_event_errors(obs, sim, dates)
    return base
