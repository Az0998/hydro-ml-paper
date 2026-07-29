# -*- coding: utf-8 -*-
"""Hydrological evaluation metrics."""

import numpy as np


def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs, sim = obs[mask], sim[mask]
    if len(obs) == 0:
        return float("nan")
    denom = np.sum((obs - np.mean(obs)) ** 2)
    if denom == 0:
        return float("nan")
    return 1 - np.sum((obs - sim) ** 2) / denom


def kge(obs: np.ndarray, sim: np.ndarray) -> float:
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs, sim = obs[mask], sim[mask]
    if len(obs) < 2:
        return float("nan")
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / (np.std(obs) + 1e-12)
    beta = np.mean(sim) / (np.mean(obs) + 1e-12)
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def rmse(obs: np.ndarray, sim: np.ndarray) -> float:
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    return float(np.sqrt(np.mean((obs[mask] - sim[mask]) ** 2)))


def mae(obs: np.ndarray, sim: np.ndarray) -> float:
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    return float(np.mean(np.abs(obs[mask] - sim[mask])))


def pbias(obs: np.ndarray, sim: np.ndarray) -> float:
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs, sim = obs[mask], sim[mask]
    return float(100 * np.sum(sim - obs) / (np.sum(obs) + 1e-12))


def compute_all(obs: np.ndarray, sim: np.ndarray) -> dict:
    return {
        "NSE": nse(obs, sim),
        "KGE": kge(obs, sim),
        "RMSE": rmse(obs, sim),
        "MAE": mae(obs, sim),
        "PBIAS": pbias(obs, sim),
    }
