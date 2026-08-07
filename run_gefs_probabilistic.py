# -*- coding: utf-8 -*-
"""
GEFS / ensemble QPF → probabilistic streamflow forecast (skeleton).

Design (for next implementation pass):
1) Ingest ensemble daily precip members for leads 1/3/7 (GEFS via NOAA or Open-Meteo ensemble).
2) Drive LSTM-Attention / XGBoost once per member OR train a quantile head.
3) Evaluate with CRPS, PIT reliability, and P90 exceedance probability CSI.

This skeleton documents the API and writes a placeholder summary so the roadmap is actionable.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent
RES = ROOT / "results" / "gefs"
RES.mkdir(parents=True, exist_ok=True)

PLAN = {
    "status": "skeleton",
    "goal": "Replace deterministic GFS QPF with ensemble precip → probabilistic streamflow",
    "data_options": [
        "Open-Meteo Ensemble API (member precip)",
        "NOAA GEFS reforecast / operational archive",
    ],
    "models": [
        "Member-wise forcing of trained Attention (non-intrusive)",
        "Quantile regression / CRPS-trained head (intrusive)",
    ],
    "metrics": ["CRPS", "reliability diagram", "P(Q>=P90) CSI at chosen probability thresholds"],
    "deliverables": [
        "results/gefs/prob_metrics.csv",
        "figures/gefs/fig_crps_by_horizon.png",
        "figures/gefs/fig_reliability_p90.png",
    ],
    "next_code_hooks": {
        "download": "download_gefs_ensemble.py",
        "run": "run_gefs_probabilistic.py (expand this file)",
    },
}


def main():
    (RES / "gefs_plan.json").write_text(json.dumps(PLAN, indent=2), encoding="utf-8")
    print("Wrote GEFS plan ->", RES / "gefs_plan.json")
    print("Implement download + member loop next when ready.")


if __name__ == "__main__":
    main()
