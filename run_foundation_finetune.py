# -*- coding: utf-8 -*-
"""
Hydrology foundation-model fine-tune contrast (skeleton).

Question: does a pretrained regional/global hydrologic representation beat
small-basin LSTM-Attention trained from scratch?

Practical near-term protocol (before full ClimaX/Prithvi wiring):
1) Multi-basin pretrain: shared LSTM encoder on Potomac+James+Willamette+Animas+Verde
2) Fine-tune decoder on each target basin
3) Compare vs scratch Attention on identical splits/metrics

Full foundation weights (ClimaX/Prithvi) can replace step (1) when GPU + checkpoints available.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent
RES = ROOT / "results" / "foundation"
RES.mkdir(parents=True, exist_ok=True)

PLAN = {
    "status": "skeleton",
    "near_term_proxy": "multi-basin pretrain + per-basin fine-tune (implementable now)",
    "aspirational": ["ClimaX", "Prithvi-WxC", "other open hydro FM checkpoints"],
    "baselines": ["LSTM-Attention scratch (current paper)"],
    "metrics": ["NSE/KGE by horizon", "ablation ΔNSE", "sample-efficiency curves"],
    "deliverables": [
        "results/foundation/finetune_vs_scratch.csv",
        "figures/foundation/fig_sample_efficiency.png",
    ],
}


def main():
    (RES / "foundation_plan.json").write_text(json.dumps(PLAN, indent=2), encoding="utf-8")
    print("Wrote foundation plan ->", RES / "foundation_plan.json")


if __name__ == "__main__":
    main()
