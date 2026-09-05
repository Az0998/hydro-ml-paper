![status](https://img.shields.io/badge/status-open%20code-brightgreen)

# Hydro-ML：多预见期流量预报与信息价值

> **一句话：** 公开 USGS + Open-Meteo 数据上做多预见期流量预报，评估上游站与降水预见的信息价值（不是新网络刷榜）。个人兴趣自学；曾作为 HSJ 投稿流水线，**尚无录用**。更新叙事见姊妹仓 [forecast-information-value](https://github.com/Az0998/forecast-information-value)。

**仓库：** https://github.com/Az0998/hydro-ml-paper  
**作者：** 张森捷（Senjie Zhang），兰州大学

| 项目 | 说明 |
|------|------|
| 流域 | Potomac / James 协议 + 气候带迁移（Willamette / Animas / Verde） |
| 方法 | 路由基线、消融、QPF 阶梯、洪水 CSI、SHAP |
| 复现 | `download_data.py` → `run_experiment.py` |

---

# Hydro-ML Paper

**Code:** <https://github.com/Az0998/hydro-ml-paper>

Multi-horizon streamflow forecasting with public USGS + Open-Meteo data. Focus: **information value** of upstream gauges and precipitation foresight (not a new architecture claim).

## Highlights

- Potomac / James mid-Atlantic protocol + routing baseline, ablation, QPF ladder, flood CSI  
- Climate-zone transfer: Willamette (humid NW), Animas (snowmelt), Verde (semi-arid)  
- SHAP channel importance aligned with permutation tests  

## Quick start

```bash
git clone https://github.com/Az0998/hydro-ml-paper.git
cd hydro-ml-paper
pip install -r requirements.txt
python download_data.py
python run_experiment.py
```

Climate transfer / SHAP:

```bash
python download_climate_basins.py
python run_climate_transfer.py
python run_shap_explain.py
```

## Key paths

| Path | Content |
|------|---------|
| `docs/` | GitHub Pages site |
| `results/climate_transfer/` | Multi-basin metrics |
| `results/shap/` | SHAP summary |
| `ROADMAP_NEXT.md` | Next upgrades (GEFS, foundation models) |
| `paper/` | Manuscript package |

## Data

Scripts download from USGS NWIS and Open-Meteo. Processed CSVs under `data/` may be committed for convenience; re-run download scripts to refresh.
