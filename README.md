![status](https://img.shields.io/badge/status-open%20code-brightgreen)
![pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue)

# Hydro-ML Paper

**Project site:** <https://az0998.github.io/hydro-ml-paper/>  
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
