# 投稿包清单（Journal of Hydrology: Regional Studies）

## 已选定期刊

| 优先级 | 期刊 | 大致 JIF | 角色 |
|--------|------|----------|------|
| 1 | **Journal of Hydrology: Regional Studies** (Elsevier) | ~4.7 | 主投 |
| 2 | Hydrological Sciences Journal | ~2.5–3.5 | 退稿转投 |
| 3 | Water (MDPI) | ~3.0 | 保底 |

详见 `JOURNAL_STRATEGY.md`。

## 稿件与材料

| 文件 | 说明 |
|------|------|
| `paper/JHRS_Potomac_upstream_streamflow_forecasting.docx` | 主文（结构化摘要、Highlights、双倍行距、作者–年份文献） |
| `paper/JHRS_cover_letter.txt` | 投稿信草稿 |
| `figures/` 与 `figures/enhanced/` | 插图 |
| `results/` 与 `results/enhanced/` | 指标与消融 |
| `download_data.py` / `run_experiment.py` / `run_enhanced_analysis.py` | 可复现流水线 |

## 提交前人工必做

1. 填写真实作者单位、通讯邮箱、基金号。  
2. 按 Editorial Manager 上传：Manuscript / Highlights / Cover letter / Figures（也可嵌在 Word，但建议另传高清图）。  
3. 确认摘要含且仅含三段标题：Study region / Study focus / New hydrological insights for the region。  
4. 用 Turnitin/iThenticate 自查；对本仓库生成文本再做一轮人工改写（尤其 Introduction 末段与 Discussion）。  
5. 声明数据来自 USGS 与 Open-Meteo；上传代码压缩包或公开仓库链接。  
6. 增强实验跑完后执行：`python build_jhrs_manuscript.py` 刷新文中数值与图。

## 科学表述原则（降低“AI 稿”观感）

- 明确写清 **LSTM-Attention 并非在所有预见期最优**。  
- 强调区域结论（华盛顿控制站、中大西洋）而非“通用新算法”。  
- Discussion 写清周尺度失败原因（缺未来降水）。  
- 数字全部来自本项目实验结果，禁止空泛形容词堆砌。

## 复现命令

```bash
cd hydro-ml-paper
pip install -r requirements.txt
python download_data.py          # 若 data/ 已有可跳过
python run_enhanced_analysis.py  # 增强诊断（较慢）
python build_jhrs_manuscript.py
```
