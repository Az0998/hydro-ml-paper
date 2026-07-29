![status](https://img.shields.io/badge/status-paper%20submit-blue) ![venue](https://img.shields.io/badge/target-HSJ-lightgrey)

# 融合上游水文信息的河流流量多预见期预报（投稿版）

**当前主攻：** *Hydrological Sciences Journal*  
**已试投：** JHRS（desk reject；已按拒稿理由升级）  
详见 `JOURNAL_STRATEGY.md`、`LESSONS_FROM_JHRS_REJECTION.md`、`paper/HSJ_SUBMISSION_CHECKLIST.md`。

## 一键生成投稿包

```bash
cd hydro-ml-paper
python download_extended.py      # Potomac+James+QPF（已下载可跳过）
python run_hsj_upgrade.py        # 路由/oracle/洪水CSI
python run_tier2_upgrade.py      # James迁移 + 真实QPF
python run_qpf_extra.py          # QPF降水检验（可选）
python make_hsj_graphical_abstract.py
python build_hsj_manuscript.py
```

## HSJ 投稿文件（`paper/`）

| 文件 | 用途 |
|------|------|
| `HSJ_Potomac_forecast_information_value.docx` | 主稿 |
| `HSJ_highlights.docx` | Highlights |
| `HSJ_cover_letter.txt` | 附信 |
| `HSJ_Declaration_of_Interest.docx` | 利益冲突 |
| `HSJ_graphical_abstract.png` | 图形摘要 |

## 核心结果（测试集 2021–2023，增强实验）

| 尺度 | 最优模型 | NSE | 备注 |
|------|----------|-----|------|
| 1 天 | LSTM-Attention | **0.931** | 上游消融：0.931→0.907 |
| 3 天 | LSTM | **0.543** | Attention 非最优 |
| 7 天 | LSTM-Attention | **0.196** | 周尺度整体偏弱 |

## 研究定位（审稿友好）

不宣称“发明新网络”，而强调：
1. 上游站网对华盛顿控制站短预见期的边际价值  
2. 深学 vs 树模型随预见期变化的条件比较优势  
3. 洪峰/季节分层诊断对中大西洋业务预报的区域认识  

## 旧版 LaTeX / 水科学进展 Word

仍保留 `paper/manuscript.tex` 与 `build_word_manuscript.py`（中文核心格式）作为备份，**主投请用 JHRS 英文稿**。