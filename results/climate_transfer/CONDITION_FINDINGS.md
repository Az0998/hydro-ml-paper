# 多气候带初步结论（可写进 HSJ 大修/下一稿）

数据：`results/climate_transfer/climate_basin_compare.csv`  
图：`figures/climate_transfer/`

## 1 天上游消融 ΔNSE（full − local）

| 流域 | 气候 | Routing NSE | Attn NSE | 消融 ΔNSE | 解读 |
|------|------|-------------|----------|-----------|------|
| James | 湿润中大西洋 | 0.92 | 0.93 | **+0.13** | 上游信息价值最大 |
| Verde | 半干旱西南 | 0.61 | 0.38 | +0.06 |  Upstream 有用，但整体难报；深学可劣于路由 |
| Potomac | 湿润中大西洋 | 0.86 | 0.93 | +0.02 | 有限残差增益 |
| Animas | 融雪落基 | 0.98 | 0.99 | +0.01 | 强自相关，上游边际小 |
| Willamette | 湿润西北 | 0.97 | 0.99 | ≈0 | 次日已近饱和，消融几乎无增益 |

## 可写成的条件句

> 上游站网的边际信息价值**不是普适常数**：在汇流结构清晰、自相关尚未饱和的湿润流域（如 James）最大；在强日自相关的融雪/大河干流（Animas、Willamette）趋近于零；在半干旱、暴雨驱动流域（Verde）技能整体更低，且非线性模型相对线性路由并不稳占优。

这直接回答审稿人：“信息价值在什么条件下成立？”

## SHAP（Potomac 1d XGB）

排序：`Q_target` ≫ 上游 Q ≫ 温度/降水 —— 与 permutation 一致。  
`figures/shap/fig_shap_xgb_1d.png`
