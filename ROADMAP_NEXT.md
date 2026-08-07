# 下一阶段升级路线图（HSJ 审稿 / 扩刊）

> 原则：先回答审稿人“条件依赖性”，再碰业务概率预报与大模型。

## 优先级矩阵

| 方向 | 星级 | 对 HSJ 审稿 | 工作量 | 本仓库落地状态 |
|------|------|-------------|--------|----------------|
| 多气候带同协议迁移（湿润/干旱/融雪） | ⭐⭐⭐ | **最高**（条件何时成立） | 中 | **本轮主攻：跑通** |
| GEFS 集合 → 概率径流 | ⭐⭐⭐⭐ | 高（业务对接） | 高 | 接口骨架 + 设计说明 |
| 水文基础模型 fine-tune | ⭐⭐⭐⭐ | 中高（新方向） | 很高 | 设计说明 + 对照协议 |
| SHAP/LIME 可解释性 | （你列的） | 高（物理可信度） | 低–中 | **本轮加 SHAP** |
| 方法快报（Hydroinformatics / HESS） | ⭐⭐ | 曝光 | 低 | 选题提纲 |

## 建议执行顺序

```
① 多气候带迁移  →  ② SHAP  →  ③ GEFS 概率预报  →  ④ 基础模型对照  →  ⑤ 方法快报
```

### ① 多气候带（回答“信息价值在什么条件下成立”）

已有：Potomac / James（湿润中大西洋）  
新增：

| 气候类型 | 流域 | Target USGS |
|----------|------|-------------|
| 湿润西北 | Willamette @ Salem | 14191000 |
| 融雪落基 | Animas @ Durango | 09363500 |
| 半干旱西南 | Verde @ Camp Verde | 09506000 |

统一协议：Persistence / Routing-LR / XGBoost / LSTM-Attention + 上游消融，1/3/7 天，同一时间划分。

**可发表结论形态：**  
“上游信息价值在湿润、汇流时间匹配的日尺度最强；融雪流域更依赖季节记忆；半干旱流域消融增益与暴雨驱动相关……”

### ② SHAP

在 Potomac 1 天 Attention（或 XGBoost）上做特征/通道级 SHAP，与已有 permutation 交叉验证。

### ③ GEFS（设计要点）

- 输入：集合降水成员（或 Open-Meteo ensemble）  
- 输出：径流分位数 / 超阈概率（P90）  
- 指标：CRPS、reliability、CSI@概率阈值  
- 相对确定性 QPF：展示业务预警增益

### ④ 基础模型

- 基线：现有 LSTM-Attention（小样本训练）  
- 对照：公开水文/气候基础模型（若权重可得）或简化的多流域预训练→单流域 fine-tune  
- 问题：通用表征是否降低对本地长序列的依赖

### ⑤ 方法快报选题

短文焦点二选一：  
- “Routing baseline + information ceiling” 方法笔记；或  
- “QPF ladder for streamflow ML” 可复现协议

## 脚本入口

```bash
python download_climate_basins.py   # 新流域数据
python run_climate_transfer.py      # 多气候带同协议
python run_shap_explain.py          # SHAP
# 后续：
# python run_gefs_probabilistic.py
# python run_foundation_finetune.py
```
