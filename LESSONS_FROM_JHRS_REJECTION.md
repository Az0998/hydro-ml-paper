# 从 JHRS desk reject 学到的，以及转投 HSJ 的优化方向

## 拒稿原文要点（2026-07）

> science is not sufficiently innovative… Scenario simulation to evaluate the impacts of upstream streamflow and precipitation on the systemic response of a watershed does not introduce novelty to existing concepts.

编辑把稿件读成：**“上游流量/降雨影响下游”的情景模拟**——这在水文学里已是常识，不足以支撑 JHRS 的国际读者门槛。

## 同类已发表工作告诉我们什么

Environmental Modelling & Software、Journal of Hydroinformatics 等近年工作已反复证明：

- 上游流量是下游预报的关键输入；
- LSTM / CNN-LSTM 在短预见期有效；
- **“换一个流域再跑一遍 ML”** 很难被当成新贡献。

因此下一稿必须把贡献从 **“上游有没有用”** 换成 **“相对什么基准有用、对业务决策有什么用、信息上限在哪里”**。

## HSJ（二志愿）更吃什么

*Hydrological Sciences Journal*（IAHS / T&F）偏好：

- 推进对水文过程/预报系统的理解（不必新架构）；
- 方法可复现、数据公开；
- **可迁移的洞见**（条件、阈值、信息瓶颈），而非单点刷分。

适合的贡献句：

> 在波托马克出口，相对**滞后上游线性路由**基准，量化 ML 的边际增益；用**完美预见降水天花板**标明 3–7 天技能的信息瓶颈；并用 **P90 超阈 POD/FAR/CSI** 把技能翻译成防洪预警效用。

## 已落地的科学增量（本轮）

| 增量 | 目的 |
|------|------|
| 滞后上游线性回归路由基准 | 证明增益相对“物理/简单水文参考”，不是只打 Persistence |
| Oracle 未来降水天花板 | 回答“每周差是不是缺 QPF”，避免被说成空泛情景模拟 |
| 洪水超阈 CSI/POD/FAR | 把 NSE 变成区域防洪可解释指标 |
| 叙事重写（HSJ 稿） | 明确不是情景模拟，而是信息价值 + 预报效用 |

## 不再做的事

- 短期不再投 JHRS / JoH 主刊（同一创新点难翻盘）
- 不再主打 “LSTM-Attention 新结构”
