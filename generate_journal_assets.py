# -*- coding: utf-8 -*-
"""按《水科学进展》规范生成全部论文图表（中文标注、期刊制图风格）。"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager

ROOT = Path(__file__).parent
FIG_DIR = ROOT / "paper" / "figures"
TAB_DIR = ROOT / "paper" / "tables"
DATA = ROOT / "data" / "potomac_daily.csv"
METRICS = ROOT / "results" / "metrics_summary.csv"
ABLATION = ROOT / "results" / "ablation_results.json"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# 中文字体
for fn in ["SimHei", "Microsoft YaHei", "SimSun"]:
    try:
        plt.rcParams["font.sans-serif"] = [fn]
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        pass
plt.rcParams["font.size"] = 10
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"


def save_fig(name):
    p = FIG_DIR / name
    plt.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  图: {p}")
    return p


# ── 表1 水文站基本信息 ──────────────────────────────────────────
def table1_stations():
    rows = [
        ["01636500", "Shenandoah River at Millville, WV", "支流", "39.29°N", "77.79°W", "—"],
        ["01638480", "Potomac River at Harpers Ferry, WV", "中游", "39.32°N", "77.73°W", "—"],
        ["01646500", "Potomac River at Washington, DC", "下游控制站", "38.95°N", "77.13°W", "14670"],
    ]
    df = pd.DataFrame(rows, columns=["站码", "站名", "河网位置", "纬度", "经度", "流域面积/km²"])
    df.to_csv(TAB_DIR / "表1_水文站基本信息.csv", index=False, encoding="utf-8-sig")
    return df


# ── 表2 数据集划分 ──────────────────────────────────────────────
def table2_split():
    df = pd.read_csv(DATA, parse_dates=["date"])
    splits = [
        ("训练集", "2000-01-01", "2018-12-31", len(df[df["date"] <= "2018-12-31"])),
        ("验证集", "2019-01-01", "2020-12-31", len(df[(df["date"] > "2018-12-31") & (df["date"] <= "2020-12-31")])),
        ("测试集", "2021-01-01", "2023-12-31", len(df[df["date"] > "2020-12-31"])),
        ("合计", "2000-01-01", "2023-12-31", len(df)),
    ]
    tdf = pd.DataFrame(splits, columns=["数据集", "起始日期", "结束日期", "样本数/d"])
    tdf.to_csv(TAB_DIR / "表2_数据集划分.csv", index=False, encoding="utf-8-sig")
    return tdf


# ── 表3 模型参数 ────────────────────────────────────────────────
def table3_params():
    rows = [
        ["输入序列长度 L/d", "30", "30", "30", "—"],
        ["隐藏层维度", "64", "64", "—", "—"],
        ["LSTM层数", "2", "2", "—", "—"],
        ["Dropout", "0.2", "0.2", "—", "—"],
        ["学习率", "0.001", "0.001", "0.05", "—"],
        ["批大小", "64", "64", "—", "—"],
        ["最大训练轮次", "80", "80", "300棵树", "60天重率定"],
        ["ARIMA阶数", "—", "—", "—", "(1,0,1)"],
        ["预报尺度/d", "1, 3, 7", "1, 3, 7", "1, 3, 7", "1, 3, 7"],
    ]
    tdf = pd.DataFrame(rows, columns=["参数", "LSTM", "LSTM-Attention", "XGBoost", "ARIMA"])
    tdf.to_csv(TAB_DIR / "表3_模型参数设置.csv", index=False, encoding="utf-8-sig")
    return tdf


# ── 表4 模型性能（来自实验结果）────────────────────────────────
def table4_metrics():
    df = pd.read_csv(METRICS)
    label_map = {
        1: "1", 3: "3", 7: "7",
    }
    rows = []
    model_cn = {
        "Persistence": "持续性预报",
        "ARIMA": "ARIMA",
        "XGBoost": "XGBoost",
        "LSTM": "LSTM",
        "LSTM-Attention": "LSTM-Attention",
    }
    for _, r in df.iterrows():
        rows.append({
            "预报尺度/d": int(r["horizon"]),
            "模型": model_cn.get(r["model"], r["model"]),
            "NSE": round(r["NSE"], 3),
            "KGE": round(r["KGE"], 3),
            "RMSE/(ft³·s⁻¹)": round(r["RMSE"], 1),
            "MAE/(ft³·s⁻¹)": round(r["MAE"], 1),
            "PBIAS/%": round(r["PBIAS"], 1),
        })
    tdf = pd.DataFrame(rows)
    tdf.to_csv(TAB_DIR / "表4_模型性能对比.csv", index=False, encoding="utf-8-sig")
    return tdf


# ── 表5 消融实验 ────────────────────────────────────────────────
def table5_ablation():
    ab = json.loads(ABLATION.read_text(encoding="utf-8"))
    rows = []
    for h in ["1", "3", "7"]:
        rows.append({
            "预报尺度/d": int(h),
            "融合上游站": round(ab["full_upstream"][h]["NSE"], 3),
            "仅本地站": round(ab["local_only"][h]["NSE"], 3),
            "NSE提升": round(ab["full_upstream"][h]["NSE"] - ab["local_only"][h]["NSE"], 3),
            "RMSE降幅/(ft³·s⁻¹)": round(
                ab["local_only"][h]["RMSE"] - ab["full_upstream"][h]["RMSE"], 1
            ),
        })
    tdf = pd.DataFrame(rows)
    tdf.to_csv(TAB_DIR / "表5_消融实验结果.csv", index=False, encoding="utf-8-sig")
    return tdf


# ── 表6 输入变量说明 ────────────────────────────────────────────
def table6_features():
    rows = [
        ["Q_target", "目标站日平均流量", "ft³·s⁻¹", "USGS-01646500"],
        ["precip_target", "目标站日降水量", "mm", "Open-Meteo"],
        ["temp_target", "目标站日平均气温", "℃", "Open-Meteo"],
        ["Q_01636500", "上游支流日平均流量", "ft³·s⁻¹", "USGS-01636500"],
        ["precip_01636500", "上游支流日降水量", "mm", "Open-Meteo"],
        ["Q_01638480", "上游中游日平均流量", "ft³·s⁻¹", "USGS-01638480"],
        ["precip_01638480", "上游中游日降水量", "mm", "Open-Meteo"],
    ]
    tdf = pd.DataFrame(rows, columns=["变量符号", "含义", "单位", "数据来源"])
    tdf.to_csv(TAB_DIR / "表6_输入变量说明.csv", index=False, encoding="utf-8-sig")
    return tdf


# ── 图1 流域示意图 ──────────────────────────────────────────────
def fig1_basin():
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    stations = [
        (1.5, 3.5, "01636500\nShenandoah河\n(支流)", "#40916C"),
        (5, 3.5, "01638480\nPotomac河\n(中游)", "#2D6A4F"),
        (8.5, 2, "01646500\nPotomac河\n(下游控制站)", "#1B4332"),
    ]
    for x, y, label, color in stations:
        box = FancyBboxPatch((x - 0.9, y - 0.7), 1.8, 1.4, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor="white", alpha=0.92)
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    ax.annotate("", xy=(7.6, 2.3), xytext=(5.9, 3.2), arrowprops=dict(arrowstyle="->", color="#444", lw=1.8))
    ax.annotate("", xy=(4.1, 3.5), xytext=(2.4, 3.5), arrowprops=dict(arrowstyle="->", color="#444", lw=1.8))
    ax.text(5, 0.4, "水流方向 →", ha="center", fontsize=10, color="#333")
    return save_fig("图1_研究流域水文站分布示意.png")


# ── 图2 流量过程线（全时段）──────────────────────────────────────
def fig2_flow_series():
    df = pd.read_csv(DATA, parse_dates=["date"])
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.plot(df["date"], df["Q_target"], color="#2D6A4F", linewidth=0.5, label="日平均流量")
    ax.axvspan(pd.Timestamp("2021-01-01"), pd.Timestamp("2023-12-31"), alpha=0.12, color="#E76F51", label="测试期")
    ax.set_xlabel("日期")
    ax.set_ylabel(r"流量/(ft$^3$·s$^{-1}$)")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_xlim(df["date"].min(), df["date"].max())
    return save_fig("图2_研究时段流量过程线.png")


# ── 图3-5 各尺度预报 hydrograph（复用已有图并重生成中文版）──────
def fig_hydrograph_cn(horizon):
    src = ROOT / "figures" / f"hydrograph_{horizon}d.png"
    if not src.exists():
        return None
    # 重新绘制中文版
    from run_experiment import load_data, run_horizon
    df = load_data()
    _, preds = run_horizon(df, horizon)
    fig, ax = plt.subplots(figsize=(7, 2.8))
    dates = preds["dates"]
    ax.plot(dates, preds["observed"], "k-", label="实测值", linewidth=1.0)
    ax.plot(dates, preds["LSTM-Attention"], color="#1B4332", label="LSTM-Attention", linewidth=0.9)
    ax.plot(dates, preds["LSTM"], color="#40916C", label="LSTM", linewidth=0.8, alpha=0.85)
    ax.plot(dates, preds["XGBoost"], color="#E76F51", label="XGBoost", linewidth=0.8, alpha=0.85)
    ax.plot(dates, preds["ARIMA"], color="#457B9D", label="ARIMA", linewidth=0.7, alpha=0.7)
    ax.set_xlabel("日期")
    ax.set_ylabel(r"流量/(ft$^3$·s$^{-1}$)")
    ax.legend(loc="upper right", fontsize=7, ncol=2, frameon=False)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    return save_fig(f"图{2+horizon}_{horizon}d预报流量过程线对比.png")


# ── 图6 指标对比柱状图 ───────────────────────────────────────────
def fig6_metrics_bar():
    df = pd.read_csv(METRICS)
    models = ["Persistence", "ARIMA", "XGBoost", "LSTM", "LSTM-Attention"]
    model_cn = ["持续性", "ARIMA", "XGBoost", "LSTM", "LSTM-Attn"]
    horizons = [1, 3, 7]
    colors = ["#999", "#457B9D", "#E76F51", "#40916C", "#1B4332"]
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.6))
    for ax, metric, ylab in zip(axes, ["NSE", "KGE", "RMSE"], ["NSE", "KGE", r"RMSE/(ft$^3$·s$^{-1}$)"]):
        x = np.arange(len(horizons))
        w = 0.15
        for i, (m, mc, c) in enumerate(zip(models, model_cn, colors)):
            vals = [df[(df["horizon"] == h) & (df["model"] == m)][metric].values[0] for h in horizons]
            ax.bar(x + i * w, vals, w, label=mc, color=c)
        ax.set_xticks(x + w * 2)
        ax.set_xticklabels([f"{h}d" for h in horizons])
        ax.set_ylabel(ylab)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
        if metric in ("NSE", "KGE"):
            ax.set_ylim(-0.5, 1.05)
    axes[0].legend(loc="lower left", fontsize=6, ncol=2, frameon=False)
    return save_fig("图7_不同预报尺度模型性能对比.png")


# ── 图7 散点图 ───────────────────────────────────────────────────
def fig7_scatter():
    from run_experiment import load_data, run_horizon
    df = load_data()
    _, preds = run_horizon(df, 1)
    obs, sim = preds["observed"], preds["LSTM-Attention"]
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.scatter(obs, sim, alpha=0.35, s=8, c="#2D6A4F", edgecolors="none")
    lim = [min(obs.min(), sim.min()), max(obs.max(), sim.max())]
    ax.plot(lim, lim, "k--", linewidth=0.8)
    nse = 1 - np.sum((obs - sim) ** 2) / np.sum((obs - obs.mean()) ** 2)
    ax.text(0.05, 0.92, f"NSE={nse:.3f}", transform=ax.transAxes, fontsize=9)
    ax.set_xlabel(r"实测流量/(ft$^3$·s$^{-1}$)")
    ax.set_ylabel(r"预测流量/(ft$^3$·s$^{-1}$)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    return save_fig("图7_1d预报散点图.png")


# ── 图8 消融实验 ─────────────────────────────────────────────────
def fig8_ablation():
    ab = json.loads(ABLATION.read_text(encoding="utf-8"))
    horizons = [1, 3, 7]
    full = [ab["full_upstream"][str(h)]["NSE"] for h in horizons]
    local = [ab["local_only"][str(h)]["NSE"] for h in horizons]
    x = np.arange(3)
    w = 0.32
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.bar(x - w / 2, full, w, label="融合上游站", color="#2D6A4F")
    ax.bar(x + w / 2, local, w, label="仅本地站", color="#95A842")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}d" for h in horizons])
    ax.set_ylabel("NSE")
    ax.set_xlabel("预报尺度/d")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    ax.axhline(0, color="#666", linewidth=0.6)
    return save_fig("图9_上游信息消融实验.png")


# ── 图9 模型结构示意 ─────────────────────────────────────────────
def fig9_architecture():
    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")
    boxes = [
        (0.5, 1.5, 2.2, 1.2, "多站输入序列\n(L=30 d)", "#E8F5E9"),
        (3.5, 1.5, 2, 1.2, "双层\nLSTM", "#C8E6C9"),
        (6.2, 1.5, 2.2, 1.2, "时间注意力\n机制", "#A5D6A7"),
        (9.2, 1.5, 1.8, 1.2, "全连接层", "#81C784"),
        (11.5, 1.5, 2, 1.2, r"$\hat{Q}_{t+h}$", "#66BB6A"),
    ]
    for x, y, w, h, txt, fc in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                     facecolor=fc, edgecolor="#2D6A4F", linewidth=1))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=8)
    for i in range(4):
        x0 = boxes[i][0] + boxes[i][2]
        x1 = boxes[i + 1][0]
        ax.annotate("", xy=(x1, 2.1), xytext=(x0, 2.1),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))
    return save_fig("图3_LSTM-Attention模型结构示意.png")


def main():
    print("生成表格...")
    table1_stations()
    table2_split()
    table3_params()
    table4_metrics()
    table5_ablation()
    table6_features()
    print("生成图片...")
    fig1_basin()
    fig2_flow_series()
    # 使用已有结果快速绘制 hydrograph（避免重训）
    for h, idx in [(1, 4), (3, 5), (7, 6)]:
        _plot_hydro_from_metrics(h, idx)
    fig6_metrics_bar()
    fig7_scatter_quick()
    fig8_ablation()
    fig9_architecture()  # 保存为图3
    print("全部期刊资产生成完毕。")


def _plot_hydro_from_metrics(horizon, fig_idx):
    """从已有图片数据快速绘制（读取 metrics 不重训）。"""
    # 直接读原图重绘风格 - 若有 preds 缓存则用；否则复制英文图
    pred_cache = ROOT / "results" / f"preds_{horizon}d.npz"
    if pred_cache.exists():
        d = np.load(pred_cache, allow_pickle=True)
        preds = {k: d[k] for k in d.files}
    else:
        # 导出 preds 需要重跑 - 用简化版从原图
        from run_experiment import load_data, run_horizon
        print(f"  生成 {horizon}d 预报图（需加载模型）...")
        _, preds = run_horizon(load_data(), horizon)
        np.savez(pred_cache, **preds)

    fig, ax = plt.subplots(figsize=(7, 2.8))
    dates = preds["dates"]
    ax.plot(dates, preds["observed"], "k-", label="实测值", linewidth=1.0)
    ax.plot(dates, preds["LSTM-Attention"], color="#1B4332", label="LSTM-Attention", linewidth=0.9)
    ax.plot(dates, preds["LSTM"], color="#40916C", label="LSTM", linewidth=0.8, alpha=0.85)
    ax.plot(dates, preds["XGBoost"], color="#E76F51", label="XGBoost", linewidth=0.8, alpha=0.85)
    ax.plot(dates, preds["ARIMA"], color="#457B9D", label="ARIMA", linewidth=0.7, alpha=0.7)
    ax.set_xlabel("日期")
    ax.set_ylabel(r"流量/(ft$^3$·s$^{-1}$)")
    ax.legend(loc="upper right", fontsize=7, ncol=2, frameon=False)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    save_fig(f"图{fig_idx}_{horizon}d预报流量过程线对比.png")


def fig7_scatter_quick():
    pred_cache = ROOT / "results" / "preds_1d.npz"
    if not pred_cache.exists():
        from run_experiment import load_data, run_horizon
        _, preds = run_horizon(load_data(), 1)
        np.savez(pred_cache, **preds)
    else:
        d = np.load(pred_cache, allow_pickle=True)
        preds = {k: d[k] for k in d.files}
    obs, sim = preds["observed"], preds["LSTM-Attention"]
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.scatter(obs, sim, alpha=0.35, s=8, c="#2D6A4F", edgecolors="none")
    lim = [min(obs.min(), sim.min()), max(obs.max(), sim.max())]
    ax.plot(lim, lim, "k--", linewidth=0.8)
    nse = 1 - np.sum((obs - sim) ** 2) / np.sum((obs - obs.mean()) ** 2)
    ax.text(0.05, 0.92, f"NSE={nse:.3f}", transform=ax.transAxes, fontsize=9)
    ax.set_xlabel(r"实测流量/(ft$^3$·s$^{-1}$)")
    ax.set_ylabel(r"预测流量/(ft$^3$·s$^{-1}$)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    save_fig("图8_1d预报散点图.png")


if __name__ == "__main__":
    main()
