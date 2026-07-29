# -*- coding: utf-8 -*-
"""
Build submission-ready Hydrological Sciences Journal package.
Incorporates: routing baseline, flood CSI, James transfer, oracle ceiling, real GFS QPF.
"""
from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, Inches

ROOT = Path(__file__).parent
PAPER = ROOT / "paper"
PAPER.mkdir(parents=True, exist_ok=True)

RES_UP = ROOT / "results" / "hsj_upgrade" / "hsj_upgrade_summary.json"
RES_T2 = ROOT / "results" / "tier2" / "tier2_summary.json"
RES_ENH = ROOT / "results" / "enhanced" / "enhanced_summary.json"
FIG_UP = ROOT / "figures" / "hsj_upgrade"
FIG_T2 = ROOT / "figures" / "tier2"
FIG_ENH = ROOT / "figures" / "enhanced"
FIG = ROOT / "figures"

OUT = PAPER / "HSJ_Potomac_forecast_information_value.docx"
OUT_ALT = PAPER / "HSJ_Potomac_forecast_information_value_FULLFIGS.docx"

TITLE = (
    "Information value of upstream gauges and precipitation foresight for multi-horizon "
    "streamflow forecasting: Potomac–James mid-Atlantic evidence"
)

HIGHLIGHTS = [
    "ML gains are judged against lagged-upstream routing, not persistence alone.",
    "James River replication confirms protocol transfer in the mid-Atlantic.",
    "Oracle precip lifts 3–7d NSE; raw GFS QPF needs careful use at longer leads.",
    "P90 flood CSI is useful at 1 day but weak for weekly observation-only alerts.",
    "Upstream ablation shows basin-dependent gauge value for next-day outlooks.",
]


def set_run(run, size=12, bold=False, italic=False):
    run.bold = bold
    run.italic = italic
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")


def set_style(doc):
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    style.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
    style.paragraph_format.space_after = Pt(0)
    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)
        # HSJ ScholarOne: do NOT upload manuscripts with author-added line numbers
        sect = section._sectPr
        for child in list(sect):
            if child.tag == qn("w:lnNumType"):
                sect.remove(child)


def P(doc, text, *, bold=False, italic=False, center=False, indent=True, size=12):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
    if indent and not center:
        p.paragraph_format.first_line_indent = Pt(24)
    else:
        p.paragraph_format.first_line_indent = Pt(0)
    r = p.add_run(text)
    set_run(r, size=size, bold=bold, italic=italic)
    return p


def H(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.first_line_indent = Pt(0)
    r = p.add_run(text)
    set_run(r, bold=True)
    return p


def Fig(doc, path, cap):
    if path.exists():
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Pt(0)
        p.add_run().add_picture(str(path), width=Inches(5.8))
    P(doc, cap, center=True, indent=False, size=11)


def fmt(x, nd=3):
    if x is None:
        return "n/a"
    try:
        v = float(x)
        if abs(v) >= 100:
            return f"{v:.0f}"
        return f"{v:.{nd}f}"
    except Exception:
        return "n/a"


def load():
    up = json.loads(RES_UP.read_text(encoding="utf-8")) if RES_UP.exists() else {}
    t2 = json.loads(RES_T2.read_text(encoding="utf-8")) if RES_T2.exists() else {}
    enh = json.loads(RES_ENH.read_text(encoding="utf-8")) if RES_ENH.exists() else {}
    return up, t2, enh


def build():
    for h in HIGHLIGHTS:
        assert len(h) <= 85, f"{len(h)} {h}"
    up, t2, enh = load()

    pot = t2.get("potomac", {}).get("horizons", {})
    james = t2.get("james", {}).get("horizons", {})
    qpf = t2.get("qpf_2024", {}).get("horizons", {})
    qpf_ver = t2.get("qpf_extra", {}).get("precip_verification_2024", {})
    u1 = up.get("horizons", {}).get("1", {})
    u3 = up.get("horizons", {}).get("3", {})
    u7 = up.get("horizons", {}).get("7", {})
    thr = up.get("threshold_P90_cfs", 26200)

    p1, p3, p7 = pot.get("1", {}), pot.get("3", {}), pot.get("7", {})
    j1, j3, j7 = james.get("1", {}), james.get("3", {}), james.get("7", {})
    q1, q3, q7 = qpf.get("1", {}), qpf.get("3", {}), qpf.get("7", {})

    doc = Document()
    set_style(doc)

    P(doc, TITLE, bold=True, center=True, indent=False, size=14)
    P(doc, "Senjie Zhang¹,*", center=True, indent=False)
    P(doc, "¹ [Affiliation: Department / University / City / Country]", center=True, indent=False, size=11)
    P(doc, "* Corresponding author. E-mail: [email@institution.edu]  |  ORCID: [0000-0000-0000-0000]",
      center=True, indent=False, size=11)
    P(doc, "Manuscript type: Research Article  |  Journal: Hydrological Sciences Journal",
      center=True, indent=False, size=10)

    H(doc, "Highlights")
    for h in HIGHLIGHTS:
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
        p.paragraph_format.left_indent = Pt(18)
        p.paragraph_format.first_line_indent = Pt(0)
        set_run(p.add_run("• " + h))

    H(doc, "Abstract")
    abstract = (
        "Whether upstream gauges improve downstream forecasts is no longer a novel claim. "
        "The operational question is how forecast information is partitioned among routing memory, "
        "nonlinear learning, and precipitation foresight, and whether that partition transfers "
        "across neighboring basins. We address this for two mid-Atlantic outlets—the Potomac River "
        "at Washington, D.C. (USGS 01646500) and the James River near Richmond, VA (02037500)—using "
        "public USGS discharge and Open-Meteo meteorology (2000–2023 test window) plus GFS previous-run "
        "quantitative precipitation forecasts (QPF) for 2024. Models are scored against persistence "
        "and a lagged-upstream linear routing baseline; upstream value is tested by ablation; flood "
        "utility uses P90 exceedance CSI/POD/FAR; and precip foresight is compared as "
        "observation-only, precipitation persistence, real GFS QPF, and perfect-foresight oracle. "
        f"On the Potomac, 1-day LSTM-Attention NSE = {fmt(p1.get('lstm_attention_NSE'))} exceeds "
        f"routing ({fmt(p1.get('routing_NSE'))}); James replication shows even stronger routing "
        f"({fmt(j1.get('routing_NSE'))}) with a large upstream-ablation gain "
        f"(ΔNSE = {fmt((j1.get('ablation') or {}).get('delta_NSE'))}). "
        f"Oracle precip raises longer-lead skill, while raw GFS QPF improves 1-day 2024 NSE relative "
        f"to observation-only inputs but can degrade uncorrected 3-day forecasts. Flood CSI is useful "
        f"next day (~{fmt(u1.get('flood_P90', {}).get('attn', {}).get('CSI'), 2)}) yet weak weekly. "
        "The transferable message is conditional gauge-network and QPF design guidance for "
        "mid-Atlantic short-range forecasting."
    )
    # fix ablation delta reference - use basin_compare field
    # Actually j1 from json has ablation as dict with delta_NSE
    P(doc, abstract, indent=False)
    P(doc,
      "Keywords: streamflow forecasting; information value; upstream gauges; QPF; "
      "flood warning; Potomac River; James River; LSTM",
      indent=False)

    H(doc, "1. Introduction")
    P(doc,
      "Upstream discharge is a physically expected predictor of downstream short-range streamflow. "
      "Repeating that fact for a single humid basin, framed as a scenario simulation of upstream "
      "forcing, does not constitute a sufficient contribution for an international hydrology "
      "readership. What remains useful is a sharper information-value question: relative to a "
      "simple hydrologic routing reference, how large is the residual gain from machine learning; "
      "at which leads does skill become limited by missing future rainfall rather than by gauge "
      "density; do bulk NSE gains translate into flood-threshold alert skill; and do these "
      "patterns transfer to a neighboring mid-Atlantic basin (Kratzert et al., 2019; Frame et al., "
      "2022; Nearing et al., 2021).")
    P(doc,
      "Deep learning and gradient boosting provide strong benchmarks when dense archives exist "
      "(Kratzert et al., 2018; Chen and Guestrin, 2016; Shen, 2018). Architecture novelty is "
      "secondary unless paired with hydrological insight about information pathways. Here we "
      "deliberately avoid claiming a new universal network. Instead we quantify an "
      "information partition for the Potomac outlet and test protocol transfer on the James River.")
    P(doc,
      "Objectives are to (i) measure multi-horizon skill against persistence and lagged-upstream "
      "linear routing, (ii) quantify upstream-gauge value by ablation on both basins, "
      "(iii) compare observation-only forecasts with precipitation persistence, real GFS QPF and "
      "an oracle precipitation ceiling, and (iv) report P90 flood-threshold CSI/POD/FAR.")

    H(doc, "2. Study areas and data")
    P(doc,
      "Potomac network: target USGS 01646500 (Washington, D.C.) with upstream 01636500 "
      "(Shenandoah at Millville) and 01638480 (Potomac at Harpers Ferry). James network: target "
      "02037500 (near Richmond) with upstream 02035000 (Cartersville) and 02034000 (Rivanna at "
      "Palmyra). Both are humid subtropical mid-Atlantic corridors with strong societal exposure. "
      "Daily discharge (USGS NWIS) and Open-Meteo precipitation/temperature span 2000–2024. "
      "Real QPF uses Open-Meteo GFS previous-run hourly precipitation aggregated to daily sums "
      "for lead times 1, 3 and 7 days in 2024 (Fig. 1).")
    if (FIG / "basin_schematic.png").exists():
        Fig(doc, FIG / "basin_schematic.png",
            "Fig. 1. Potomac gauge schematic used in the primary experiments (James network analogous).")

    P(doc, "Table 1. Gauge networks.", center=True, indent=False, size=11)
    t = doc.add_table(rows=7, cols=4)
    hdr = ["Basin", "Site ID", "Name", "Role"]
    rows = [
        ["Potomac", "01646500", "Potomac at Washington, DC", "Target"],
        ["Potomac", "01636500", "Shenandoah at Millville, WV", "Upstream"],
        ["Potomac", "01638480", "Potomac at Harpers Ferry, WV", "Upstream"],
        ["James", "02037500", "James near Richmond, VA", "Target"],
        ["James", "02035000", "James at Cartersville, VA", "Upstream"],
        ["James", "02034000", "Rivanna at Palmyra, VA", "Upstream"],
    ]
    for j, h in enumerate(hdr):
        t.rows[0].cells[j].text = h
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            t.rows[i + 1].cells[j].text = v

    H(doc, "3. Methods")
    H(doc, "3.1 Forecast task")
    P(doc,
      "Predict Q_target(t+h), h ∈ {1, 3, 7}, from 30-day sequences of local discharge, "
      "precipitation and temperature plus upstream discharge and precipitation. Chronological "
      "split for basin experiments: train ≤2018, validation 2019–2020, test 2021–2023. "
      "QPF experiments use train ≤2022, validation 2023, test 2024 because the previous-run "
      "archive used here begins in 2024. Pre-2024 foresight features for the QPF-trained model "
      "use precipitation persistence fill; 2024 test uses real GFS QPF.")
    H(doc, "3.2 Models and baselines")
    P(doc,
      "Persistence; lagged-upstream linear regression with lag L ∈ {0–3} selected on "
      "train+validation; XGBoost; LSTM-Attention (2×64, dropout 0.2). Upstream ablation "
      "retrains Attention on local-only inputs.")
    H(doc, "3.3 Precipitation foresight ladder")
    P(doc,
      "Four Attention configurations on Potomac 2024: (1) observation-only; (2) precipitation "
      "persistence foresight; (3) real GFS QPF foresight; (4) perfect-foresight oracle "
      "precipitation. Separately, oracle-versus-observation ceilings on the 2021–2023 Potomac "
      "window quantify the information gap without QPF archive constraints.")
    H(doc, "3.4 Evaluation")
    P(doc,
      f"NSE/KGE/RMSE/MAE/PBIAS; P90 flood threshold from training discharge "
      f"(Potomac P90 ≈ {fmt(thr, 0)} cfs) with POD/FAR/CSI; GFS QPF verified against observed "
      "daily precipitation (correlation and RMSE).")

    H(doc, "4. Results")
    H(doc, "4.1 Potomac skill against routing")
    P(doc,
      f"At 1 day, routing already reaches NSE = {fmt(p1.get('routing_NSE'))}, while Attention "
      f"reaches {fmt(p1.get('lstm_attention_NSE'))} and XGBoost {fmt(p1.get('xgboost_NSE'))} "
      f"(Figs. 2–3; Table 2). The ML residual over routing is finite (~{fmt(p1.get('attn_minus_routing'))} "
      f"NSE). At 3–7 days, observation-driven skill collapses for all methods, with Attention "
      f"retaining a larger 7-day residual ({fmt(p7.get('lstm_attention_NSE'))}) than routing "
      f"({fmt(p7.get('routing_NSE'))}). Hydrograph excerpts (Figs. 4–6) illustrate progressive "
      "loss of amplitude and timing fidelity with lead time.")
    Fig(doc, FIG_UP / "fig_routing_vs_ml.png",
        "Fig. 2. Potomac test NSE versus persistence and lagged-upstream linear routing.")
    if (FIG / "metrics_comparison.png").exists():
        Fig(doc, FIG / "metrics_comparison.png",
            "Fig. 3. Bulk NSE/KGE/RMSE comparison across 1-, 3- and 7-day horizons (Potomac).")
    for path, cap in [
        (FIG_ENH / "fig_hydro_excerpt_1d.png", "Fig. 4. Potomac test hydrograph excerpt (1-day ahead)."),
        (FIG_ENH / "fig_hydro_excerpt_3d.png", "Fig. 5. Potomac test hydrograph excerpt (3-day ahead)."),
        (FIG_ENH / "fig_hydro_excerpt_7d.png", "Fig. 6. Potomac test hydrograph excerpt (7-day ahead)."),
    ]:
        if path.exists():
            Fig(doc, path, cap)

    H(doc, "4.2 Transfer to the James River")
    P(doc,
      f"The same protocol on the James outlet yields 1-day routing NSE = {fmt(j1.get('routing_NSE'))} "
      f"and XGBoost/Attention ≈ {fmt(j1.get('xgboost_NSE'))}/{fmt(j1.get('lstm_attention_NSE'))} "
      f"(Fig. 7; Table 2). Upstream ablation ΔNSE at 1 day is "
      f"{fmt(j1.get('ablation', {}).get('delta_NSE'))} on James versus "
      f"{fmt(p1.get('ablation', {}).get('delta_NSE'))} on Potomac, indicating stronger "
      "near-term dependence on upstream gauges for the Richmond target. Patterns of "
      "horizon-dependent skill decay are shared across basins.")
    Fig(doc, FIG_T2 / "fig_potomac_james_transfer.png",
        "Fig. 7. Protocol transfer: Potomac versus James River test NSE.")

    P(doc, "Table 2. Test NSE by basin and horizon.", center=True, indent=False, size=11)
    t2 = doc.add_table(rows=7, cols=6)
    for j, h in enumerate(["Basin", "h (d)", "Pers.", "Routing", "XGB", "Attn"]):
        t2.rows[0].cells[j].text = h
    r = 1
    for bname, block in [("Potomac", pot), ("James", james)]:
        for h in ["1", "3", "7"]:
            d = block.get(h, {})
            vals = [bname, h, fmt(d.get("persistence_NSE")), fmt(d.get("routing_NSE")),
                    fmt(d.get("xgboost_NSE")), fmt(d.get("lstm_attention_NSE"))]
            for j, v in enumerate(vals):
                t2.rows[r].cells[j].text = v
            r += 1

    P(doc, "Table 3. Upstream ablation ΔNSE for LSTM-Attention (full minus local-only).",
      center=True, indent=False, size=11)
    t_ab = doc.add_table(rows=7, cols=4)
    for j, h in enumerate(["Basin", "h (d)", "Full NSE", "ΔNSE (full−local)"]):
        t_ab.rows[0].cells[j].text = h
    r = 1
    for bname, block in [("Potomac", pot), ("James", james)]:
        for h in ["1", "3", "7"]:
            d = block.get(h, {})
            ab = d.get("ablation") or {}
            vals = [bname, h, fmt(ab.get("full_NSE")), fmt(ab.get("delta_NSE"))]
            for j, v in enumerate(vals):
                t_ab.rows[r].cells[j].text = v
            r += 1

    H(doc, "4.3 Precipitation foresight: oracle ceiling and real GFS QPF")
    P(doc,
      f"On the 2021–2023 Potomac window, perfect-foresight precipitation leaves 1-day NSE nearly "
      f"unchanged (ΔNSE ≈ {fmt(u1.get('oracle_precip', {}).get('delta_NSE'))}) but raises 3-day "
      f"and 7-day NSE by {fmt(u3.get('oracle_precip', {}).get('delta_NSE'))} and "
      f"{fmt(u7.get('oracle_precip', {}).get('delta_NSE'))}, respectively (Fig. 8). Thus weekly "
      "observation-only forecasts are information-limited.")
    Fig(doc, FIG_UP / "fig_oracle_precip_ceiling.png",
        "Fig. 8. Oracle precipitation ceiling for Potomac LSTM-Attention (2021–2023 test).")

    v1 = qpf_ver.get("1", {})
    v3 = qpf_ver.get("3", {})
    P(doc,
      f"For 2024, GFS previous-run daily precip correlates with observations at "
      f"r = {fmt(v1.get('corr'), 2)} (1-day lead) and r = {fmt(v3.get('corr'), 2)} (3-day lead). "
      f"In streamflow mode, real QPF raises 1-day Attention NSE from {fmt(q1.get('obs', {}).get('NSE'))} "
      f"(obs-only) to {fmt(q1.get('qpf', {}).get('NSE'))}, approaching the oracle "
      f"({fmt(q1.get('oracle', {}).get('NSE'))}). At 3 days, however, uncorrected QPF ingestion "
      f"can degrade skill ({fmt(q3.get('qpf', {}).get('NSE'))}) relative to obs-only "
      f"({fmt(q3.get('obs', {}).get('NSE'))}), while oracle remains highest "
      f"({fmt(q3.get('oracle', {}).get('NSE'))}) (Fig. 9; Table 4). This cautions against "
      "treating raw QPF as a drop-in feature without bias-aware training.")
    Fig(doc, FIG_T2 / "fig_real_qpf_ladder.png",
        "Fig. 9. Potomac 2024 NSE ladder: obs-only, precip persistence, real GFS QPF, oracle.")

    P(doc, "Table 4. Potomac 2024 Attention NSE under precip foresight options.",
      center=True, indent=False, size=11)
    t3 = doc.add_table(rows=4, cols=5)
    for j, h in enumerate(["h (d)", "Obs-only", "Persist P", "GFS QPF", "Oracle"]):
        t3.rows[0].cells[j].text = h
    for i, (h, block) in enumerate([("1", q1), ("3", q3), ("7", q7)]):
        vals = [h, fmt(block.get("obs", {}).get("NSE")), fmt(block.get("persist", {}).get("NSE")),
                fmt(block.get("qpf", {}).get("NSE")), fmt(block.get("oracle", {}).get("NSE"))]
        for j, v in enumerate(vals):
            t3.rows[i + 1].cells[j].text = v

    H(doc, "4.4 Flood-threshold, seasonal skill, uncertainty and interpretability")
    P(doc,
      f"Using Potomac training P90, 1-day Attention CSI = "
      f"{fmt(u1.get('flood_P90', {}).get('attn', {}).get('CSI'), 2)} "
      f"(POD = {fmt(u1.get('flood_P90', {}).get('attn', {}).get('POD'), 2)}, "
      f"FAR = {fmt(u1.get('flood_P90', {}).get('attn', {}).get('FAR'), 2)}), exceeding routing "
      f"CSI = {fmt(u1.get('flood_P90', {}).get('routing', {}).get('CSI'), 2)} (Fig. 10; Table 5). "
      "CSI collapses at weekly lead times under observation-only forcing. Seasonal NSE panels "
      "(Figs. 11–13), multi-seed uncertainty bands (Figs. 14–15), permutation importance "
      "(Fig. 16) and the weekly scatter plot (Fig. 17) provide supporting diagnostics.")
    Fig(doc, FIG_UP / "fig_flood_csi.png",
        "Fig. 10. Potomac P90 flood-threshold critical success index.")

    P(doc, "Table 5. Potomac P90 flood-threshold CSI.", center=True, indent=False, size=11)
    t5 = doc.add_table(rows=4, cols=4)
    for j, h in enumerate(["h (d)", "Routing CSI", "XGB CSI", "Attn CSI"]):
        t5.rows[0].cells[j].text = h
    for i, (h, u) in enumerate([("1", u1), ("3", u3), ("7", u7)]):
        f = u.get("flood_P90", {})
        vals = [h, fmt(f.get("routing", {}).get("CSI"), 2),
                fmt(f.get("xgb", {}).get("CSI"), 2),
                fmt(f.get("attn", {}).get("CSI"), 2)]
        for j, v in enumerate(vals):
            t5.rows[i + 1].cells[j].text = v

    for path, cap in [
        (FIG_ENH / "fig_seasonal_nse_1d.png", "Fig. 11. Seasonal NSE at the 1-day horizon (Potomac)."),
        (FIG_ENH / "fig_seasonal_nse_3d.png", "Fig. 12. Seasonal NSE at the 3-day horizon (Potomac)."),
        (FIG_ENH / "fig_seasonal_nse_7d.png", "Fig. 13. Seasonal NSE at the 7-day horizon (Potomac)."),
        (FIG_ENH / "fig_uncertainty_attn_1d.png", "Fig. 14. Multi-seed uncertainty band for LSTM-Attention (1-day)."),
        (FIG_ENH / "fig_uncertainty_attn_7d.png", "Fig. 15. Multi-seed uncertainty band for LSTM-Attention (7-day)."),
        (FIG_ENH / "fig_permutation_importance_1d.png", "Fig. 16. Permutation importance for Potomac 1-day Attention."),
        (FIG / "scatter_7d.png", "Fig. 17. Observed versus predicted discharge at the weekly horizon."),
    ]:
        if path.exists():
            Fig(doc, path, cap)

    H(doc, "5. Discussion")
    P(doc,
      "The contribution is not that upstream flow matters, but that mid-Atlantic short-range "
      "forecast skill is routing-dominated at 1 day, with a basin-dependent upstream-ablation "
      "signal (stronger on James), a finite ML residual over linear routing, and a clear "
      "precipitation-foresight bottleneck beyond about three days. Real GFS QPF can help next-day "
      "products yet is not a substitute for bias-aware integration at longer leads, even though "
      "oracle experiments show substantial unused skill.")
    P(doc,
      "These findings align with broader LSTM rainfall–runoff evidence and caution against "
      "overstating weekly deterministic skill without precipitation foresight "
      "(Kratzert et al., 2018, 2019; Frame et al., 2022; Nearing et al., 2021). Limitations "
      "include reanalysis meteorology for non-QPF inputs, a single year of previous-run QPF, "
      "omitted regulation/tidal effects, and two-basin rather than continental scope.")

    H(doc, "6. Conclusions")
    P(doc,
      "Main findings. Across Potomac and James outlets, next-day forecasts are largely explained "
      "by lagged upstream routing plus local memory; ML adds a finite residual. Precipitation "
      "foresight, not additional architecture complexity, is the binding constraint for 3–7-day "
      "deterministic skill.")
    P(doc,
      "Broader impacts. Mid-Atlantic forecasting design should protect upstream telemetry for "
      "1-day flood-threshold alerts and couple weekly products to QPF with explicit bias handling.")
    P(doc,
      "Outlook. Extending previous-run QPF archives, testing bias-corrected QPF predictors, and "
      "adding further Atlantic-coast basins would strengthen transfer claims.")

    H(doc, "Data availability")
    P(doc,
      "USGS NWIS discharge (https://waterservices.usgs.gov/) and Open-Meteo archive / previous-run "
      "APIs (https://open-meteo.com/) are public. Processed tables and scripts are available from "
      "the corresponding author upon reasonable request.",
      indent=False)
    H(doc, "Disclosure statement")
    P(doc, "The authors report there are no competing interests to declare.", indent=False)
    H(doc, "Funding")
    P(doc,
      "This research received no specific grant from any funding agency in the public, commercial, "
      "or not-for-profit sectors.",
      indent=False)
    H(doc, "Acknowledgments")
    P(doc, "The authors thank USGS and Open-Meteo for open data services.", indent=False)
    H(doc, "References")
    refs = [
        "Chen, T. and Guestrin, C., 2016. XGBoost: A scalable tree boosting system. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 785–794.",
        "Frame, J.M., et al., 2022. Deep learning rainfall–runoff predictions of extreme events. Hydrology and Earth System Sciences, 26, 3377–3392.",
        "Gupta, H.V., Kling, H., Yilmaz, K.K. and Martinez, G.F., 2009. Decomposition of the mean squared error and NSE performance criteria. Journal of Hydrology, 377, 80–91.",
        "Kratzert, F., et al., 2018. Rainfall–runoff modelling using Long Short-Term Memory (LSTM) networks. Hydrology and Earth System Sciences, 22, 6005–6022.",
        "Kratzert, F., et al., 2019. Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets. Hydrology and Earth System Sciences, 23, 5089–5110.",
        "Nash, J.E. and Sutcliffe, J.V., 1970. River flow forecasting through conceptual models part I — A discussion of principles. Journal of Hydrology, 10, 282–290.",
        "Nearing, G.S., et al., 2021. What role does hydrological science play in the age of machine learning? Water Resources Research, 57, e2020WR028091.",
        "Shen, C., 2018. A transdisciplinary review of deep learning research and its relevance for water resources scientists. Water Resources Research, 54, 8558–8593.",
        "U.S. Geological Survey, 2024. National Water Information System. https://waterdata.usgs.gov/nwis.",
    ]
    for ref in refs:
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
        p.paragraph_format.first_line_indent = Pt(-24)
        p.paragraph_format.left_indent = Pt(24)
        set_run(p.add_run(ref))

    try:
        doc.save(OUT)
        saved = OUT
    except PermissionError:
        doc.save(OUT_ALT)
        saved = OUT_ALT
        print("NOTE: original docx locked; wrote", OUT_ALT)

    # Highlights
    (PAPER / "HSJ_highlights.txt").write_text("\n".join("• " + h for h in HIGHLIGHTS) + "\n", encoding="utf-8")
    hl = Document()
    set_style(hl)
    P(hl, "Highlights", bold=True, center=True, indent=False, size=14)
    for h in HIGHLIGHTS:
        p = hl.add_paragraph()
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
        set_run(p.add_run("• " + h))
    hl.save(PAPER / "HSJ_highlights.docx")

    # DOI / competing interest
    doi = Document()
    set_style(doi)
    P(doi, "Declaration of Interest Statement", bold=True, center=True, indent=False, size=14)
    P(doi,
      "The authors declare that they have no known competing financial interests or personal "
      "relationships that could have appeared to influence the work reported in this paper.",
      indent=False)
    doi.save(PAPER / "HSJ_Declaration_of_Interest.docx")

    # Do not overwrite hand-tuned cover letter / checklist if present with cleaner versions.
    # Cover letter is maintained in paper/HSJ_cover_letter.txt (no prior-rejection language).

    print("Wrote", saved)
    return saved


if __name__ == "__main__":
    build()
