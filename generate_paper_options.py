from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import datetime

doc = Document()

# ── Page margins ────────────────────────────────────────────────────────────
section = doc.sections[0]
section.top_margin    = Cm(2.0)
section.bottom_margin = Cm(2.0)
section.left_margin   = Cm(2.5)
section.right_margin  = Cm(2.5)

# ── Helper functions ─────────────────────────────────────────────────────────
def set_font(run, name="Calibri", size=11, bold=False, color=None):
    run.font.name  = name
    run.font.size  = Pt(size)
    run.font.bold  = bold
    if color:
        run.font.color.rgb = RGBColor(*color)

def add_heading(doc, text, level=1, color=(30, 90, 160)):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run(text)
    set_font(run, size=14 if level == 1 else 12, bold=True, color=color)
    p.paragraph_format.space_before = Pt(14 if level == 1 else 8)
    p.paragraph_format.space_after  = Pt(4)
    return p

def add_label(doc, label, value, label_color=(0, 102, 153)):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after  = Pt(2)
    r1 = p.add_run(label + " ")
    set_font(r1, bold=True, size=11, color=label_color)
    r2 = p.add_run(value)
    set_font(r2, size=11)
    return p

def add_body(doc, text, italic=False):
    p = doc.add_paragraph(text)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after  = Pt(4)
    for run in p.runs:
        set_font(run, size=11)
        run.italic = italic
    return p

def add_divider(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after  = Pt(2)
    run = p.add_run("─" * 80)
    set_font(run, size=9, color=(180, 180, 180))

def shade_cell(cell, fill_hex="D9E8F5"):
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  fill_hex)
    tcPr.append(shd)

# ── Title block ──────────────────────────────────────────────────────────────
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("Research Paper Options — Next Paper Planning")
set_font(r, size=18, bold=True, color=(15, 55, 120))
p.paragraph_format.space_after = Pt(4)

p2 = doc.add_paragraph()
p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
r2 = p2.add_run(f"Prepared: {datetime.date.today().strftime('%B %d, %Y')}  |  Author: Avanish Kasar")
set_font(r2, size=10, color=(120, 120, 120))
p2.paragraph_format.space_after = Pt(12)

doc.add_paragraph()  # spacer

# ════════════════════════════════════════════════════════════════════════════
# SECTION A  — THREE CANDIDATE PAPERS
# ════════════════════════════════════════════════════════════════════════════
add_heading(doc, "SECTION A: Three Candidate Papers", level=1)
add_divider(doc)

# ── Paper 1 ─────────────────────────────────────────────────────────────────
add_heading(doc, "Paper 1 — Kolmogorov-Arnold Networks (KAN)", level=2, color=(180, 60, 0))

add_label(doc, "Reference:", "Liu et al., MIT — \"KAN: Kolmogorov-Arnold Networks\" (ICLR 2025 | arXiv: 2404.19756)")
add_label(doc, "Source Code:", "github.com/KindXiaoming/pykan  ✅ Full code available")

add_label(doc, "Description:", "")
add_body(doc,
    "KAN introduces a fundamentally new neural network architecture that replaces the fixed activation "
    "functions of traditional MLPs with learnable univariate functions placed on the edges between nodes. "
    "This makes the network more mathematically interpretable and often more parameter-efficient for "
    "structured, non-linear data. Accepted at ICLR 2025 and one of the most talked-about architecture "
    "innovations in recent AI research.")

add_label(doc, "What the paper does:", "")
add_body(doc,
    "The paper proves that any continuous multivariate function can be decomposed into a composition of "
    "simpler univariate functions (Kolmogorov-Arnold Theorem), then builds a trainable neural network "
    "around this idea. It shows KAN outperforms MLPs of the same size on scientific datasets while "
    "producing human-readable symbolic formulas from the learned functions.")

add_label(doc, "What we will do differently:", "")
add_body(doc,
    "Apply KAN to a new domain — such as energy load or clinical vital-sign time series — and compare "
    "it against XGBoost (our Paper 1 baseline) and a standard LSTM. We will evaluate whether KAN's "
    "interpretability advantage over black-box deep learning translates into measurable accuracy gains "
    "on real-world sequential data, which the original paper does not test.", italic=True)

doc.add_paragraph()

# ── Paper 2 ─────────────────────────────────────────────────────────────────
add_heading(doc, "Paper 2 — PatchTST / iTransformer (Transformer for Time Series)", level=2, color=(0, 120, 50))

add_label(doc, "References:",
    "Nie et al., \"A Time Series is Worth 64 Words: PatchTST\" (ICLR 2023 | arXiv: 2211.14730)\n"
    "           Liu et al., \"iTransformer\" (ICLR 2024 | arXiv: 2310.06625)")
add_label(doc, "Source Code:",
    "github.com/yuqinie98/PatchTST  ✅  |  github.com/thuml/iTransformer  ✅")

add_label(doc, "Description:", "")
add_body(doc,
    "PatchTST adapts the Transformer (the same architecture behind ChatGPT) for time series by treating "
    "fixed-length segments ('patches') of the series as 'tokens', similar to words in a sentence. "
    "iTransformer takes a different angle, inverting the attention mechanism so it operates across "
    "multiple variables (channels) rather than across time steps. Both models set new state-of-the-art "
    "benchmarks on standard long-horizon forecasting tasks.")

add_label(doc, "What the paper does:", "")
add_body(doc,
    "PatchTST demonstrates that patching + channel-independence drastically reduces the sequence length "
    "the Transformer must process, improving both speed and accuracy. iTransformer shows that attending "
    "across variates first captures inter-variable correlations that purely temporal attention misses. "
    "Both papers include comprehensive benchmarks across ETT, Weather, and Traffic datasets.")

add_label(doc, "What we will do differently:", "")
add_body(doc,
    "Apply these architectures to a new, underexplored domain (e.g., air quality index prediction, "
    "hospital patient vitals, or electricity grid data for an Indian city). We will benchmark both "
    "PatchTST and iTransformer against our XGBoost baseline and a standard LSTM, providing an "
    "applied comparison in a domain where no such study currently exists.", italic=True)

doc.add_paragraph()

# ── Paper 3 ─────────────────────────────────────────────────────────────────
add_heading(doc, "Paper 3 — CastFlow (Agentic Forecasting Framework)", level=2, color=(100, 0, 160))

add_label(doc, "Reference:", "\"CastFlow: A Dynamic Agentic Forecasting Framework\" (arXiv 2026, cs.LG)")
add_label(doc, "Source Code:", "arXiv 2026 — code link in paper  ⚠️ Partial / in-progress")

add_label(doc, "Description:", "")
add_body(doc,
    "CastFlow introduces an agentic AI system for time-series forecasting where a general-purpose LLM "
    "acts as a 'planner' that decides which deep learning model to run, monitors the output quality, "
    "and autonomously retries with a different strategy if the forecast is poor. It implements a "
    "Plan → Act → Forecast → Reflect loop, effectively turning forecasting into an autonomous workflow "
    "rather than a fixed pipeline.")

add_label(doc, "What the paper does:", "")
add_body(doc,
    "The paper proposes role-specialized design: a reasoning LLM selects strategies and a fine-tuned "
    "domain-specific neural network handles the actual numerical prediction. Results show agentic "
    "workflows improve robustness on distribution-shifted and noisy data compared to static single-model "
    "approaches.")

add_label(doc, "What we will do differently:", "")
add_body(doc,
    "Build an open-source, lightweight version of this framework using freely available tools (LangGraph "
    "+ PyTorch). Swap the closed-API LLM for an open-source model (e.g., LLaMA-3 or Gemini API). "
    "Apply the framework to a new domain dataset, add an explainability step where the agent also "
    "generates a written natural-language report, and compare agentic vs. non-agentic deep learning "
    "performance side-by-side.", italic=True)

doc.add_paragraph()
add_divider(doc)
doc.add_paragraph()

# ════════════════════════════════════════════════════════════════════════════
# SECTION B  — 4 COMBINATION OPTIONS (Paper 2 + Paper 3)
# ════════════════════════════════════════════════════════════════════════════
add_heading(doc, "SECTION B: 4 Options — Combining Paper 2 + Paper 3 into One Final Paper", level=1)
add_divider(doc)

p_intro = doc.add_paragraph(
    "The following four options each combine the deep learning architecture of Paper 2 (Transformers for "
    "time series) with the agentic workflow of Paper 3 (CastFlow). Each option targets a different "
    "domain and contribution angle.")
for run in p_intro.runs:
    set_font(run, size=11, color=(80, 80, 80))
    run.italic = True
p_intro.paragraph_format.space_after = Pt(12)

# ── Option 1 ─────────────────────────────────────────────────────────────────
add_heading(doc, "Option 1 — Agentic Energy Load Forecasting", level=2, color=(200, 100, 0))
add_label(doc, "Draft Title:",
    '"An Agentic Deep Learning Framework for Autonomous Electricity Load Forecasting '
    'Using Transformer Architectures"')
add_label(doc, "Domain:", "Energy / Smart Grid (completely new domain, huge public datasets available)")
add_label(doc, "What you build:", "")
add_body(doc,
    "Build a LangGraph-based agent that autonomously decides between PatchTST and iTransformer based "
    "on input data characteristics (e.g., how long the forecast horizon is). The agent also fetches "
    "real-time weather features as external context, runs the selected model, and generates a "
    "plain-English summary of predicted peak load hours — useful for grid operators.")
add_label(doc, "Key Contribution:", "")
add_body(doc,
    "First study to combine iTransformer/PatchTST with an autonomous agentic model-selection pipeline "
    "in the smart grid domain. Demonstrates that agentic model selection reduces MAPE by dynamically "
    "matching model architecture to data patterns.", italic=True)
add_label(doc, "Datasets:", "UCI Electricity, ETT (Electricity Transformer Temperature), AEMO (Australia grid)")
add_label(doc, "Feasibility:", "⭐⭐⭐ Medium — all datasets public, Colab friendly")
add_label(doc, "Target Journal:", "Applied Energy, Energy & AI, IEEE Transactions on Smart Grid")

doc.add_paragraph()

# ── Option 2 ─────────────────────────────────────────────────────────────────
add_heading(doc, "Option 2 — Agentic Clinical Vital Signs Monitoring", level=2, color=(0, 130, 80))
add_label(doc, "Draft Title:",
    '"AutoClinical-Agent: An Autonomous Transformer-Based System for Patient Vital Sign '
    'Forecasting and Alert Generation in ICUs"')
add_label(doc, "Domain:", "Healthcare / Medical Informatics (high-impact, fast publication)")
add_label(doc, "What you build:", "")
add_body(doc,
    "A multi-agent system where PatchTST forecasts the next 6-hour trajectory of patient vitals "
    "(heart rate, blood pressure, SpO2). A second 'critic' agent monitors the forecast and triggers "
    "an alert if the prediction crosses a clinical threshold (e.g., SpO2 < 90%). A third agent "
    "generates a written clinical note: 'Patient #42 shows elevated HR trend, recommend review.'")
add_label(doc, "Key Contribution:", "")
add_body(doc,
    "First agentic pipeline integrating Transformer-based forecasting with rule-based clinical "
    "alerting and automatic report generation. Bridges the gap between ML prediction and actionable "
    "clinical decision support, a widely cited research gap in medical AI literature.", italic=True)
add_label(doc, "Datasets:", "PhysioNet MIMIC-III/IV (public, requires registration), MIT-BIH")
add_label(doc, "Feasibility:", "⭐⭐ Medium-Low — dataset requires free registration but code is well-supported")
add_label(doc, "Target Journal:", "Journal of Biomedical Informatics, npj Digital Medicine, IEEE JBHI")

doc.add_paragraph()

# ── Option 3 ─────────────────────────────────────────────────────────────────
add_heading(doc, "Option 3 — Agentic Air Quality Prediction", level=2, color=(0, 80, 160))
add_label(doc, "Draft Title:",
    '"AQI-Agent: An Agentic Transformer Framework for Real-Time Air Quality Index '
    'Forecasting in Indian Urban Environments"')
add_label(doc, "Domain:", "Environmental Science / Smart Cities (strong India-specific angle)")
add_label(doc, "What you build:", "")
add_body(doc,
    "An agentic forecasting pipeline that autonomously retrieves real-time PM2.5/PM10/NO2 sensor "
    "readings from Indian cities (CPCB Open API), feeds them into iTransformer for 24-hour AQI "
    "forecasting, and uses an LLM agent to produce daily air quality advisories ('Delhi AQI forecast: "
    "Severe. Avoid outdoor activity between 8AM-11AM.'). Agent also learns which cities' models "
    "need retraining when drift is detected.")
add_label(doc, "Key Contribution:", "")
add_body(doc,
    "First India-specific agentic AQI forecasting system using Transformer architectures and open "
    "government data. Addresses a clear public health need with a novel autonomous pipeline that "
    "self-monitors for distributional drift — a problem unaddressed in existing AQI literature.", italic=True)
add_label(doc, "Datasets:", "CPCB India Open Data (free API), UCI Beijing PM2.5, OpenAQ")
add_label(doc, "Feasibility:", "⭐⭐⭐⭐ High — free Indian government API, very relevant, strong novelty")
add_label(doc, "Target Journal:", "Environmental Research Letters, Atmospheric Environment, IEEE Access")

doc.add_paragraph()

# ── Option 4 ─────────────────────────────────────────────────────────────────
add_heading(doc, "Option 4 — Agentic Stock Market Direction Prediction", level=2, color=(120, 0, 100))
add_label(doc, "Draft Title:",
    '"QuantAgent: A Multi-Agent Deep Learning Framework for Financial Time Series '
    'Direction Forecasting Using Transformer Architectures"')
add_label(doc, "Domain:", "Quantitative Finance / FinTech (highest career impact)")
add_label(doc, "What you build:", "")
add_body(doc,
    "A multi-agent system with three roles: (1) a Data Agent that collects NSE/BSE stock OHLCV data "
    "and fetches related news headlines; (2) a Forecast Agent that runs PatchTST to predict the next "
    "5-day price direction (up/down); and (3) a Reasoning Agent (LLM) that combines the numerical "
    "forecast with news sentiment to produce a structured investment recommendation report with "
    "confidence scores and key risk factors.")
add_label(doc, "Key Contribution:", "")
add_body(doc,
    "First open-source multi-agent system combining PatchTST price forecasting with LLM-based news "
    "sentiment reasoning for Indian equity markets (NSE/BSE). Provides a transparent, auditable "
    "decision pipeline addressing the interpretability gap in automated financial AI systems.", italic=True)
add_label(doc, "Datasets:", "Yahoo Finance API (free), NSE India Open Data, GDELT News Dataset")
add_label(doc, "Feasibility:", "⭐⭐⭐ Medium — data easily available, career value is highest of all options")
add_label(doc, "Target Journal:", "Expert Systems with Applications, Finance Research Letters, IEEE Access")

doc.add_paragraph()
add_divider(doc)
doc.add_paragraph()

# ── Summary comparison table ──────────────────────────────────────────────────
add_heading(doc, "Quick Decision Table", level=1)

headers = ["Option", "Domain", "Novelty", "Difficulty", "Career Impact", "Colab?", "Best For"]
rows = [
    ["1 — Energy",    "Smart Grid",      "🔴 High",  "⭐⭐⭐ Med",    "High",         "✅ Yes", "Clean energy / MLOps roles"],
    ["2 — Healthcare","Medical AI",      "🔴 High",  "⭐⭐ Med-Low", "Very High",    "⚠️ GPU", "Healthcare AI / research roles"],
    ["3 — AQI India", "Env. Science",    "🔴 V.High","⭐⭐⭐⭐ Easy", "High",         "✅ Yes", "Public sector / India-specific"],
    ["4 — Finance",   "FinTech / Quant", "🔴 High",  "⭐⭐⭐ Med",    "Highest 🏆",  "✅ Yes", "Fintech / quant trading roles"],
]

table = doc.add_table(rows=1 + len(rows), cols=len(headers))
table.style = "Table Grid"
table.alignment = WD_TABLE_ALIGNMENT.CENTER

# Header row
for i, h in enumerate(headers):
    cell = table.cell(0, i)
    shade_cell(cell, "1E5A9C")
    run  = cell.paragraphs[0].add_run(h)
    set_font(run, bold=True, size=10, color=(255, 255, 255))
    cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

# Data rows
for ri, row_data in enumerate(rows):
    fill = "EAF2FB" if ri % 2 == 0 else "FFFFFF"
    for ci, val in enumerate(row_data):
        cell = table.cell(ri + 1, ci)
        shade_cell(cell, fill)
        run  = cell.paragraphs[0].add_run(val)
        set_font(run, size=10)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_paragraph()

# ── Footer note ──────────────────────────────────────────────────────────────
p_foot = doc.add_paragraph(
    "Note: All four options build directly on the deep learning skills from Paper 2 (PatchTST/iTransformer) "
    "and the agentic workflow principles from Paper 3 (CastFlow). The recommended starting point is "
    "Option 3 (AQI India) for fastest execution or Option 4 (Finance) for maximum career impact.")
for run in p_foot.runs:
    set_font(run, size=10, color=(100, 100, 100))
    run.italic = True

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = r"d:\VS Code\rescech\Paper_Options_NextResearch.docx"
doc.save(out_path)
print(f"Document saved: {out_path}")
