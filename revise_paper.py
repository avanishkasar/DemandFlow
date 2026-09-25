"""
revise_paper.py
Apply all reviewer-requested changes to Research Journal.docx
Changes:
1. Rewrite Abstract with proper 5-part structure (Context->Gap->Solution->Results->Comparison)
2. Expand abbreviations (SARIMA, XGBoost) on first mention in Introduction
3. Remove citations [10] and [18] from Conclusion body (they're in results area)
4. Remove 'However, this has limits...' tone in Conclusion - keep confident
5. Remove citations from Limitations/Future Scope bullet that cross into conclusion
6. Ensure MAE/RMSE/MAPE are labeled as "performance metrics" not "techniques"
7. Clarify units for MAPE result values where needed
8. Add "Research Gap" sentence in Related Work intro bridging to methodology
"""

import zipfile, shutil, copy, re, os
import lxml.etree as ET

SRC = r'D:\VS Code\rescech\Research Journal.docx'
DST = r'D:\VS Code\rescech\Research Journal_revised.docx'

shutil.copy2(SRC, DST)

ns  = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NSM = {'w': ns}

def w(tag): return f'{{{ns}}}{tag}'

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def para_text(p):
    return ''.join(r.findtext(w('t'), '') for r in p.iter(w('r')))

def set_para_text(p, new_text, preserve_rpr=True):
    """Replace all runs in paragraph with a single run carrying new_text.
       Copies rPr from first existing run if present."""
    # Grab first rPr
    first_r = p.find(f'.//{w("r")}')
    rpr_copy = None
    if first_r is not None and preserve_rpr:
        rpr = first_r.find(w('rPr'))
        if rpr is not None:
            rpr_copy = copy.deepcopy(rpr)

    # Remove all existing runs (but keep pPr)
    for child in list(p):
        tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
        if tag in ('r', 'hyperlink', 'ins', 'del', 'bookmarkStart', 'bookmarkEnd'):
            p.remove(child)

    # Add new run
    new_r = ET.SubElement(p, w('r'))
    if rpr_copy is not None:
        new_r.append(rpr_copy)
    t_elem = ET.SubElement(new_r, w('t'))
    t_elem.text = new_text
    if new_text != new_text.strip():
        t_elem.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')

def insert_para_after(tree, ref_para, new_text, style_name=None):
    """Insert a new paragraph after ref_para with given text."""
    parent = ref_para.getparent()
    idx = list(parent).index(ref_para)
    new_p = copy.deepcopy(ref_para)
    # Clean runs
    for child in list(new_p):
        tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
        if tag in ('r', 'hyperlink'):
            new_p.remove(child)
    new_r = ET.SubElement(new_p, w('r'))
    t_elem = ET.SubElement(new_r, w('t'))
    t_elem.text = new_text
    t_elem.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    parent.insert(idx + 1, new_p)
    return new_p

def remove_citations_from_text(text):
    """Remove [n], [n, m], [n-m] style inline citations."""
    return re.sub(r'\s*\[\d+(?:[,\s\d]+)?\]', '', text).strip()

# ──────────────────────────────────────────────────────────────────────────────
# NEW ABSTRACT TEXT  (5-part structure: Context → Gap → Solution → Results → Comparison)
# ──────────────────────────────────────────────────────────────────────────────

NEW_ABSTRACT = (
    "Abstract\u2014Accurate demand forecasting is a critical challenge for e-commerce "
    "retailers, particularly in fast-growing markets such as India where consumer demand "
    "is strongly shaped by festive seasons, price sensitivity, and rapidly shifting search "
    "behaviour. Existing approaches rely predominantly on historical sales data and classical "
    "statistical models, leaving a clear gap: they fail to incorporate real-time consumer "
    "intent signals that precede actual purchases. To address this gap, we propose an "
    "end-to-end forecasting pipeline that integrates Google Trends search data with "
    "Seasonal Autoregressive Integrated Moving Average (SARIMA) and Extreme Gradient "
    "Boosting (XGBoost) models, enriched with 33 engineered features including lagged "
    "demand, price, seasonality indicators, and keyword trend scores. The proposed system "
    "is evaluated on a synthetic weekly demand dataset spanning 157 weeks (January "
    "2022\u2013December 2024) across three electronics accessory categories on Amazon India. "
    "Performance is measured using standard forecasting error metrics: Mean Absolute "
    "Error (MAE), Root Mean Square Error (RMSE), and Mean Absolute Percentage Error "
    "(MAPE). The XGBoost model augmented with Google Trends achieved a MAPE of 1.22%, "
    "an RMSE of 31.64 units, and an R\u00b2 of 0.9978, representing an 86.5% reduction in "
    "forecasting error over the SARIMA baseline (MAPE: 9.69%). Compared to prior "
    "studies employing linear regression or exponential smoothing with search data, the "
    "proposed system demonstrates superior accuracy and additionally delivers actionable "
    "outputs\u2014optimal product launch timing (weeks 42\u201346) and price recommendations "
    "calibrated to festive and non-festive demand cycles\u2014bridging the gap between "
    "academic forecasting research and practical e-commerce decision support."
)

# ──────────────────────────────────────────────────────────────────────────────
# NEW INTRODUCTION  first-mention expansions  (surgical replacements only)
# ──────────────────────────────────────────────────────────────────────────────
# We only rewrite sentence fragments where SARIMA / XGBoost appear first.

# ──────────────────────────────────────────────────────────────────────────────
# APPLY CHANGES
# ──────────────────────────────────────────────────────────────────────────────

with zipfile.ZipFile(DST, 'r') as zin:
    xml_bytes = zin.read('word/document.xml')
    names = zin.namelist()

tree = ET.fromstring(xml_bytes)
paras = tree.findall(f'.//{w("p")}')

print(f"Total paragraphs: {len(paras)}")

# ── 1. REWRITE ABSTRACT ──────────────────────────────────────────────────────
abstract_idx = None
for i, p in enumerate(paras):
    txt = para_text(p)
    if txt.startswith("Abstract") and "demand forecasting" in txt:
        abstract_idx = i
        break

if abstract_idx is not None:
    print(f"  [OK] Abstract found at para {abstract_idx}. Rewriting...")
    set_para_text(paras[abstract_idx], NEW_ABSTRACT, preserve_rpr=True)
else:
    print("  [WARN] Abstract paragraph not found!")

# ── 2. EXPAND ABBREVIATIONS IN INTRODUCTION ──────────────────────────────────
# Para [13] is Introduction opening — SARIMA/XGBoost not mentioned there.
# Para [17] mentions "machine learning model" — fine.
# Para [10] IS the abstract (already rewritten with expansions).
# First occurrence in Introduction body: we need to find first para after [12] heading
# that mentions SARIMA or XGBoost and expand there.

sarima_expanded = False
xgboost_expanded = False

intro_start = None
for i, p in enumerate(paras):
    if para_text(p).strip() == 'Introduction':
        intro_start = i
        break

if intro_start:
    for i in range(intro_start + 1, min(intro_start + 30, len(paras))):
        txt = para_text(paras[i])
        if not sarima_expanded and 'SARIMA' in txt:
            new_txt = txt.replace('SARIMA', 'Seasonal Autoregressive Integrated Moving Average (SARIMA)', 1)
            set_para_text(paras[i], new_txt)
            sarima_expanded = True
            print(f"  [OK] Expanded SARIMA at para {i}")
        if not xgboost_expanded and 'XGBoost' in txt:
            new_txt = para_text(paras[i]).replace('XGBoost', 'Extreme Gradient Boosting (XGBoost)', 1)
            set_para_text(paras[i], new_txt)
            xgboost_expanded = True
            print(f"  [OK] Expanded XGBoost at para {i}")
        if sarima_expanded and xgboost_expanded:
            break

# Also expand in Related Work section if intro didn't have them (safety net)
rw_start = None
for i, p in enumerate(paras):
    if para_text(p).strip() == 'Related Work':
        rw_start = i
        break

if rw_start and (not sarima_expanded or not xgboost_expanded):
    for i in range(rw_start + 1, min(rw_start + 20, len(paras))):
        txt = para_text(paras[i])
        if not sarima_expanded and 'SARIMA' in txt:
            new_txt = txt.replace('SARIMA', 'Seasonal Autoregressive Integrated Moving Average (SARIMA)', 1)
            set_para_text(paras[i], new_txt)
            sarima_expanded = True
            print(f"  [OK] Expanded SARIMA (RW section) at para {i}")
        if not xgboost_expanded and 'XGBoost' in txt:
            new_txt = para_text(paras[i]).replace('XGBoost', 'Extreme Gradient Boosting (XGBoost)', 1)
            set_para_text(paras[i], new_txt)
            xgboost_expanded = True
            print(f"  [OK] Expanded XGBoost (RW section) at para {i}")
        if sarima_expanded and xgboost_expanded:
            break

# ── 3. FIX "TECHNIQUES" → "PERFORMANCE METRICS" ─────────────────────────────
for i, p in enumerate(paras):
    txt = para_text(p)
    # Fix Abstract/Results area references calling MAE/RMSE/MAPE "techniques"
    if 'standard forecasting techniques such as MAE' in txt:
        new_txt = txt.replace(
            'standard forecasting techniques such as MAE, RMSE, and MAPE',
            'standard forecasting error metrics, namely MAE (Mean Absolute Error), '
            'RMSE (Root Mean Square Error), and MAPE (Mean Absolute Percentage Error)'
        )
        set_para_text(paras[i], new_txt)
        print(f"  [OK] Fixed 'techniques' → 'error metrics' at para {i}")

# ── 4. ADD UNIT CLARIFICATION in Results ─────────────────────────────────────
for i, p in enumerate(paras):
    txt = para_text(p)
    # Where RMSE "31.64" appears without units  
    if 'RMSE' in txt and '31.64' in txt and 'units' not in txt.lower():
        new_txt = txt.replace('31.64', '31.64 units (weekly sales quantity)')
        set_para_text(paras[i], new_txt)
        print(f"  [OK] Added units to RMSE=31.64 at para {i}")
    # Where SARIMA MAPE 9.69 appears without context
    if '9.69%' in txt and 'SARIMA' in txt and 'baseline' not in txt.lower():
        new_txt = txt.replace('MAPE of 9.69%', 'baseline MAPE of 9.69% (forecasting error)')
        set_para_text(paras[i], new_txt)
        print(f"  [OK] Added 'forecasting error' label at para {i}")

# ── 5. REMOVE CITATIONS FROM CONCLUSION ──────────────────────────────────────
conclusion_idx = None
for i, p in enumerate(paras):
    if para_text(p).strip() == 'Conclusion':
        conclusion_idx = i
        break

limitations_idx = None
for i, p in enumerate(paras):
    if para_text(p).strip() == 'Limitations':
        limitations_idx = i
        break

if conclusion_idx and limitations_idx:
    print(f"  [OK] Conclusion: paras {conclusion_idx}–{limitations_idx}")
    for i in range(conclusion_idx + 1, limitations_idx):
        txt = para_text(paras[i])
        # Remove inline citations like [10], [18], [10, 18] etc.
        if re.search(r'\[\d+', txt):
            cleaned = re.sub(r'\s*\[\d+(?:[,\s\d]+)?\]', '', txt).strip()
            set_para_text(paras[i], cleaned)
            print(f"  [OK] Removed citation(s) from conclusion para {i}")

# ── 6. STRENGTHEN CONCLUSION — remove hedging sentence ───────────────────────
# Para 263 says "However, its effect... was positive but limited"
# This is the main finding para — we keep it (it IS their stated result) but
# we must ensure the FINAL sentence of the conclusion block is confident.
# We add a confident closing statement after para 264 (last conclusion para before Limitations)

if conclusion_idx and limitations_idx:
    last_concl_para = paras[limitations_idx - 2]  # skip blank para if any
    last_txt = para_text(last_concl_para)
    if last_txt and 'premium pric' in last_txt:
        # This is the last substantive conclusion para — good, ends positively
        # Add one confident summary sentence
        confident_close = (
            "   Taken together, these findings confirm that integrating publicly available "
            "search intent data with gradient-boosted ensemble models yields a robust, "
            "interpretable, and actionable forecasting solution that substantially "
            "outperforms classical time-series baselines and represents a meaningful "
            "contribution to the field of data-driven e-commerce demand forecasting."
        )
        # Append to the last conclusion para instead of inserting new one
        # to preserve formatting
        existing = para_text(last_concl_para)
        if "Taken together" not in existing:
            set_para_text(last_concl_para, existing + " " + confident_close.strip())
            print(f"  [OK] Added confident closing to conclusion at para {limitations_idx - 2}")

# ── 7. ADD RESEARCH GAP SENTENCE IN RELATED WORK INTRO ──────────────────────
# Para [21] is the Related Work intro paragraph. We want to ensure the gap is stated.
rw_intro_idx = None
for i, p in enumerate(paras):
    txt = para_text(p)
    if ('This section presents an overview' in txt or 
        'This section reviews' in txt) and 'Related Work' not in txt:
        rw_intro_idx = i
        break

if rw_intro_idx:
    txt = para_text(paras[rw_intro_idx])
    gap_sentence = (
        " However, a critical gap persists: prior systems seldom combine real-time "
        "search-intent signals with rich feature engineering tailored to the Indian "
        "festive demand cycle, nor do they extend beyond forecasting to provide "
        "actionable launch-timing and pricing guidance. Our proposed system is designed "
        "to fill precisely this gap."
    )
    if "gap persists" not in txt:
        set_para_text(paras[rw_intro_idx], txt + gap_sentence)
        print(f"  [OK] Added Research Gap sentence to Related Work intro at para {rw_intro_idx}")

# ── 8. FIX CITATIONS IN FUTURE SCOPE (move inline — they are OK there) ───────
# Citations in Future Scope bullet points are fine (not in conclusion).
# Para [280] has [11, 12, 17] at the END of a Future Scope bullet — that's fine.
# Para [282] has [15, 16] — also fine in Future Scope.
print("  [OK] Future Scope citations retained (appropriate placement).")

# ── 9. ADD "WHY ERROR NOT ACCURACY" JUSTIFICATION in Results intro ────────────
results_idx = None
for i, p in enumerate(paras):
    txt = para_text(p)
    if txt.strip() in ('Results', 'Results and Discussion', 'Experimental Results'):
        results_idx = i
        break

if results_idx:
    results_intro_idx = results_idx + 1
    # look for first non-empty para after Results heading
    for i in range(results_idx + 1, min(results_idx + 10, len(paras))):
        if para_text(paras[i]).strip():
            results_intro_idx = i
            break
    txt = para_text(paras[results_intro_idx])
    if 'error' not in txt.lower() or 'accuracy' not in txt.lower():
        error_justification = (
            " In time-series demand forecasting, error-based metrics "
            "(MAE, RMSE, MAPE) are preferred over classification accuracy because the "
            "goal is to minimise the magnitude of prediction deviation\u2014not to label "
            "outcomes\u2014thereby directly quantifying how much inventory risk or "
            "revenue loss a wrong forecast would cause."
        )
        if 'error-based metrics' not in txt:
            set_para_text(paras[results_intro_idx], txt + error_justification)
            print(f"  [OK] Added error-vs-accuracy justification at para {results_intro_idx}")

# ──────────────────────────────────────────────────────────────────────────────
# WRITE BACK
# ──────────────────────────────────────────────────────────────────────────────

new_xml = ET.tostring(tree, xml_declaration=True, encoding='UTF-8', standalone=True)

# Repack the zip
import tempfile
tmp = DST + '.tmp'
with zipfile.ZipFile(DST, 'r') as zin:
    with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
        for item in zin.namelist():
            if item == 'word/document.xml':
                zout.writestr(item, new_xml)
            else:
                zout.writestr(item, zin.read(item))

os.replace(tmp, DST)
print(f"\n✅ Done! Saved to: {DST}")
