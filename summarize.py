"""Collects every number quoted in the manuscript and the response letter into outputs/numbers.json,
and writes the paper's tables to outputs/tables.xlsx."""
import json, platform
import numpy as np, pandas as pd
from scipy import stats
import sim

o = "outputs/"
N = {}
f1 = lambda x: f"{x:.1f}"
f0 = lambda x: f"{x:.0f}"
f2 = lambda x: f"{x:.2f}"

# ---------------- data description
base = sim.load_base_demand()
N["nprod"] = {k: v["n_products"] for k, v in base.items()}
N["price"] = {k: f"{v['avg_price']:.0f}" for k, v in base.items()}
gt = {}
for key, kw in (("e", "earbuds india"), ("s", "smart watches")):
    g = sim.load_trends(kw)
    peaks = []
    for yr in (2023, 2024, 2025):
        s = g[(g.index >= f"{yr}-07-01") & (g.index < f"{yr + 1}-03-01")]
        peaks.append(s.idxmax().strftime("%d %b %Y"))
    gt[key] = {"start": g.index.min().strftime("%d %b %Y"), "end": g.index.max().strftime("%d %b %Y"), "n": len(g),
               "mean": f1(g.mean()), "sd": f1(g.std()), "peaks": "; ".join(peaks)}
N["gt"] = gt
N["cal"] = {"diwali": "12 November 2023, 1 November 2024 and 20 October 2025",
            "prime": "15 July 2023, 20 July 2024 and 12 July 2025"}
s42 = pd.read_csv(o + "simulated_weekly_demand_seed42.csv", parse_dates=["Date"])
N["sim"] = {"start": s42.Date.min().strftime("%d %B %Y"), "end": s42.Date.max().strftime("%d %B %Y"),
            "weeks": int(s42.Date.nunique()), "rows": len(s42)}
info = json.load(open(o + "run_info_seed42.json"))
feat = pd.read_csv(o + "model_input_features_seed42.csv", parse_dates=["Date"])
N["data"] = {"weeks": int(feat.Date.nunique()), "rows": len(feat), "first": feat.Date.min().strftime("%d %B %Y"),
             "last": feat.Date.max().strftime("%d %B %Y"), "test_first": pd.Timestamp(info["first_test"]).strftime("%d %B %Y"),
             "train_weeks": info["train0_weeks"], "train_rows": info["n_train0_rows"]}
sel = pd.read_csv(o + "sarima_selection.csv")
orig = sel[(sel.order == "(1, 1, 1)") & (sel.seasonal_order == "(1, 1, 1, 52)")].iloc[0]
N["sarima"] = {"ncand": len(sel), "selmape": f1(sel.iloc[0].val_MAPE), "origmape": f1(orig.val_MAPE)}
N["ridge"] = info["ridge_alpha"]
desc = lambda p: f"{p['n_estimators']} trees, max_depth {p['max_depth']}, learning_rate {p['learning_rate']}, min_child_weight {p['min_child_weight']}"
N["xgb"] = {"A": desc(info["xgb_params_a"]), "B": desc(info["xgb_params_b"])}
import sklearn, statsmodels, xgboost
N["env"] = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "sklearn": sklearn.__version__, "statsmodels": statsmodels.__version__, "xgboost": xgboost.__version__}

# ---------------- leakage audit
la = pd.read_csv(o + "table_leakage_audit.csv")
N["leak"] = la.to_dict("records")
cnt = pd.read_csv(o + "original_data_counts.csv", index_col=0)["value"]
N["orig_counts"] = {k: float(v) for k, v in cnt.items()}

# ---------------- main experiment
mt = pd.read_csv(o + "table_main_metrics_ci.csv").set_index("Model")
def M(m):
    r = mt.loc[m]
    return {"rmse": f0(r.RMSE), "rmse_ci": f"{r.RMSE_lo:.0f}–{r.RMSE_hi:.0f}", "mae": f0(r.MAE), "mae_ci": f"{r.MAE_lo:.0f}–{r.MAE_hi:.0f}",
            "mape": f1(r.MAPE), "mape_ci": f"{r.MAPE_lo:.1f}–{r.MAPE_hi:.1f}", "r2": f2(r.R2), "r2_ci": f"{r.R2_lo:.2f}–{r.R2_hi:.2f}",
            "r2_cat": {c: f2(r[f"R2_{c}"]) for c in ("HDMICables", "USBCables", "WirelessUSBAdapters")},
            "mape_cat": {c: f1(r[f"MAPE_{c}"]) for c in ("HDMICables", "USBCables", "WirelessUSBAdapters")}}
N["main"] = {m: M(m) for m in mt.index}
dm = pd.read_csv(o + "table_dm_tests_seed42.csv")
N["dm"] = {f"{r.Model_1}|{r.Model_2}|{r.loss}": {"stat": f2(r.DM_HLN), "p": f"{r.p_value:.3f}"} for r in dm.itertuples()}
imp = pd.read_csv(o + "feature_importance_gain_seed42.csv", index_col=0)["gain"]
imp = imp / imp.sum()
N["imp"] = {"gt_share": f"{imp[[i for i in imp.index if i.startswith('GT')]].sum() * 100:.0f}",
            "top": [(k, f"{v * 100:.0f}") for k, v in imp.head(5).items()]}

# ---------------- replications
def rep_summary(r, group):
    rows = []
    for g, x in r.groupby(group, sort=False):
        row = {group: g, "n": len(x)}
        for m in ("SeasonalNaive", "SARIMA_sel", "SARIMAX", "Ridge_A", "Ridge_B", "XGB_A", "XGB_B", "XGB_B_placeboGT", "Ridge_B_placeboGT"):
            if f"{m}_MAPE" in x:
                row[m] = x[f"{m}_MAPE"].mean()
        for a, b in (("XGB_B", "XGB_A"), ("Ridge_B", "Ridge_A"), ("XGB_B", "XGB_B_placeboGT"), ("Ridge_B", "Ridge_B_placeboGT")):
            dd = x[f"{a}_MAPE"] - x[f"{b}_MAPE"]
            row[f"d_{a}_{b}"] = dd.mean(); row[f"ci_{a}_{b}"] = 1.96 * dd.std() / np.sqrt(len(dd))
            row[f"wil_{a}_{b}"] = stats.wilcoxon(dd).pvalue if (dd != 0).any() else 1.0
            row[f"better_{a}_{b}"] = int(((x[f"p_{a}_vs_{b}"] < 0.05) & (x[f"DM_{a}_vs_{b}"] < 0)).sum())
            row[f"worse_{a}_{b}"] = int(((x[f"p_{a}_vs_{b}"] < 0.05) & (x[f"DM_{a}_vs_{b}"] > 0)).sum())
        rows.append(row)
    return pd.DataFrame(rows)

try:
    rb = pd.read_csv(o + "replications_beta.csv"); rb["beta"] = rb["beta"].astype(float)
    sb = rep_summary(rb.sort_values("beta"), "beta"); sb.to_csv(o + "table_beta_summary.csv", index=False)
    N["beta"] = sb.to_dict("records")
except FileNotFoundError:
    sb = None
try:
    rs = pd.read_csv(o + "replications_sens.csv")
    ss = rep_summary(rs, "scenario"); ss.to_csv(o + "table_sensitivity_summary.csv", index=False)
    N["sens"] = ss.to_dict("records")
except FileNotFoundError:
    ss = None

# ---------------- decision modules
try:
    ls = pd.read_csv(o + "launch_window_summary.csv").to_dict("records")
    le = pd.read_csv(o + "launch_window_eval.csv")
    N["launch"] = {"summary": ls, "n": int(le.seed.nunique()),
                   "main": le[(le.seed == le.seed.min()) & (le["shift"] == 0)].iloc[0].to_dict()}
    pt = pd.read_csv(o + "table_price_sensitivity.csv"); N["price_tab"] = pt.to_dict("records")
    pk = pd.read_csv(o + "real_search_peak_timing.csv"); N["peaks"] = pk.to_dict("records")
except FileNotFoundError:
    pass

json.dump(N, open(o + "numbers.json", "w"), indent=1, default=str)
with pd.ExcelWriter(o + "tables.xlsx") as xw:
    for name, df in (("leakage_audit", la), ("main_metrics_ci", mt.reset_index()), ("dm_tests", dm), ("sarima_selection", sel),
                     ("beta_summary", sb), ("sensitivity_summary", ss)):
        if df is not None:
            df.to_excel(xw, sheet_name=name, index=False)
print(json.dumps({k: N[k] for k in ("data", "sarima", "imp")}, indent=1))
