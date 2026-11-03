"""Main experiment (seed 42, beta=0.20, lag=2, sigma=0.10): all models, variants, CIs and DM tests."""
import json, time
import numpy as np, pandas as pd
import sim, pipeline as P

t0 = time.time()
sales, gt = sim.simulate(seed=42, beta=0.2)
sales.to_csv("outputs/simulated_weekly_demand_seed42.csv", index=False)
out, imp, info, df = P.run_once(sales, gt, extra_variants=True)
df.to_csv("outputs/model_input_features_seed42.csv", index=False)
out.to_csv("outputs/test_predictions_seed42.csv", index=False)
imp.to_csv("outputs/feature_importance_gain_seed42.csv", header=["gain"])
models = ["Naive", "SeasonalNaive", "SARIMA_orig", "SARIMA_sel", "SARIMAX", "Ridge_B", "XGB_A", "XGB_B",
          "XGB_B_lagsonly", "XGB_B_now", "XGB_B_placeboGT", "Ridge_A", "Ridge_B_placeboGT", "XGB_B_noDemandLags", "XGB_A_raw", "XGB_B_raw", "XGB_B_orig"]
rows = []
for m in models:
    r = P.metrics(out["y"], out[m]); lo, hi = P.block_bootstrap_ci(out, m, n_boot=1000)
    row = {"Model": m}
    for k in ("RMSE", "MAE", "MAPE", "R2"):
        row[k] = r[k]; row[k + "_lo"] = lo[k]; row[k + "_hi"] = hi[k]
    for c in out["Category"].unique():
        s = out[out["Category"] == c]
        row["R2_" + c] = P.metrics(s["y"], s[m])["R2"]
        row["MAPE_" + c] = P.metrics(s["y"], s[m])["MAPE"]
    rows.append(row)
res = pd.DataFrame(rows)
res.to_csv("outputs/table_main_metrics_ci.csv", index=False)
dm = []
pairs = [("XGB_B", "XGB_A"), ("XGB_B", "SARIMA_sel"), ("XGB_B", "SARIMAX"), ("XGB_B", "SeasonalNaive"),
         ("XGB_B", "Ridge_B"), ("Ridge_B", "Ridge_A"), ("Ridge_B", "SARIMAX"), ("Ridge_B", "Ridge_B_placeboGT"), ("Ridge_B", "SeasonalNaive"), ("XGB_A", "SARIMA_sel"), ("SARIMAX", "SARIMA_sel"), ("XGB_B", "XGB_B_placeboGT"),
         ("XGB_B_placeboGT", "XGB_A"), ("XGB_B", "Naive"), ("XGB_B", "SARIMA_orig")]
for a, b in pairs:
    for kind in ("ape", "se"):
        s, p = P.dm_test(P.weekly_loss(out, a, kind), P.weekly_loss(out, b, kind))
        dm.append({"Model_1": a, "Model_2": b, "loss": kind, "DM_HLN": s, "p_value": p})
dm = pd.DataFrame(dm); dm.to_csv("outputs/table_dm_tests_seed42.csv", index=False)
info["runtime_s"] = time.time() - t0
json.dump(info, open("outputs/run_info_seed42.json", "w"), indent=2, default=str)
pd.set_option("display.width", 250)
print(res[["Model", "RMSE", "RMSE_lo", "RMSE_hi", "MAE", "MAPE", "MAPE_lo", "MAPE_hi", "R2", "R2_lo", "R2_hi"]].round(3).to_string())
print(res[["Model"] + [c for c in res.columns if c.startswith("R2_") and not c.endswith(("lo", "hi"))]].round(3).to_string())
print(dm.round(4).to_string()); print(json.dumps(info, indent=1, default=str)); print(imp.head(12))
