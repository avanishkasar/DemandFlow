"""Repeated simulation experiments: coupling grid (beta) and sensitivity scenarios."""
import sys, json, itertools
import numpy as np, pandas as pd
from multiprocessing import Pool
import sim, pipeline as P

MODELS = ["Naive", "SeasonalNaive", "SARIMA_sel", "SARIMAX", "Ridge_A", "Ridge_B", "XGB_A", "XGB_B",
          "XGB_B_placeboGT", "Ridge_B_placeboGT"]

def job(args):
    scen, seed, kw, stats_on = args
    P.XGB_DEFAULT["n_jobs"] = 1
    sales, gt = sim.simulate(seed=seed, **kw)
    fixed = {}
    if not stats_on:   # sensitivity runs: hyperparameters fixed at the values tuned in the main experiment
        info0 = json.load(open("outputs/run_info_seed42.json"))
        fixed = dict(xgb_params_a={**info0["xgb_params_a"], "n_jobs": 1}, xgb_params_b={**info0["xgb_params_b"], "n_jobs": 1},
                     ridge_alpha=info0["ridge_alpha"])
    out, imp, info, df = P.run_once(sales, gt, with_stats=stats_on, stat_models=("SARIMA_sel", "SARIMAX"), placebo=True, **fixed)
    row = {"scenario": scen, "seed": seed, **{k: v for k, v in kw.items()}}
    for m in [m for m in MODELS if m in out]:
        r = P.metrics(out["y"], out[m])
        for k in r: row[f"{m}_{k}"] = r[k]
    for a, b in [("XGB_B", "XGB_A"), ("Ridge_B", "Ridge_A"), ("XGB_B", "XGB_B_placeboGT"), ("Ridge_B", "Ridge_B_placeboGT")]:
        s, p = P.dm_test(P.weekly_loss(out, a, "ape"), P.weekly_loss(out, b, "ape"))
        row[f"DM_{a}_vs_{b}"] = s; row[f"p_{a}_vs_{b}"] = p
    return row

if __name__ == "__main__":
    which = sys.argv[1]
    jobs = []
    if which == "beta":
        for beta in (0.0, 0.1, 0.2, 0.3, 0.4):
            for seed in range(1, 21):
                jobs.append((f"beta={beta}", seed, {"beta": beta}, True))
    else:
        base = {"beta": 0.2}
        scen = {"sigma=0.05": {"sigma": 0.05}, "sigma=0.20": {"sigma": 0.20},
                "festive=1.20": {"festive_mult": 1.20}, "festive=1.60": {"festive_mult": 1.60},
                "lead=1": {"lag": 1}, "lead=4": {"lag": 4},
                "growth=0": {"growth": 0.0}, "growth=0.30": {"growth": 0.30},
                "amp=0": {"amp": 0.0}, "amp=0.20": {"amp": 0.20},
                "signal=smart watches": {"signal": "smart watches"}}
        for name, kw in scen.items():
            for seed in range(1, 11):
                jobs.append((name, seed, {**base, **kw}, False))
    with Pool(2) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(job, jobs)):
            rows.append(r); print(i, r["scenario"], r["seed"], round(r["XGB_A_MAPE"], 2), round(r["XGB_B_MAPE"], 2),
                                   round(r["Ridge_A_MAPE"], 2), round(r["Ridge_B_MAPE"], 2), flush=True)
            pd.DataFrame(rows).to_csv(f"outputs/replications_{which}.csv", index=False)
