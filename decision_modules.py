"""Launch-window and price modules (revised), plus timing of real search peaks.

Launch window: the 5-week window with the highest total forecast demand in the test year,
compared with (a) the window of highest true simulated demand and (b) the imposed festive-sale weeks.
Counterfactual: all sale events shifted by +6 weeks to check whether the recommendation simply follows the
sale calendar that was put into the simulation.

Price module: linear local demand response around the point forecast Q0 at base price p0 with an ASSUMED
point elasticity e = (dQ/Q0)/(dp/p0):  Q(p) = Q0 * (1 + e * (p - p0) / p0).
Revenue R(p) = p * Q(p) is maximised at p* = p0 * (e - 1) / (2 e), clipped to [0.70 p0, 1.30 p0].
The same e and the same definition are used in the equations and in the code."""
import json
import numpy as np, pandas as pd
import sim, pipeline as P

W = 5


def best_window(s):
    r = s.rolling(W).sum()
    end = r.idxmax()
    i = list(s.index).index(end)
    return s.index[i - W + 1], end


INFO = json.load(open("outputs/run_info_seed42.json"))


def launch_eval(seed, beta=0.2, shift=0):
    """Hyperparameters fixed at the values tuned for the main experiment (seed 42) to save computation."""
    sales, gt = sim.simulate(seed=seed, beta=beta, shift_weeks=shift)
    out, imp, info, df = P.run_once(sales, gt, with_stats=False, xgb_params_a=INFO["xgb_params_a"],
                                    xgb_params_b=INFO["xgb_params_b"], ridge_alpha=INFO["ridge_alpha"])
    tot = out.groupby("Date")[["y", "XGB_B", "Ridge_B"]].sum().sort_index()
    fest = df[df.Festive_Sale == 1].Date.unique()
    fest = [d for d in fest if d >= tot.index.min()]
    row = {"seed": seed, "shift": shift}
    for col in ("y", "XGB_B", "Ridge_B"):
        a, b = best_window(tot[col])
        row[f"{col}_start"] = a.date(); row[f"{col}_end"] = b.date()
        row[f"{col}_start_isoweek"] = int(a.isocalendar().week)
        wk = pd.date_range(a, b, freq="W-SUN")
        row[f"{col}_overlap_festive_weeks"] = int(sum(w in set(fest) for w in wk))
    tw = pd.date_range(row["y_start"], row["y_end"], freq="W-SUN")
    for col in ("XGB_B", "Ridge_B"):
        fw = pd.date_range(row[f"{col}_start"], row[f"{col}_end"], freq="W-SUN")
        row[f"{col}_overlap_true_weeks"] = len(set(tw) & set(fw))
    row["festive_start"] = min(fest).date(); row["festive_end"] = max(fest).date()
    return row, out


def price_table(q0, p0, elasticities=(-0.5, -0.8, -1.0, -1.5, -2.0, -3.0)):
    rows = []
    grid = np.linspace(0.70 * p0, 1.30 * p0, 40)
    for e in elasticities:
        q = np.maximum(q0 * (1 + e * (grid - p0) / p0), 0)
        rev = grid * q
        p_star_closed = np.clip(p0 * (e - 1) / (2 * e), 0.70 * p0, 1.30 * p0)
        rows.append({"elasticity": e, "p_star_grid": grid[rev.argmax()], "p_star_closed_form": p_star_closed,
                     "p_star_over_p0": grid[rev.argmax()] / p0, "revenue_at_p_star": rev.max(),
                     "revenue_at_p0": p0 * q0, "revenue_gain_pct": (rev.max() / (p0 * q0) - 1) * 100})
    return pd.DataFrame(rows), grid


if __name__ == "__main__":
    rows = []
    for seed in range(1, 11):
        r, out = launch_eval(seed); rows.append(r)
        r2, _ = launch_eval(seed, shift=6); rows.append(r2)
        print(seed, r["XGB_B_start"], r["y_start"], r2["XGB_B_start"], r2["y_start"], flush=True)
    L = pd.DataFrame(rows); L.to_csv("outputs/launch_window_eval.csv", index=False)
    summ = L.groupby("shift").agg(
        xgb_overlap_true=("XGB_B_overlap_true_weeks", "mean"), ridge_overlap_true=("Ridge_B_overlap_true_weeks", "mean"),
        xgb_overlap_fest=("XGB_B_overlap_festive_weeks", "mean"), true_overlap_fest=("y_overlap_festive_weeks", "mean"),
        xgb_start_week_median=("XGB_B_start_isoweek", "median"), true_start_week_median=("y_start_isoweek", "median"))
    summ.to_csv("outputs/launch_window_summary.csv"); print(summ)

    # price module on the main experiment (seed 42): one festive and one non-festive test week, HDMI cables
    pred = pd.read_csv("outputs/test_predictions_seed42.csv", parse_dates=["Date"])
    feats = pd.read_csv("outputs/model_input_features_seed42.csv", parse_dates=["Date"])
    h = pred[pred.Category == "HDMICables"].merge(feats[["Date", "Category", "Festive_Sale"]], on=["Date", "Category"])
    base = sim.load_base_demand()["HDMICables"]["avg_price"]
    fw = h[h.Festive_Sale == 1].iloc[0]; nw = h[h.Festive_Sale == 0].iloc[10]
    allp = []
    for label, w in (("festive", fw), ("non-festive", nw)):
        t, _ = price_table(w["XGB_B"], base)
        t.insert(0, "week", str(w["Date"].date())); t.insert(0, "week_type", label); t.insert(2, "forecast_units", w["XGB_B"])
        t.insert(3, "base_price", base); allp.append(t)
    pd.concat(allp).to_csv("outputs/table_price_sensitivity.csv", index=False)
    print(pd.concat(allp).round(2).to_string())

    # timing of real search peaks (independent real-world evidence)
    g = sim.load_trends("earbuds india")
    pk = []
    for yr in (2023, 2024, 2025):
        s = g[(g.index >= f"{yr}-07-01") & (g.index < f"{yr + 1}-03-01")]
        a, b = best_window(s)
        pk.append({"season": yr, "peak5_start": a.date(), "peak5_end": b.date(), "start_isoweek": int(a.isocalendar().week),
                   "peak_week": s.idxmax().date(), "peak_value": s.max(), "diwali": sim.DIWALI[yr]})
    pd.DataFrame(pk).to_csv("outputs/real_search_peak_timing.csv", index=False); print(pd.DataFrame(pk))
