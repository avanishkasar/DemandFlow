"""Generates every revised figure from the CSV outputs (run after run_main.py, run_replications.py,
decision_modules.py and leakage_audit.py)."""
import sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import sim

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"})
OUT = "figures/"
NAMES = {"Naive": "Naïve", "SeasonalNaive": "Seasonal naïve", "SARIMA_orig": "SARIMA(1,1,1)(1,1,1)$_{52}$",
         "SARIMA_sel": "SARIMA(0,0,0)(0,1,0)$_{52}$+drift", "SARIMAX": "SARIMAX (sales + search)",
         "Ridge_A": "Ridge (no search)", "Ridge_B": "Ridge + search", "XGB_A": "XGBoost (no search)",
         "XGB_B": "XGBoost + search"}
C_SEARCH, C_NOSEARCH, C_STAT = "#1b7837", "#7f7f7f", "#2166ac"


def fig_architecture():
    fig, ax = plt.subplots(figsize=(10, 6.4)); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    layers = [
        (8.3, "1. INPUT LAYER", "#dbe9f6", ["Amazon India catalogue\n(3 categories, 275 products)", "Real Google Trends index\n'earbuds india', geo=IN, weekly",
                                             "Public sale calendar\n(Diwali, Republic Day, Prime Day)"]),
        (6.0, "2. DATA LAYER", "#e3f1df", ["Controlled demand simulator\n(growth, seasonality, promotions,\nsearch coupling β, noise σ)", "Weekly alignment\n(Sunday-start weeks)",
                                            "Leakage checks\n(only data ≤ t−1 for week t)"]),
        (3.7, "3. FEATURE LAYER", "#fdf0d5", ["Calendar + Fourier terms\n+ sale-event flags", "Past demand: lags 1–4,\nmeans of past weeks",
                                               "Search: lags 1–4,\nmomentum, acceleration"]),
        (1.4, "4. MODEL & EVALUATION LAYER", "#ece2f0", ["Benchmarks: naïve, seasonal naïve,\nSARIMA, SARIMAX", "Ridge / XGBoost\nwith and without search",
                                                         "Rolling origin, block-bootstrap CIs,\nDM tests, ablation, sensitivity"]),
    ]
    for y, title, col, boxes in layers:
        ax.add_patch(FancyBboxPatch((0.2, y - 0.95), 9.6, 1.95, boxstyle="round,pad=0.02", fc=col, ec="#555", lw=1))
        ax.text(5, y + 0.78, title, ha="center", va="center", fontsize=11, fontweight="bold")
        for i, b in enumerate(boxes):
            x = 0.5 + i * 3.1
            ax.add_patch(FancyBboxPatch((x, y - 0.8), 2.85, 1.25, boxstyle="round,pad=0.02", fc="white", ec="#777", lw=0.8))
            ax.text(x + 1.425, y - 0.17, b, ha="center", va="center", fontsize=8.3)
        if y > 2:
            ax.annotate("", xy=(5, y - 1.0), xytext=(5, y - 0.93), arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.4))
    ax.text(5, 0.1, "Outputs: one-week-ahead forecasts → launch-window module and assumption-based price module (prototype web app)",
            ha="center", fontsize=8.5, style="italic")
    fig.savefig(OUT + "fig02_architecture.png"); plt.close(fig)


def fig_gt():
    for kw, fn, col in (("earbuds india", "fig03_gt_earbuds.png", "#2166ac"), ("smart watches", "fig04_gt_smartwatches.png", "#1b9e77")):
        g = sim.load_trends(kw)
        _, fest, rday, prime = sim.sale_calendar(g.index)
        fig, ax = plt.subplots(figsize=(10, 3.4))
        ax.fill_between(g.index, g.values, color=col, alpha=0.2); ax.plot(g.index, g.values, color=col, lw=1.4)
        for dd in g.index[fest == 1]:
            ax.axvspan(dd, dd + pd.Timedelta(days=7), color="#f4a582", alpha=0.45, lw=0)
        for dd in g.index[(rday == 1) | (prime == 1)]:
            ax.axvspan(dd, dd + pd.Timedelta(days=7), color="#b2abd2", alpha=0.6, lw=0)
        for yr in (2023, 2024, 2025):
            s = g[(g.index >= f"{yr}-07-01") & (g.index < f"{yr + 1}-03-01")]
            ax.annotate(f"max {s.max():.0f}\n{s.idxmax():%d %b %Y}", (s.idxmax(), s.max()), xytext=(0, -38), textcoords="offset points",
                        ha="center", fontsize=7, arrowprops=dict(arrowstyle="-", lw=0.6))
        ax.set_ylabel("Search index (0–100)"); ax.set_ylim(0, 108)
        from matplotlib.patches import Patch
        ax.legend([Patch(color="#f4a582", alpha=0.6), Patch(color="#b2abd2", alpha=0.7)],
                  ["Festive sale (Diwali and 3 weeks before)", "Republic Day / Prime Day sale week"], fontsize=7.5, loc="upper left", ncol=2, frameon=False)
        ax.set_title(f"Google Trends, '{kw}', India, weekly (real data)", fontsize=9.5)
        fig.savefig(OUT + fn); plt.close(fig)


def fig_series():
    s = pd.read_csv("outputs/simulated_weekly_demand_seed42.csv", parse_dates=["Date"])
    gt = sim.load_trends("earbuds india")
    h = s[s.Category == "HDMICables"].set_index("Date")
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(h.index, h.Units_Sold, color="#2166ac", lw=1.4, label="Simulated weekly demand, HDMI cables (seed 42, β = 0.2)")
    for d in h.index[h.Festive_Sale == 1]:
        ax.axvspan(d, d + pd.Timedelta(days=7), color="#f4a582", alpha=0.35, lw=0)
    for d in h.index[(h.RepublicDay_Sale == 1) | (h.PrimeDay_Sale == 1)]:
        ax.axvspan(d, d + pd.Timedelta(days=7), color="#b2abd2", alpha=0.5, lw=0)
    split = pd.read_csv("outputs/test_predictions_seed42.csv", parse_dates=["Date"]).Date.min()
    ax.axvline(split, color="k", ls="--", lw=1); ax.text(split, ax.get_ylim()[0] * 1.02, "  test year →", fontsize=8, va="bottom")
    ax.set_ylabel("Units per week")
    ax2 = ax.twinx(); ax2.plot(gt.index, gt.values, color="#b2182b", lw=1, alpha=0.6, label="Google Trends 'earbuds india' (real)")
    ax2.set_ylabel("Search index (0–100)"); ax2.spines["right"].set_visible(True)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    from matplotlib.patches import Patch
    ax.legend(h1 + h2 + [Patch(color="#f4a582", alpha=0.5), Patch(color="#b2abd2", alpha=0.7)],
              l1 + l2 + ["Festive sale (Diwali and 3 weeks before)", "Republic Day / Prime Day sale week"], fontsize=7.5,
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    fig.savefig(OUT + "fig03_simulated_series.png"); plt.close(fig)


def fig_corr():
    f = pd.read_csv("outputs/model_input_features_seed42.csv", parse_dates=["Date"])
    f = f[f.Date < pd.read_csv("outputs/test_predictions_seed42.csv", parse_dates=["Date"]).Date.min()]
    cols = ["target_ratio", "Festive_Sale", "RepublicDay_Sale", "PrimeDay_Sale", "sin1", "cos1", "rel_lag1", "rel_MA2",
            "GT_lag1", "GT_lag2", "GT_lag3", "GT_lag4", "GT_momentum", "GT_acceleration"]
    c = f[cols].corr()
    lab = ["Demand ratio (target)"] + cols[1:]
    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(lab, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(lab, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{c.values[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(c.values[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r (training weeks only)")
    fig.savefig(OUT + "fig06_correlation_heatmap.png"); plt.close(fig)


def fig_metrics():
    t = pd.read_csv("outputs/table_main_metrics_ci.csv")
    order = ["SeasonalNaive", "SARIMA_orig", "SARIMA_sel", "SARIMAX", "Ridge_A", "Ridge_B", "XGB_A", "XGB_B"]
    t = t.set_index("Model").loc[order]
    cols = [C_NOSEARCH, C_STAT, C_STAT, C_SEARCH, C_NOSEARCH, C_SEARCH, C_NOSEARCH, C_SEARCH]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, m, lab in zip(axs, ["RMSE", "MAE", "MAPE"], ["RMSE (units)", "MAE (units)", "MAPE (%)"]):
        v = t[m].values; err = [v - t[m + "_lo"].values, t[m + "_hi"].values - v]
        ax.bar(range(len(order)), v, yerr=err, color=cols, capsize=3, edgecolor="k", lw=0.5)
        ax.set_xticks(range(len(order))); ax.set_xticklabels([NAMES[o] for o in order], rotation=55, ha="right", fontsize=7.5)
        ax.set_title(lab)
        for i, x in enumerate(v):
            ax.text(i, x * 0.04, f"{x:.1f}", ha="center", fontsize=7, color="white", fontweight="bold")
    from matplotlib.patches import Patch
    axs[0].legend([Patch(color=C_NOSEARCH), Patch(color=C_STAT), Patch(color=C_SEARCH)],
                  ["Uses past sales/calendar only", "Univariate SARIMA", "Uses Google Trends"], fontsize=8,
                  loc="lower center", bbox_to_anchor=(1.7, -0.75), ncol=3, frameon=False)
    fig.suptitle("Test year, one-week-ahead rolling-origin forecasts (seed 42, β = 0.2); bars = 95% moving-block bootstrap CI", fontsize=9.5)
    fig.savefig(OUT + "fig09_metrics_ci.png"); plt.close(fig)


def fig_sarima_and_models():
    p = pd.read_csv("outputs/test_predictions_seed42.csv", parse_dates=["Date"])
    s = pd.read_csv("outputs/simulated_weekly_demand_seed42.csv", parse_dates=["Date"])
    h = p[p.Category == "HDMICables"].sort_values("Date"); hs = s[s.Category == "HDMICables"]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(hs.Date, hs.Units_Sold, color="#555", lw=1, label="Simulated demand")
    ax.plot(h.Date, h.SARIMA_sel, color=C_STAT, lw=1.4, ls="--", label="SARIMA(0,0,0)(0,1,0)$_{52}$+drift, one-week-ahead")
    ax.fill_between(h.Date, h.SARIMA_sel_lo, h.SARIMA_sel_hi, color=C_STAT, alpha=0.15, label="95% model-based prediction interval")
    ax.plot(h.Date, h.SARIMA_orig, color="#d6604d", lw=1, ls=":", label="SARIMA(1,1,1)(1,1,1)$_{52}$ (original specification)")
    ax.axvline(h.Date.min(), color="k", ls="--", lw=0.8); ax.set_ylabel("Units per week"); ax.legend(fontsize=7.5, loc="upper left")
    fig.savefig(OUT + "fig10_sarima.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(h.Date, h.y, "k-o", ms=3, lw=1.3, label="Simulated demand")
    for m, c, ls in (("SARIMAX", C_STAT, "--"), ("XGB_A", C_NOSEARCH, "-."), ("XGB_B", C_SEARCH, "-"), ("Ridge_B", "#e08214", ":")):
        ax.plot(h.Date, h[m], color=c, ls=ls, lw=1.3, label=NAMES[m])
    ax.set_ylabel("Units per week"); ax.legend(fontsize=7.5, ncol=3, loc="upper left")
    fig.savefig(OUT + "fig11_model_comparison.png"); plt.close(fig)


def fig_importance():
    from sklearn.inspection import permutation_importance
    imp = pd.read_csv("outputs/feature_importance_gain_seed42.csv", index_col=0)["gain"]
    imp = (imp / imp.sum()).sort_values().tail(16)
    fig, ax = plt.subplots(figsize=(7, 5.2))
    ax.barh(range(len(imp)), imp.values, color=[C_SEARCH if i.startswith("GT") else C_STAT for i in imp.index], edgecolor="k", lw=0.4)
    ax.set_yticks(range(len(imp))); ax.set_yticklabels(imp.index, fontsize=8)
    ax.set_xlabel("Share of total gain (XGBoost + search, averaged over 13 refits)")
    from matplotlib.patches import Patch
    ax.legend([Patch(color=C_SEARCH), Patch(color=C_STAT)], ["Google Trends features", "Calendar and past-demand features"], fontsize=8, loc="lower right")
    fig.savefig(OUT + "fig12_feature_importance.png"); plt.close(fig)


def fig_beta():
    r = pd.read_csv("outputs/replications_beta.csv")
    r["beta"] = r["beta"].astype(float)
    g = r.groupby("beta")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for m, c, ls in (("SeasonalNaive", "#999", ":"), ("SARIMAX", C_STAT, "--"), ("Ridge_A", "#e08214", "-."), ("Ridge_B", "#e08214", "-"),
                     ("XGB_A", C_NOSEARCH, "-."), ("XGB_B", C_SEARCH, "-")):
        mu = g[f"{m}_MAPE"].mean(); lo = g[f"{m}_MAPE"].quantile(0.025); hi = g[f"{m}_MAPE"].quantile(0.975)
        axs[0].plot(mu.index, mu.values, color=c, ls=ls, marker="o", ms=3, label=NAMES[m])
        if m in ("XGB_B", "Ridge_B"):
            axs[0].fill_between(mu.index, lo, hi, color=c, alpha=0.12)
    axs[0].set_xlabel("Search–demand coupling strength β"); axs[0].set_ylabel("Test-year MAPE (%), mean of 20 replications")
    axs[0].legend(fontsize=7.5)
    for a, b, c in (("XGB_B", "XGB_A", C_SEARCH), ("Ridge_B", "Ridge_A", "#e08214")):
        d = r[f"{a}_MAPE"] - r[f"{b}_MAPE"]
        dd = d.groupby(r["beta"])
        axs[1].errorbar(dd.mean().index + (0.006 if a == "Ridge_B" else -0.006), dd.mean().values,
                        yerr=1.96 * dd.std().values / np.sqrt(dd.count().values), color=c, marker="o", capsize=3,
                        label=f"{NAMES[a]} minus {NAMES[b]}")
    axs[1].axhline(0, color="k", lw=0.8)
    axs[1].set_xlabel("Search–demand coupling strength β"); axs[1].set_ylabel("Δ MAPE (percentage points), mean ± 95% CI")
    axs[1].legend(fontsize=7.5)
    fig.savefig(OUT + "fig13_coupling_grid.png"); plt.close(fig)


def fig_sens():
    r = pd.read_csv("outputs/replications_sens.csv")
    order = list(dict.fromkeys(r.scenario))
    rows = []
    for s in order:
        x = r[r.scenario == s]
        for a, b in (("XGB_B", "XGB_A"), ("Ridge_B", "Ridge_A")):
            d = x[f"{a}_MAPE"] - x[f"{b}_MAPE"]
            rows.append((s, a, d.mean(), 1.96 * d.std() / np.sqrt(len(d))))
    d = pd.DataFrame(rows, columns=["s", "m", "mu", "ci"])
    fig, ax = plt.subplots(figsize=(8, 5))
    for k, (m, c) in enumerate((("XGB_B", C_SEARCH), ("Ridge_B", "#e08214"))):
        q = d[d.m == m].reset_index(drop=True)
        ax.errorbar(q.mu, np.arange(len(q)) + (k - 0.5) * 0.25, xerr=q.ci, fmt="o", color=c, capsize=3,
                    label=f"{NAMES[m]} minus {'XGBoost' if m == 'XGB_B' else 'Ridge'} without search")
    lab = {"sigma=0.05": "Noise σ = 0.05", "sigma=0.20": "Noise σ = 0.20", "festive=1.20": "Festive uplift ×1.20", "festive=1.60": "Festive uplift ×1.60",
           "lead=1": "Search lead 1 week", "lead=4": "Search lead 4 weeks", "growth=0": "Growth 0%/year", "growth=0.30": "Growth 30%/year",
           "amp=0": "Seasonal amplitude 0", "amp=0.20": "Seasonal amplitude 0.20", "signal=smart watches": "Signal: 'smart watches'"}
    ax.set_yticks(range(len(order))); ax.set_yticklabels([lab.get(o, o) for o in order], fontsize=8)
    ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("Δ MAPE (percentage points), mean ± 95% CI over 10 replications")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.45, -0.12), ncol=2, frameon=False); ax.invert_yaxis()
    fig.savefig(OUT + "fig14_sensitivity.png"); plt.close(fig)


def fig_launch_price():
    p = pd.read_csv("outputs/test_predictions_seed42.csv", parse_dates=["Date"])
    f = pd.read_csv("outputs/model_input_features_seed42.csv", parse_dates=["Date"])
    tot = p.groupby("Date")[["y", "XGB_B"]].sum()
    fest = f[(f.Festive_Sale == 1) & (f.Date >= tot.index.min())].Date.unique()
    r = tot["XGB_B"].rolling(5).sum(); end = r.idxmax(); i = list(tot.index).index(end); start = tot.index[i - 4]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(tot.index, tot.y, "k-", lw=1.2, label="Simulated demand (all three categories)")
    ax.plot(tot.index, tot.XGB_B, color=C_SEARCH, ls="--", lw=1.3, label="XGBoost + search, one-week-ahead")
    for d in fest:
        ax.axvspan(d, d + pd.Timedelta(days=7), color="#f4a582", alpha=0.35, lw=0)
    ax.axvspan(start, end + pd.Timedelta(days=7), fill=False, ec="#b8860b", lw=2, ls="-", label="Recommended 5-week window")
    ax.set_ylabel("Units per week"); ax.legend(fontsize=7.5, loc="upper left")
    fig.savefig(OUT + "fig15_launch_window.png"); plt.close(fig)
    pt = pd.read_csv("outputs/table_price_sensitivity.csv")
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=False)
    for ax, wt in zip(axs, ("festive", "non-festive")):
        q = pt[pt.week_type == wt]; q0 = q.forecast_units.iloc[0]; p0 = q.base_price.iloc[0]
        grid = np.linspace(0.7 * p0, 1.3 * p0, 40)
        for e, c in zip((-0.5, -1.0, -1.5, -2.0), ("#1b7837", "#7fbf7b", "#e08214", "#b2182b")):
            rev = grid * np.maximum(q0 * (1 + e * (grid - p0) / p0), 0) / 1e5
            ax.plot(grid, rev, color=c, label=f"ε = {e}")
            ax.plot(grid[rev.argmax()], rev.max(), "*", color=c, ms=10)
        ax.axvline(p0, color="k", ls=":", lw=0.8)
        ax.set_title(f"{wt.capitalize()} week {q.week.iloc[0]} (forecast {q0:.0f} units)", fontsize=9)
        ax.set_xlabel("Price (INR)"); ax.set_ylabel("Projected revenue (INR lakh)")
    axs[0].legend(fontsize=7.5)
    fig.savefig(OUT + "fig08_price_sensitivity.png"); plt.close(fig)


if __name__ == "__main__":
    import os; os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1:] or ["gt", "arch", "series", "corr", "metrics", "sarima", "imp", "beta", "sens", "launch"]
    fns = {"gt": fig_gt, "arch": fig_architecture, "series": fig_series, "corr": fig_corr, "metrics": fig_metrics, "sarima": fig_sarima_and_models,
           "imp": fig_importance, "beta": fig_beta, "sens": fig_sens, "launch": fig_launch_price}
    for w in which:
        fns[w](); print("done", w)
