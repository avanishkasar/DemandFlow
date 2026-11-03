"""
Leakage-free feature engineering, models and rolling-origin evaluation for the revised study.

Every feature for target week t uses only information available at the end of week t-1,
except calendar variables and the planned sale calendar, which are known in advance.
"""
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

TEST_WEEKS = 52
REFIT_EVERY = 4          # ML models are re-estimated every 4 weeks inside the test year
STAT_REFIT_EVERY = 13    # SARIMA-type models are re-estimated every 13 weeks, filtered weekly in between

CAL = ["Week_of_Year", "Month", "Quarter", "sin1", "cos1", "sin2", "cos2", "Festive_Sale", "RepublicDay_Sale", "PrimeDay_Sale", "Category_Encoded"]
DEM = ["rel_lag1", "rel_lag2", "rel_lag3", "rel_lag4", "rel_MA2", "log_MA4"]
GT_LAGS = ["GT_lag1", "GT_lag2", "GT_lag3", "GT_lag4"]
GT_DER = ["GT_momentum", "GT_acceleration"]
FEATS_A = CAL + DEM
FEATS_B = CAL + DEM + GT_LAGS + GT_DER

XGB_DEFAULT = dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                   colsample_bytree=0.8, min_child_weight=1, random_state=42, n_jobs=2, verbosity=0)


# ----------------------------------------------------------------------------- features
def build_features(sales, gt, gt_contemporaneous=False):
    df = sales.copy().sort_values(["Category", "Date"]).reset_index(drop=True)
    df["Week_of_Year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    for k in (1, 2):   # Fourier terms of the week of year (smooth annual seasonality, known in advance)
        df[f"sin{k}"] = np.sin(2 * np.pi * k * df["Week_of_Year"] / 52.0)
        df[f"cos{k}"] = np.cos(2 * np.pi * k * df["Week_of_Year"] / 52.0)
    df["Category_Encoded"] = df["Category"].astype("category").cat.codes
    g = df.groupby("Category")["Units_Sold"]
    for k in range(1, 5):
        df[f"lag{k}"] = g.shift(k)
    df["lag52"] = g.shift(52)
    df["MA2"] = df[["lag1", "lag2"]].mean(axis=1)            # rolling means of PAST weeks only
    df["MA4"] = df[["lag1", "lag2", "lag3", "lag4"]].mean(axis=1)
    for k in range(1, 5):
        df[f"rel_lag{k}"] = df[f"lag{k}"] / df["MA4"]
    df["rel_MA2"] = df["MA2"] / df["MA4"]
    df["log_MA4"] = np.log(df["MA4"])
    gts = gt.copy()
    for k in range(1, 5):
        df[f"GT_lag{k}"] = df["Date"].map(gts.shift(k))
    df["GT_now"] = df["Date"].map(gts)
    df["GT_momentum"] = df["GT_lag1"] - df["GT_lag3"]                                   # 2-week change
    df["GT_acceleration"] = (df["GT_lag1"] - df["GT_lag2"]) - (df["GT_lag2"] - df["GT_lag3"])  # change of change
    df["target_ratio"] = df["Units_Sold"] / df["MA4"]
    df = df.dropna(subset=["lag4", "GT_lag4"]).reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------- metrics
def metrics(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    e = y - p
    return {"RMSE": np.sqrt(np.mean(e ** 2)), "MAE": np.mean(np.abs(e)),
            "MAPE": np.mean(np.abs(e) / y) * 100, "R2": 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)}


def block_bootstrap_ci(pred_df, model, n_boot=2000, block=4, seed=0):
    """Moving-block bootstrap over test weeks (all categories of a week kept together)."""
    weeks = np.sort(pred_df["Date"].unique())
    n = len(weeks)
    rng = np.random.default_rng(seed)
    by_week = {w: pred_df[pred_df["Date"] == w] for w in weeks}
    out = []
    nb = int(np.ceil(n / block))
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        d = pd.concat([by_week[weeks[i]] for i in idx])
        out.append(metrics(d["y"], d[model]))
    o = pd.DataFrame(out)
    return o.quantile(0.025), o.quantile(0.975)


def dm_test(e1, e2, h=1, loss="abs"):
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample correction.
    e1, e2: weekly loss series (already averaged over categories). Negative statistic => model 1 better."""
    d = np.asarray(e1) - np.asarray(e2)
    n = len(d)
    dbar = d.mean()
    gamma = [np.sum((d[k:] - dbar) * (d[:n - k] - dbar)) / n for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    dm = dbar / np.sqrt(var)
    hln = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(abs(hln), df=n - 1)
    return hln, p


def weekly_loss(pred_df, model, kind="ape"):
    d = pred_df.copy()
    if kind == "ape":
        d["l"] = np.abs(d["y"] - d[model]) / d["y"] * 100
    else:  # squared error scaled by category mean to keep categories comparable
        scale = d.groupby("Category")["y"].transform("mean")
        d["l"] = ((d["y"] - d[model]) / scale) ** 2
    return d.groupby("Date")["l"].mean().sort_index().values


# ----------------------------------------------------------------------------- GT lag selection
def select_gt_lag(train):
    """Pick the search lag (1-4 weeks) whose value is most correlated with the demand ratio
    after removing the planned-sale effect, using the training window only."""
    t = train.copy()
    resid = t["target_ratio"] - t.groupby(["Festive_Sale", "RepublicDay_Sale", "PrimeDay_Sale"])["target_ratio"].transform("mean")
    cors = {k: np.corrcoef(t[f"GT_lag{k}"], resid)[0, 1] for k in range(1, 5)}
    return max(cors, key=lambda k: abs(cors[k]))


# ----------------------------------------------------------------------------- ML models
def fit_predict_xgb(train, test, feats, params, ratio_target=True):
    m = XGBRegressor(**params)
    if ratio_target:
        m.fit(train[feats], train["target_ratio"])
        return m.predict(test[feats]) * test["MA4"].values, m
    m.fit(train[feats], train["Units_Sold"])
    return m.predict(test[feats]), m


def fit_predict_ridge(train, test, feats, alpha):
    m = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    m.fit(train[feats], train["target_ratio"])
    return m.predict(test[feats]) * test["MA4"].values


def tune_xgb(train, feats, n_folds=3, fold_len=13):
    grid = [dict(max_depth=d, n_estimators=n, learning_rate=lr, min_child_weight=mcw)
            for d in (2, 3, 4, 6) for n in (100, 300) for lr in (0.05, 0.1) for mcw in (1, 5)]
    dates = np.sort(train["Date"].unique())
    best, best_score = None, np.inf
    for g in grid:
        params = {**XGB_DEFAULT, **g}
        errs = []
        for f in range(n_folds, 0, -1):
            cut = dates[-f * fold_len]
            end = dates[-(f - 1) * fold_len] if f > 1 else None
            tr = train[train["Date"] < cut]
            va = train[(train["Date"] >= cut) & ((train["Date"] < end) if end is not None else True)]
            p, _ = fit_predict_xgb(tr, va, feats, params)
            errs.append(metrics(va["Units_Sold"], p)["MAPE"])
        s = np.mean(errs)
        if s < best_score:
            best, best_score = params, s
    return best, best_score


def tune_ridge(train, feats, n_folds=3, fold_len=13):
    dates = np.sort(train["Date"].unique())
    best, best_score = None, np.inf
    for a in (0.01, 0.1, 1, 10, 100):
        errs = []
        for f in range(n_folds, 0, -1):
            cut = dates[-f * fold_len]
            end = dates[-(f - 1) * fold_len] if f > 1 else None
            tr = train[train["Date"] < cut]
            va = train[(train["Date"] >= cut) & ((train["Date"] < end) if end is not None else True)]
            errs.append(metrics(va["Units_Sold"], fit_predict_ridge(tr, va, feats, a))["MAPE"])
        if np.mean(errs) < best_score:
            best, best_score = a, np.mean(errs)
    return best


# ----------------------------------------------------------------------------- statistical models
def sarima_rolling(series, exog, test_dates, order, sorder, refit_every=STAT_REFIT_EVERY, trend=None):
    """One-step-ahead rolling forecasts. Parameters re-estimated every `refit_every` weeks;
    between re-estimations the state is updated with new observations (no parameter change)."""
    preds, lo, hi = [], [], []
    test_dates = list(test_dates)
    for b in range(0, len(test_dates), refit_every):
        origin = test_dates[b]
        block = test_dates[b:b + refit_every]
        ytr = series[series.index < origin]
        xtr = exog.loc[ytr.index] if exog is not None else None
        mod = SARIMAX(ytr, exog=xtr, order=order, seasonal_order=sorder, trend=trend,
                      enforce_stationarity=False, enforce_invertibility=False)
        res = mod.fit(disp=False, maxiter=200)
        for d in block:
            x1 = exog.loc[[d]] if exog is not None else None
            f = res.get_forecast(1, exog=x1)
            preds.append(float(f.predicted_mean.iloc[0]))
            ci = f.conf_int(alpha=0.05)
            lo.append(float(ci.iloc[0, 0])); hi.append(float(ci.iloc[0, 1]))
            res = res.append(series.loc[[d]], exog=x1, refit=False)
    return np.array(preds), np.array(lo), np.array(hi)


# ----------------------------------------------------------------------------- one full run
def run_once(sales, gt, xgb_params_a=None, xgb_params_b=None, ridge_alpha=None,
             with_stats=True, stat_models=("SARIMA_orig", "SARIMA_sel", "SARIMAX"),
             extra_variants=False, placebo=False):
    df = build_features(sales, gt)
    dates = np.sort(df["Date"].unique())
    test_dates = dates[-TEST_WEEKS:]
    first_test = test_dates[0]
    train0 = df[df["Date"] < first_test]
    if xgb_params_a is None:
        xgb_params_a, _ = tune_xgb(train0, FEATS_A)
    if xgb_params_b is None:
        xgb_params_b, _ = tune_xgb(train0, FEATS_B)
    if ridge_alpha is None:
        ridge_alpha = {"A": tune_ridge(train0, FEATS_A), "B": tune_ridge(train0, FEATS_B)}

    test = df[df["Date"].isin(test_dates)].copy()
    out = test[["Date", "Category", "Units_Sold"]].rename(columns={"Units_Sold": "y"}).copy()
    out["Naive"] = test["lag1"].values
    out["SeasonalNaive"] = test["lag52"].values
    cols = {"XGB_A": [], "XGB_B": [], "Ridge_A": [], "Ridge_B": []}
    if extra_variants:
        for k in ("XGB_B_lagsonly", "XGB_B_now", "XGB_A_raw", "XGB_B_raw", "XGB_B_orig", "XGB_B_noDemandLags"):
            cols[k] = []
    imps = []
    for b in range(0, len(test_dates), REFIT_EVERY):
        origin = test_dates[b]
        block = test_dates[b:b + REFIT_EVERY]
        tr = df[df["Date"] < origin]
        te = df[df["Date"].isin(block)]
        pa, _ = fit_predict_xgb(tr, te, FEATS_A, xgb_params_a)
        pb, mb = fit_predict_xgb(tr, te, FEATS_B, xgb_params_b)
        cols["XGB_A"].append(pd.Series(pa, index=te.index))
        cols["XGB_B"].append(pd.Series(pb, index=te.index))
        cols["Ridge_A"].append(pd.Series(fit_predict_ridge(tr, te, FEATS_A, ridge_alpha["A"]), index=te.index))
        cols["Ridge_B"].append(pd.Series(fit_predict_ridge(tr, te, FEATS_B, ridge_alpha["B"]), index=te.index))
        imps.append(pd.Series(mb.get_booster().get_score(importance_type="gain")))
        if extra_variants:
            f_l = FEATS_A + GT_LAGS
            cols["XGB_B_lagsonly"].append(pd.Series(fit_predict_xgb(tr, te, f_l, xgb_params_b)[0], index=te.index))
            f_n = FEATS_B + ["GT_now"]
            cols["XGB_B_now"].append(pd.Series(fit_predict_xgb(tr, te, f_n, xgb_params_b)[0], index=te.index))
            cols["XGB_A_raw"].append(pd.Series(fit_predict_xgb(tr, te, FEATS_A, xgb_params_a, False)[0], index=te.index))
            cols["XGB_B_raw"].append(pd.Series(fit_predict_xgb(tr, te, FEATS_B, xgb_params_b, False)[0], index=te.index))
            orig = dict(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                        random_state=42, n_jobs=2, verbosity=0)
            cols["XGB_B_orig"].append(pd.Series(fit_predict_xgb(tr, te, FEATS_B, orig)[0], index=te.index))
            f_nd = [f for f in FEATS_B if f not in DEM or f == "log_MA4"]
            cols["XGB_B_noDemandLags"].append(pd.Series(fit_predict_xgb(tr, te, f_nd, xgb_params_b)[0], index=te.index))
    for k, v in cols.items():
        out[k] = pd.concat(v).loc[test.index].values

    if placebo or extra_variants:
        # placebo: the same search series shifted by 52 weeks (same seasonal shape, but the
        # year-specific information is taken from a different year)
        gt_pl = pd.Series(np.roll(gt.values, 52), index=gt.index)
        dfp = build_features(sales, gt_pl)
        pp = []
        for b in range(0, len(test_dates), REFIT_EVERY):
            origin = test_dates[b]; block = test_dates[b:b + REFIT_EVERY]
            tr = dfp[dfp["Date"] < origin]; te = dfp[dfp["Date"].isin(block)]
            pp.append(pd.DataFrame({"Date": te["Date"], "Category": te["Category"],
                                    "XGB_B_placeboGT": fit_predict_xgb(tr, te, FEATS_B, xgb_params_b)[0],
                                    "Ridge_B_placeboGT": fit_predict_ridge(tr, te, FEATS_B, ridge_alpha["B"])}))
        out = out.merge(pd.concat(pp), on=["Date", "Category"])

    if with_stats:
        spec = {"SARIMA_orig": ((1, 1, 1), (1, 1, 1, 52), None), "SARIMA_sel": ((0, 0, 0), (0, 1, 0, 52), "c")}
        for c in out["Category"].unique():
            s = df[df["Category"] == c].set_index("Date")["Units_Sold"].astype(float)
            s.index.freq = "W-SUN"
            ex = df[df["Category"] == c].set_index("Date")[["Festive_Sale", "RepublicDay_Sale", "PrimeDay_Sale",
                                                             "GT_lag1", "GT_lag2"]].astype(float)
            m = (out["Category"] == c).values
            if "SARIMA_orig" in stat_models:
                o, so, tr_ = spec["SARIMA_orig"]
                out.loc[m, "SARIMA_orig"] = sarima_rolling(s, None, test_dates, o, so, trend=tr_)[0]
            o, so, tr_ = spec["SARIMA_sel"]
            if "SARIMA_sel" in stat_models:
                p, lo, hi = sarima_rolling(s, None, test_dates, o, so, trend=tr_)
                out.loc[m, "SARIMA_sel"] = p; out.loc[m, "SARIMA_sel_lo"] = lo; out.loc[m, "SARIMA_sel_hi"] = hi
            if "SARIMAX" in stat_models:
                out.loc[m, "SARIMAX"] = sarima_rolling(s, ex, test_dates, o, so, trend=tr_)[0]

    imp = pd.concat(imps, axis=1).fillna(0).mean(axis=1).sort_values(ascending=False)
    info = {"xgb_params_a": xgb_params_a, "xgb_params_b": xgb_params_b, "ridge_alpha": ridge_alpha,
            "n_rows": len(df), "n_train0_rows": len(train0), "n_test_rows": len(test),
            "train0_weeks": int(train0["Date"].nunique()), "first_test": str(pd.Timestamp(first_test).date()),
            "last_test": str(pd.Timestamp(test_dates[-1]).date()),
            "first_feature_week": str(pd.Timestamp(dates[0]).date())}
    return out, imp, info, df
