"""Leakage audit of the ORIGINAL submission pipeline (notebooks 01, 03, 05 of the DemandFlow repository).
Reproduces Table 2 of the first submission, then removes the leaking features one at a time."""
import numpy as np, pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import os
ROOT = os.environ.get("DEMANDFLOW_ROOT", "..")
ORIG = f"{ROOT}/data/processed/merged_sales_trends.csv"
df = pd.read_csv(ORIG, parse_dates=["Date"])
syn = pd.read_csv(f"{ROOT}/data/processed/synthetic_weekly_sales.csv")
counts = {"synthetic_rows": len(syn), "weeks_per_category": int(syn.groupby("Category").size().iloc[0]),
          "rows_after_lag_dropna": len(df), "weeks_after_lag_dropna": int(df["Date"].nunique()),
          "train_rows": int((df.Date < "2024-01-01").sum()), "test_rows": int((df.Date >= "2024-01-01").sum())}
# are the three categories scaled copies of one another?
piv = syn.pivot(index="Date", columns="Category", values="Units_Sold")
counts["corr_between_categories_min"] = float(piv.corr().values[np.triu_indices(3, 1)].min())
# Price_Elasticity is an exact function of the target
pe_corr = np.corrcoef(syn["Price_Elasticity"], syn["Units_Sold"])[0, 1]
counts["corr_PriceElasticity_UnitsSold"] = float(pe_corr)
counts["corr_WeeklyPrice_UnitsSold_within_category_min"] = float(
    syn.groupby("Category").apply(lambda g: np.corrcoef(g.Weekly_Price, g.Units_Sold)[0, 1]).abs().min())

le = {c: i for i, c in enumerate(sorted(df.Category.unique()))}
df["Category_Encoded"] = df.Category.map(le)
df["trend_momentum"] = df["earbuds india"] - df["earbuds india_lag2"]
df["trend_velocity"] = df["earbuds india_lag1"] - df["earbuds india_lag3"]
g = df.groupby("Category")["Units_Sold"]
df["MA2_past"] = g.transform(lambda x: x.shift(1).rolling(2, min_periods=1).mean())
df["MA4_past"] = g.transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
KW = ['earbuds india', 'wireless earphones', 'bluetooth earbuds', 'best earbuds under 2000', 'noise earbuds']
LAGS = [f"{k}_lag{i}" for k in KW for i in range(1, 5)]
BASE = ['Week_of_Year', 'Month', 'Quarter', 'Is_Festive_Season', 'Avg_Price', 'Weekly_Price', 'Price_Elasticity',
        'Units_Sold_MA2', 'Units_Sold_MA4', 'Category_Encoded']
ENH = BASE + KW + LAGS + ['trend_momentum', 'trend_velocity']
tr, te = df[df.Date < "2024-01-01"].dropna(subset=["MA4_past"]), df[df.Date >= "2024-01-01"]


def run(feats):
    m = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                     random_state=42, verbosity=0)
    m.fit(tr[feats], tr.Units_Sold); p = m.predict(te[feats]); y = te.Units_Sold.values
    return {"RMSE": np.sqrt(mean_squared_error(y, p)), "MAE": mean_absolute_error(y, p),
            "MAPE": np.mean(np.abs(y - p) / y) * 100, "R2": r2_score(y, p)}


def swap(f):
    return [{"Units_Sold_MA2": "MA2_past", "Units_Sold_MA4": "MA4_past"}.get(x, x) for x in f]


nopr = lambda f: [x for x in f if x not in ("Weekly_Price", "Price_Elasticity")]
rows = []
for name, A, B in [
    ("As submitted (MA include current week; price derived from target)", BASE, ENH),
    ("Moving averages use past weeks only", swap(BASE), swap(ENH)),
    ("Price features removed", nopr(BASE), nopr(ENH)),
    ("Both corrections", nopr(swap(BASE)), nopr(swap(ENH)))]:
    a, b = run(A), run(B)
    rows.append({"Setting": name, **{f"A_{k}": v for k, v in a.items()}, **{f"B_{k}": v for k, v in b.items()}})
res = pd.DataFrame(rows)
res.to_csv("outputs/table_leakage_audit.csv", index=False)
pd.Series(counts).to_csv("outputs/original_data_counts.csv", header=["value"])
pd.set_option("display.width", 250)
print(res.round(4).to_string()); print(pd.Series(counts))
