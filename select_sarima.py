"""Select the seasonal ARIMA specification on the initial training window only (main seed),
using one-step-ahead MAPE on the last 13 training weeks as the validation criterion. AIC is reported too."""
import itertools, json, warnings
import numpy as np, pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import sim, pipeline as P
warnings.filterwarnings("ignore")

sales, gt = sim.simulate(seed=42)
df = P.build_features(sales, gt)
dates = np.sort(df["Date"].unique())
first_test = dates[-P.TEST_WEEKS]
val_dates = dates[(dates < first_test)][-13:]
cands = []
for (p, d, q) in itertools.product((0, 1), (0, 1), (0, 1)):
    for (Ps, Qs) in ((0, 0), (1, 0), (0, 1), (1, 1)):
        cands.append(((p, d, q), (Ps, 1, Qs, 52), "c" if d == 0 else None))
rows = []
for order, sorder, trend in cands:
    mapes, aics = [], []
    for c in df["Category"].unique():
        s = df[df["Category"] == c].set_index("Date")["Units_Sold"].astype(float)
        s = s[s.index < first_test]; s.index.freq = "W-SUN"
        try:
            pr, _, _ = P.sarima_rolling(s, None, val_dates, order, sorder, refit_every=13, trend=trend)
            mapes.append(P.metrics(s.loc[val_dates].values, pr)["MAPE"])
            aics.append(SARIMAX(s[s.index < val_dates[0]], order=order, seasonal_order=sorder, trend=trend,
                                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False).aic)
        except Exception as e:
            mapes.append(np.nan); aics.append(np.nan)
    rows.append({"order": str(order), "seasonal_order": str(sorder), "trend": trend,
                 "val_MAPE": np.mean(mapes), "mean_AIC": np.mean(aics)})
    print(rows[-1], flush=True)
r = pd.DataFrame(rows).sort_values("val_MAPE")
r.to_csv("outputs/sarima_selection.csv", index=False)
print(r.head(8).to_string())
