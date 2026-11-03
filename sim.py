"""
Controlled demand simulation for the revised study (RINENG-D-26-22476, revision 1).

Weekly demand for category c in week t:

    D[c,t] = B[c] * G[t] * S[t] * P[t] * M[t] * eps[c,t]

    B[c]  base weekly demand  = mean(rating_count of category c in the Amazon India catalogue) / 52
    G[t]  linear growth       = 1 + g * (t - 1) / 52                       (g = annual growth rate)
    S[t]  smooth seasonality  = 1 + A * sin(2*pi*(woy[t] - phi) / 52)      (woy = ISO week of year)
    P[t]  promotion multiplier on the actual sale-event weeks of each year (see sale_calendar)
    M[t]  search coupling     = max(0.2, 1 + beta * z[t - L])             (z = standardised real Google Trends index)
    eps   multiplicative noise ~ N(1, sigma^2), independent for every category and replication

beta = 0 gives demand that has no link to search interest (placebo condition).
All parameters are explicit arguments so that every scenario in the paper can be regenerated.
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

CATEGORIES = {
    "USBCables": "Computers&Accessories|Accessories&Peripherals|Cables&Accessories|Cables|USBCables",
    "HDMICables": "Electronics|HomeTheater,TV&Video|Accessories|Cables|HDMICables",
    "WirelessUSBAdapters": "Computers&Accessories|NetworkingDevices|NetworkAdapters|WirelessUSBAdapters",
}

# Actual sale-event dates used to place promotions (verifiable public calendar).
DIWALI = {2023: "2023-11-12", 2024: "2024-11-01", 2025: "2025-10-20"}
REPUBLIC_DAY = {2024: "2024-01-26", 2025: "2025-01-26", 2026: "2026-01-26"}
PRIME_DAY = {2023: "2023-07-15", 2024: "2024-07-20", 2025: "2025-07-12"}


def load_base_demand():
    df = pd.read_csv(DATA / "amazon.csv")
    df["rating_count"] = pd.to_numeric(df["rating_count"].astype(str).str.replace(",", ""), errors="coerce").fillna(0)
    out = {}
    for short, full in CATEGORIES.items():
        sub = df[df["category"] == full]
        out[short] = {"base": sub["rating_count"].mean() / 52.0, "n_products": len(sub),
                      "avg_price": pd.to_numeric(sub["discounted_price"].astype(str).str.replace("₹", "").str.replace(",", ""), errors="coerce").mean()}
    return out


def load_trends(name="earbuds india"):
    f = {"earbuds india": "google_trends_earbuds_india.csv", "smart watches": "google_trends_smart_watches.csv"}[name]
    g = pd.read_csv(DATA / f, parse_dates=["Date"])
    g = g.rename(columns={name: "GT"}).set_index("Date")["GT"].astype(float)
    return g


def _week_start(d):
    d = pd.Timestamp(d)
    return d - pd.Timedelta(days=(d.weekday() + 1) % 7)  # Sunday that starts the week (Trends convention)


def sale_calendar(dates, festive_mult=1.40, rday_mult=1.25, prime_mult=1.20, shift_weeks=0):
    """Return promotion multiplier and event flags for each week (weeks start on Sunday).
    Festive sale = Diwali week and the three weeks before it (Great Indian Festival period).
    Republic Day sale = week containing 26 January. Prime Day = week containing the event.
    shift_weeks moves every event (used for the counterfactual launch-timing test)."""
    dates = pd.DatetimeIndex(dates)
    shift = pd.Timedelta(weeks=shift_weeks)
    fest = np.zeros(len(dates), int); rday = np.zeros(len(dates), int); prime = np.zeros(len(dates), int)
    for d in DIWALI.values():
        w = _week_start(d) + shift
        for k in range(0, 4):
            fest[dates == w - pd.Timedelta(weeks=k)] = 1
    for d in REPUBLIC_DAY.values():
        rday[dates == _week_start(d) + shift] = 1
    for d in PRIME_DAY.values():
        prime[dates == _week_start(d) + shift] = 1
    mult = np.ones(len(dates))
    mult[fest == 1] *= festive_mult
    mult[rday == 1] *= rday_mult
    mult[prime == 1] *= prime_mult
    return mult, fest, rday, prime


def simulate(seed=42, beta=0.15, lag=2, sigma=0.10, growth=0.15, amp=0.10, phi=30,
             festive_mult=1.40, rday_mult=1.25, prime_mult=1.20, signal="earbuds india",
             shift_weeks=0):
    gt = load_trends(signal)
    z = (gt - gt.mean()) / gt.std()
    dates = gt.index[lag:]                       # first `lag` weeks only feed the coupling term
    zl = z.shift(lag).loc[dates].values
    n = len(dates)
    t = np.arange(1, n + 1)
    woy = dates.isocalendar().week.values.astype(int)
    G = 1 + growth * (t - 1) / 52.0
    S = 1 + amp * np.sin(2 * np.pi * (woy - phi) / 52.0)
    P, fest, rday, prime = sale_calendar(dates, festive_mult, rday_mult, prime_mult, shift_weeks)
    M = np.maximum(0.2, 1 + beta * zl)
    base = load_base_demand()
    rng = np.random.default_rng(seed)
    rows = []
    for c, info in base.items():
        eps = rng.normal(1.0, sigma, n)
        d = info["base"] * G * S * P * M * eps
        d = np.maximum(np.round(d), 0)
        rows.append(pd.DataFrame({"Date": dates, "Category": c, "Units_Sold": d,
                                  "Festive_Sale": fest, "RepublicDay_Sale": rday, "PrimeDay_Sale": prime}))
    sales = pd.concat(rows, ignore_index=True)
    return sales, gt
