# DemandFlow – revision 1 reproduction package (RINENG-D-26-22476)

This folder reproduces every number, table and figure in the revised manuscript
"Search-Aware E-Commerce Demand Forecasting with Google Trends: A Leakage-Free Evaluation under Controlled Simulation".

## Contents
| File | Purpose |
|---|---|
| `data/amazon.csv` | Amazon India product catalogue (input; sets category scale and base price) |
| `data/google_trends_earbuds_india.csv`, `data/google_trends_smart_watches.csv` | Real weekly Google Trends series, India (inputs) |
| `sim.py` | Controlled demand simulator and sale calendar (Section 3.3, Eq. 1, Table 3) |
| `pipeline.py` | Leakage-free features, models, rolling-origin evaluation, bootstrap CIs, Diebold–Mariano test |
| `select_sarima.py` | SARIMA specification selection on the training window |
| `run_main.py` | Main experiment (seed 42, beta = 0.2): Tables 6–8 |
| `run_replications.py` | 100 replications over beta (Table 9, Fig. 11) and 110 sensitivity runs (Fig. 12) |
| `decision_modules.py` | Launch-window evaluation, counterfactual calendar, price scenarios (Sections 4.7–4.8) |
| `leakage_audit.py` | Re-runs the ORIGINAL pipeline and removes the leaking features (Table 5) |
| `summarize.py` | Collects all quoted numbers into `outputs/numbers.json` and tables into `outputs/tables.xlsx` |
| `make_figures.py` | Draws every revised figure into `figures/` |
| `run_all.py` | Runs everything in order |
| `outputs/` | All simulated data, model inputs, test-period forecasts, metrics and test results (CSV/XLSX/JSON) |

## Reproduce
```
pip install -r requirements.txt
python run_all.py            # full run
python run_all.py --quick    # main experiment + figures only
```
Random seeds: simulator seeds 42 (main) and 1–20 (replications); XGBoost random_state = 42.
`leakage_audit.py` reads the original files `data/processed/merged_sales_trends.csv` and
`data/processed/synthetic_weekly_sales.csv` from the root of the DemandFlow repository.
