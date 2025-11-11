# ev_forecast_starter.py
"""
Starter: Fit an S-curve (logistic) or Bass diffusion to EV adoption data,
then forecast 5 and 10 years ahead under scenarios.

USAGE
-----
1) Export your historical data from StatsCan/IEA into a CSV with columns:
   period, ev_share_new (0-1), total_new_registrations (optional), stock, ev_share_stock (optional)
   - period can be annual (e.g., 2018) or quarterly (e.g., 2018-Q1). If quarterly, consider converting to annual for simplicity.

2) Adjust PARAMETERS + SCENARIOS below, then run:
   python ev_forecast_starter.py

3) Outputs:
   - forecast_results.csv (per-scenario new sales share + stock share)
   - quick_plots.png (optional line plots if matplotlib is available)
"""
import re
import math
import pandas as pd
import numpy as np
from pathlib import Path

# ------------------ Load Data ------------------
DATA_CSV = Path("history_input.csv")  # Put your cleaned history here

if not DATA_CSV.exists():
    # Create an example file to show the schema
    demo = pd.DataFrame({
        "period": [2018, 2019, 2020, 2021, 2022, 2023, 2024],
        "ev_share_new": [0.02, 0.025, 0.04, 0.065, 0.095, 0.12, 0.145],
        "total_new_registrations": [1.6e6, 1.7e6, 1.5e6, 1.6e6, 1.55e6, 1.65e6, 1.7e6],
        "ev_share_stock": [np.nan]*7,
        "stock": [np.nan]*7
    })
    demo.to_csv(DATA_CSV, index=False)

df = pd.read_csv(DATA_CSV)
if "period_index" not in df.columns:
    # Create a simple 0-based time index
    # If 'period' is YYYY-Qx, convert to an integer index with equally spaced steps
    def period_to_index(val):
        s = str(val)
        m = re.match(r"(\d{4})(?:-?Q([1-4]))?$", s)
        if m:
            year = int(m.group(1))
            if m.group(2):
                q = int(m.group(2))
                return (year - df['period'].astype(str).str[:4].astype(int).min())*4 + (q-1)
            else:
                return year - int(df['period'].min())
        try:
            return int(s) - int(df['period'].min())
        except:
            return np.nan
    df["period_index"] = df["period"].apply(period_to_index)
df = df.sort_values("period_index").reset_index(drop=True)

# ------------------ Parameters ------------------
MODEL_TYPE = "logistic"  # "logistic" or "bass"
# Logistic initial guesses / defaults
K = 0.85      # saturation (new sales share)
r = 0.45      # intrinsic growth (per period; if quarterly, adjust)
t0 = df.index.min() + (df.index.max() - df.index.min())/2  # midpoint index

# Bass defaults
p = 0.03
q = 0.38

# Fleet turnover assumptions
avg_lifetime_years = 12
periods_per_year = 4  # change to 4 if using quarters
turnover_rate = 1.0 / (avg_lifetime_years * periods_per_year)

# Scenarios as multipliers on r and/or K
SCENARIOS = {
    "Low":    {"K_mult": 0.85, "r_mult": 0.7},
    "Medium": {"K_mult": 1.00, "r_mult": 1.0},
    "High":   {"K_mult": 1.05, "r_mult": 1.2},
}

HORIZON_YEARS = [5, 10]

# ------------------ Helper functions ------------------
def logistic_share(t, K, r, t0):
    return K / (1.0 + math.exp(-r * (t - t0)))

def bass_cumulative_share(t, p, q, K=1.0):
    # Bass cumulative adoption F(t) scaled by a cap K (if you want to cap at <1)
    # F(t) = 1 - exp(-(p+q)t) / (1 + (q/p) * exp(-(p+q)t))
    if t < 0: 
        return 0.0
    exp_term = math.exp(-(p+q)*t)
    F = 1.0 - exp_term / (1.0 + (q/p) * exp_term)
    return min(K * F, K)

def to_stock_share(new_share_series, initial_stock_share=0.03, turnover=turnover_rate):
    stock_share = [initial_stock_share]
    for s in new_share_series[1:]:
        # Simple stock dynamics: next_stock = (1 - turnover)*stock + turnover*new_sales_share
        next_stock = (1.0 - turnover)*stock_share[-1] + turnover*s
        stock_share.append(next_stock)
    return stock_share

# ------------------ Fit a simple model (optional) ------------------
# Here we do a minimal least-squares fit for logistic parameters using new_sales_share data if present
if df["ev_share_new"].notna().sum() >= 4 and MODEL_TYPE == "logistic":
    from math import log
    # Coarse grid search to keep it dependency-free
    y = df["ev_share_new"].values
    t = df["period_index"].values.astype(float)
    best = None
    for K_try in np.linspace(0.5, 0.95, 10):
        for r_try in np.linspace(0.1, 1.2, 12):
            for t0_try in np.linspace(t.min(), t.max()+8, 12):
                yhat = np.array([logistic_share(tt, K_try, r_try, t0_try) for tt in t])
                mse = np.nanmean((yhat - y)**2)
                if best is None or mse < best[0]:
                    best = (mse, K_try, r_try, t0_try)
    _, K, r, t0 = best

# ------------------ Build forecasts ------------------
results = []
t_last = df["period_index"].max()
for scen, mults in SCENARIOS.items():
    K_s = K * mults["K_mult"]
    r_s = r * mults["r_mult"]
    # Extend horizon
    extra_periods = max(int(round(y*periods_per_year)) for y in HORIZON_YEARS)
    t_future = np.arange(t_last+1, t_last+1+extra_periods)
    t_all = np.concatenate([df["period_index"].values, t_future])

    if MODEL_TYPE == "logistic":
        new_share_model = [logistic_share(tt, K_s, r_s, t0) for tt in t_all]
    else:
        # Re-center t for Bass (assume t0 ~ first observed period)
        t0_bass = t_all - t_all.min()
        new_share_model = [bass_cumulative_share(tt, p, q, K=K_s) for tt in t0_bass]

    stock_share_model = to_stock_share(new_share_model, initial_stock_share=float(df.get("ev_share_stock", pd.Series([0.03])).fillna(0.03).iloc[0]))
    tmp = pd.DataFrame({
        "period_index": t_all,
        "scenario": scen,
        "new_sales_share_model": new_share_model,
        "stock_share_est": stock_share_model
    })
    results.append(tmp)

out = pd.concat(results).reset_index(drop=True)

# === NEW: label calendar year/quarter for quarterly data (YYYY-Q#) ===
import re

# Find the first observed period (min period_index) and parse its year/quarter
first_idx = df["period_index"].idxmin()
m = re.match(r"(\d{4})-?Q([1-4])$", str(df.loc[first_idx, "period"]))
if not m:
    raise ValueError("Expected period like 'YYYY-Q#' (e.g., 2018-Q1).")

base_year = int(m.group(1))
base_q = int(m.group(2))  # 1..4

# Offset in quarters from the base period
offset = out["period_index"] - out["period_index"].min()

# Convert offsets to calendar year/quarter
out["year"] = base_year + ((offset + (base_q - 1)) // 4)
out["quarter"] = ((offset + (base_q - 1)) % 4) + 1
out["year_q"] = out["year"].astype(str) + "-Q" + out["quarter"].astype(str)

# Save results
out_path = Path("forecast_results.csv")
out.to_csv(out_path, index=False)

# Optional: quick plot if matplotlib exists
try:
    import matplotlib.pyplot as plt
    for scen in out["scenario"].unique():
        sub = out[out["scenario"]==scen]
        plt.figure()
        plt.plot(sub["period_index"], sub["new_sales_share_model"], label=f"{scen} - new sales share")
        plt.plot(sub["period_index"], sub["stock_share_est"], label=f"{scen} - stock share")
        plt.legend()
        plt.title(f"EV Adoption Forecast — {scen}")
        plt.xlabel("Period Index")
        plt.ylabel("Share (0-1)")
        plt.tight_layout()
    plt.savefig("quick_plots.png", dpi=160)
except Exception as e:
    pass

print(f"Saved: {out_path.resolve()}")
