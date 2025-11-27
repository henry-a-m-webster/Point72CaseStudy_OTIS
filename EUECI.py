"""
eu_eci_yoy_macro.py

Models EU ECI YoY using CPI YoY and Unemployment, drifting to EU targets
(ECB 2% inflation & ~6.7% equilibrium unemployment) and writes a 2-row
wide CSV:

    EUECIIndex.csv

Output CSV (2-row wide):
    Year,2019 Q1,2019 Q2,2019 Q3,2019 Q4,2019 FY,2020 Q1,...,2028 Q4,2028 FY
    Index,2.5,2.7,2.4,3.0,3.0,3.6,...,4.x,4.x

Notes:
- ECI is modelled as **YoY %** (not a level index).
- EUECI.csv is a wide quarterly file with one data row:
    TIME,2002-Q1,Unnamed:2,2002-Q2,Unnamed:4,...,2025-Q2,Unnamed:188,2025-Q3,...
- EUUnemployment.csv is a wide monthly file with one data row:
    TIME,2000-07,Unnamed:2,2000-08,Unnamed:4,...,2025-09,...
- CPIEU.csv is a long monthly file:
    observation_date, CP0000EZ19M086NEST

Model:
    g_t = ECI YoY_t (in decimal, e.g. 0.04 = 4%)
    g_t = α + ρ * g_{t-1} + β1 * CPI_yoy_t + β2 * Unemp_t + ε_t

We:
1) Parse ECI YoY quarterly from EUECI.csv (wide -> quarterly Series).
2) Build CPI quarterly & CPI YoY quarterly from CPIEU.csv.
3) Build Unemployment quarterly averages from EUUnemployment.csv.
4) Estimate OLS regression on overlapping sample.
5) Build a base macro path where:
      CPI_yoy drifts to TARGET_CPI_YOY (2%)
      Unemployment drifts to TARGET_UNEMP (~6.7%)
   by 2028 Q4.
6) Recursively forecast g_t each quarter to 2028 Q4.
7) Output a 2-row CSV of g_t in percent:
      Row 1: "Year", YYYY Q1..Q4, YYYY FY (FY = Q4)
      Row 2: "Index", g_t*100 values.

Requirements:
    pip install pandas statsmodels
"""

import pandas as pd
import numpy as np
import math
from pathlib import Path
import statsmodels.api as sm

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

DATA_DIR = Path("/Users/henrywebster/Desktop/Cambridge/Recruiting/Lifts Model/Data for Regressions + Modelling/EU")  # folder where your CSVs live

FILE_ECI   = DATA_DIR / "EUECI.csv"
FILE_CPI   = DATA_DIR / "CPIEU.csv"
FILE_UNEMP = DATA_DIR / "EUUnemployment.csv"

OUTPUT_CSV = DATA_DIR / "EUECIIndex.csv"

# Output horizon
START_YEAR_CONFIG = 2019
FORECAST_END_YEAR = 2028  # up to 2028 Q4 and 2028 FY

# EU macro targets (base-case path)
TARGET_CPI_YOY = 0.02  # ECB 2% inflation target (HICP, medium term)
TARGET_UNEMP   = 6.7   # Approx. euro-area equilibrium unemployment (%)


# ---------------------------------------------------------------------
# Loaders & helpers
# ---------------------------------------------------------------------

def load_eu_eci_yoy_quarterly(path: Path) -> pd.Series:
    """
    Load EUECI.csv (wide format, single data row) and return a quarterly
    Series of ECI YoY in decimal (0.04 = 4%).

    EUECI.csv structure (1 row):
        TIME,2002-Q1,Unnamed:2,2002-Q2,Unnamed:4,2002-Q3,...
        row[0] = "European Union - 27 countries (from 2020)", 4.8, NaN, 4.3, ...

    We:
    - Look for column names of the form 'YYYY-Qn'.
    - Parse the corresponding cell in row 0 as a number (percentage).
    - Convert to decimal: v / 100.
    - Index by PeriodIndex with freq='Q'.
    """
    df = pd.read_csv(path)
    if df.shape[0] < 1:
        raise ValueError("EUECI.csv must have at least one data row")

    row = df.iloc[0]
    periods = []
    values = []

    for col in df.columns[1:]:
        if isinstance(col, str) and "-Q" in col:
            try:
                year, q = col.split("-Q")
            except ValueError:
                continue
            if year.isdigit() and q in {"1", "2", "3", "4"}:
                v = pd.to_numeric(row[col], errors="coerce")
                if not math.isnan(v):
                    p = pd.Period(f"{year}Q{q}", freq="Q")
                    periods.append(p)
                    values.append(v / 100.0)  # convert % to decimal

    if not periods:
        raise ValueError("No 'YYYY-Qn' columns with numeric values found in EUECI.csv")

    s = pd.Series(values, index=pd.PeriodIndex(periods, freq="Q")).sort_index()
    return s


def load_eu_cpi_quarterly(path: Path) -> pd.Series:
    """
    Load CPIEU.csv and convert to quarterly average CPI index.
    Columns:
        - 'observation_date'
        - 'CP0000EZ19M086NEST'
    """
    df = pd.read_csv(path)

    if "observation_date" not in df.columns:
        raise ValueError("CPIEU.csv must contain 'observation_date' column")

    df["Date"] = pd.to_datetime(df["observation_date"], errors="coerce")
    value_col = [c for c in df.columns if c != "observation_date"][0]
    df["CPI"] = pd.to_numeric(df[value_col], errors="coerce")

    df = df.dropna(subset=["Date", "CPI"]).set_index("Date").sort_index()

    cpi_q = df["CPI"].resample("Q").mean()
    cpi_q.index = cpi_q.index.to_period("Q")
    return cpi_q.sort_index()


def load_eu_unemployment_quarterly(path: Path) -> pd.Series:
    """
    Load EUUnemployment.csv (wide format, single data row) and convert to
    quarterly average unemployment rate (%).

    Structure:
        TIME,2000-07,Unnamed:2,2000-08,Unnamed:4,2000-09,...
        row[0] = "European Union - 27 countries (from 2020)", 9.5, NaN, 9.4, ...

    We:
    - Look for column names of the form 'YYYY-MM'.
    - Parse the corresponding cell in row 0 as a number (%).
    - Build a monthly series, then resample to quarterly mean.
    """
    df = pd.read_csv(path)
    if df.shape[0] < 1:
        raise ValueError("EUUnemployment.csv must have at least one data row")

    row = df.iloc[0]
    dates = []
    values = []

    for col in df.columns[1:]:
        if isinstance(col, str) and len(col) == 7 and col[4] == "-" \
                and col[:4].isdigit() and col[5:7].isdigit():
            v = pd.to_numeric(row[col], errors="coerce")
            if not math.isnan(v):
                d = pd.to_datetime(col + "-01", errors="coerce")
                if pd.notna(d):
                    dates.append(d)
                    values.append(v)

    if not dates:
        raise ValueError("No 'YYYY-MM' columns with numeric values found in EUUnemployment.csv")

    s = pd.Series(values, index=pd.DatetimeIndex(dates)).sort_index()
    u_q = s.resample("Q").mean()
    u_q.index = u_q.index.to_period("Q")
    return u_q.sort_index()


# ---------------------------------------------------------------------
# Main: estimate regression, forecast g_t, write 2-row CSV
# ---------------------------------------------------------------------

def main():
    # 1) Load series
    eci_yoy_q = load_eu_eci_yoy_quarterly(FILE_ECI)    # g_t, decimal
    cpi_q     = load_eu_cpi_quarterly(FILE_CPI)
    unemp_q   = load_eu_unemployment_quarterly(FILE_UNEMP)

    # 2) CPI YoY (decimal)
    cpi_yoy_q = cpi_q / cpi_q.shift(4) - 1.0

    # 3) Align ECI YoY, CPI YoY, and Unemployment
    df = pd.concat(
        [
            eci_yoy_q.rename("g"),
            cpi_yoy_q.rename("cpi_yoy"),
            unemp_q.rename("unemp"),
        ],
        axis=1
    ).dropna()

    # Add lagged ECI YoY
    df["g_lag1"] = df["g"].shift(1)
    df = df.dropna()

    if df.empty:
        raise ValueError("Not enough overlapping data to estimate EU ECI growth regression.")

    # 4) Estimate:
    #    g_t = const + rho*g_{t-1} + beta1*CPI_yoy_t + beta2*UNEMP_t + eps
    y = df["g"]
    X = df[["g_lag1", "cpi_yoy", "unemp"]]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    params = model.params

    # 5) Forecast g_t out to FORECAST_END_YEAR Q4 with macro drift

    last_period = df.index[-1]
    target_period = pd.Period(f"{FORECAST_END_YEAR}Q4", freq="Q")

    steps = (target_period.year - last_period.year) * 4 + \
            (target_period.quarter - last_period.quarter)

    # Historical g series (full ECI YoY)
    g_hist = eci_yoy_q.dropna().copy()

    if steps > 0:
        forecast_index = pd.period_range(start=last_period + 1, periods=steps, freq="Q")

        # Last observed macro values in regression sample
        last_cpi_yoy = df["cpi_yoy"].iloc[-1]
        last_unemp   = df["unemp"].iloc[-1]
        prev_g       = df["g"].iloc[-1]

        g_forecast_vals = []

        for i, p in enumerate(forecast_index, start=1):
            # Linear convergence of CPI YoY & Unemp to targets by horizon
            w = i / steps  # fraction of the way to the horizon
            cpi_t   = last_cpi_yoy + w * (TARGET_CPI_YOY - last_cpi_yoy)
            unemp_t = last_unemp   + w * (TARGET_UNEMP   - last_unemp)

            # g_t = const + rho*prev_g + beta1*cpi_t + beta2*unemp_t
            x_vec = np.array([1.0, prev_g, cpi_t, unemp_t])
            g_t = float(np.dot(params.values, x_vec))

            g_forecast_vals.append(g_t)
            prev_g = g_t

        g_forecast = pd.Series(g_forecast_vals, index=forecast_index)
        g_full = pd.concat([g_hist, g_forecast]).sort_index()
    else:
        # Already beyond target; no forecast needed
        g_full = g_hist

    # 6) Restrict to START_YEAR_CONFIG onward for output
    start_year = max(START_YEAR_CONFIG, int(g_full.index[0].year))
    g_out = g_full[g_full.index.year >= start_year]

    if g_out.empty:
        raise ValueError(f"No EU ECI YoY data at or after {start_year}Q1 for output.")

    # 7) Build labels & values: for each year start_year..FORECAST_END_YEAR,
    #    add YYYY Q1..Q4 (if present), then YYYY FY, where FY = Q4 g_t
    rows = []
    for year in range(start_year, FORECAST_END_YEAR + 1):
        # Quarterly entries in percent
        for q in range(1, 5):
            p = pd.Period(f"{year}Q{q}", freq="Q")
            if p in g_out.index:
                label = f"{year} Q{q}"
                value = float(g_out.loc[p] * 100.0)  # convert to %
                rows.append((label, value))

        # FY = Q4 YoY
        q4_period = pd.Period(f"{year}Q4", freq="Q")
        if q4_period in g_out.index:
            fy_label = f"{year} FY"
            fy_value = float(g_out.loc[q4_period] * 100.0)
            rows.append((fy_label, fy_value))

    if not rows:
        raise ValueError("No rows constructed for EU ECI index output.")

    labels = [lbl for (lbl, _) in rows]
    values = [val for (_, val) in rows]

    # 8) Build 2-row DataFrame:
    #    Row 1 = 'Year' + labels
    #    Row 2 = 'Index' + values (YoY %, not a level index)
    wide_df = pd.DataFrame(
        [
            ["Year"] + labels,
            ["Index"] + values,
        ]
    )

    # 9) Write WITHOUT header (first row is literally "Year,2019 Q1,...")
    wide_df.to_csv(OUTPUT_CSV, index=False, header=False)

    # Optional: sanity print
    print(f"EU ECI YoY index (macro-driven, to {FORECAST_END_YEAR} FY) written to: {OUTPUT_CSV}")
    print("First few columns:")
    print(wide_df.iloc[:, :8].to_string(index=False))


if __name__ == "__main__":
    main()
