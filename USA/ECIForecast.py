"""
eci_index_yoy_macro.py

Builds a quarterly + FY ECI index from ECIData.csv using a YoY-growth
regression on CPI YoY and Unemployment, then writes a 2-row wide CSV:

    ECIIndex.csv

Output CSV (2-row wide):
    Year,2019 Q1,2019 Q2,2019 Q3,2019 Q4,2019 FY,2020 Q1,...,2028 Q4,2028 FY
    Index,123.4,125.6,...

Inputs (CSV, in the same folder as this script):

1) ECIData.csv
   Columns:
     - 'Periodicity' (e.g., 'Current dollar index number')
     - 'Year' (int)
     - 'Period' (e.g., 'March', 'June', 'September', 'December')
     - 'Estimate' (float, ECI level)

2) Unemployment.csv
   Columns:
     - 'observation_date' (YYYY-MM-DD)
     - 'UNRATE' (unemployment rate, %)

3) CPIEALL.csv
   Columns:
     - 'observation_date' (YYYY-MM-DD)
     - 'CPIEALL' (CPI level, index)

Model:

- Build quarterly ECI level series (ECI_q).
- Build quarterly CPI level series, then CPI YoY: CPI_yoy = CPI_q / CPI_q(-4) - 1.
- Build quarterly Unemployment series (quarterly average).
- Build ECI YoY growth: g_t = ECI_q / ECI_q(-4) - 1.

- Regress:
    g_t = alpha + rho * g_{t-1} + beta1 * CPI_yoy_t + beta2 * UNEMP_t + eps_t

- Create a BASE macro path to FORECAST_END_YEAR:
    * CPI YoY drifts linearly from its last observed value to TARGET_CPI_YOY.
    * Unemployment drifts linearly from its last observed value to TARGET_UNEMP.

- Forecast g_t recursively quarter-by-quarter using the regression
  and the forecasted CPI_yoy and UNEMP.

- Reconstruct ECI levels via:
    ECI_t = ECI_{t-4} * (1 + g_t)

- For each year START_YEAR..FORECAST_END_YEAR:
    - Add YYYY Q1..Q4 (if present),
    - Add YYYY FY = YYYY Q4 ECI.

Requirements:
    pip install pandas statsmodels
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

DATA_DIR = Path("/Users/henrywebster/Desktop/Cambridge/Recruiting/Lifts Model/Data for Regressions + Modelling/USA")          # folder where your CSVs live
FILE_ECI   = DATA_DIR / "ECIData.csv"
FILE_UNEMP = DATA_DIR / "Unemployment.csv"
FILE_CPI   = DATA_DIR / "CPIEALL.csv"

OUTPUT_CSV = DATA_DIR / "ECIIndex.csv"

# Output horizon
START_YEAR        = 2019
FORECAST_END_YEAR = 2028  # up to 2028 Q4 and 2028 FY

# Long-run macro anchors (base scenario)
TARGET_CPI_YOY = 0.025   # 2.5% YoY CPI in the long run
TARGET_UNEMP   = 4.5     # 4.5% unemployment in the long run (%)


# ---------------------------------------------------------------------
# Load & transform to quarterly series
# ---------------------------------------------------------------------

def load_eci_quarterly(path: Path) -> pd.Series:
    """
    Load ECIData.csv and return a quarterly Series of ECI levels
    (PeriodIndex with freq='Q'), filtered to 'Current dollar index number'.
    """
    df = pd.read_csv(path)

    required_cols = {"Periodicity", "Year", "Period", "Estimate"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"ECIData.csv missing required columns: {missing}")

    # Keep only the index-level series
    df = df[df["Periodicity"] == "Current dollar index number"].copy()

    # Map quarter-end months
    month_map = {
        "March": 3,
        "June": 6,
        "September": 9,
        "December": 12,
    }
    df["Month"] = df["Period"].map(month_map)

    df = df.dropna(subset=["Year", "Month", "Estimate"])
    df["Year"]     = pd.to_numeric(df["Year"], errors="coerce")
    df["Estimate"] = pd.to_numeric(df["Estimate"], errors="coerce")
    df = df.dropna(subset=["Year", "Month", "Estimate"])

    df["Date"] = pd.to_datetime(
        dict(year=df["Year"].astype(int),
             month=df["Month"].astype(int),
             day=1),
        errors="coerce"
    )
    df = df.dropna(subset=["Date"])
    df["Quarter"] = df["Date"].dt.to_period("Q")

    # Keep last revision per quarter
    df = df.sort_values(["Quarter", "Date"]).drop_duplicates(
        subset=["Quarter"], keep="last"
    )

    eci_q = df.set_index("Quarter")["Estimate"].sort_index()
    eci_q.index = eci_q.index.asfreq("Q")

    return eci_q


def load_unemployment_quarterly(path: Path) -> pd.Series:
    """
    Load Unemployment.csv and convert to quarterly average.
    Columns: 'observation_date', 'UNRATE'
    """
    df = pd.read_csv(path)

    if "observation_date" not in df.columns or "UNRATE" not in df.columns:
        raise ValueError("Unemployment.csv must have 'observation_date' and 'UNRATE' columns")

    df["Date"]   = pd.to_datetime(df["observation_date"], errors="coerce")
    df["UNRATE"] = pd.to_numeric(df["UNRATE"], errors="coerce")
    df = df.dropna(subset=["Date", "UNRATE"]).set_index("Date").sort_index()

    u_q = df["UNRATE"].resample("Q").mean()
    u_q.index = u_q.index.to_period("Q")
    return u_q


def load_cpi_quarterly(path: Path) -> pd.Series:
    """
    Load CPIEALL.csv and convert to quarterly average.
    Columns: 'observation_date', 'CPIEALL'
    """
    df = pd.read_csv(path)

    if "observation_date" not in df.columns or "CPIEALL" not in df.columns:
        raise ValueError("CPIEALL.csv must have 'observation_date' and 'CPIEALL' columns")

    df["Date"]    = pd.to_datetime(df["observation_date"], errors="coerce")
    df["CPIEALL"] = pd.to_numeric(df["CPIEALL"], errors="coerce")
    df = df.dropna(subset=["Date", "CPIEALL"]).set_index("Date").sort_index()

    cpi_q = df["CPIEALL"].resample("Q").mean()
    cpi_q.index = cpi_q.index.to_period("Q")
    return cpi_q


# ---------------------------------------------------------------------
# Main: build YoY-based macro model, forecast, and write 2-row CSV
# ---------------------------------------------------------------------

def main():
    # 1) Load quarterly series
    eci_q = load_eci_quarterly(FILE_ECI)
    u_q   = load_unemployment_quarterly(FILE_UNEMP)
    cpi_q = load_cpi_quarterly(FILE_CPI)

    # 2) Compute YoY ECI and CPI growth
    eci_yoy = eci_q / eci_q.shift(4) - 1.0
    cpi_yoy = cpi_q / cpi_q.shift(4) - 1.0

    # 3) Align ECI YoY, CPI YoY, and Unemployment
    df = pd.concat(
        [
            eci_yoy.rename("g"),        # ECI YoY growth
            cpi_yoy.rename("cpi_yoy"),  # CPI YoY growth
            u_q.rename("unemp"),        # unemployment level
        ],
        axis=1
    ).dropna()

    # Add lagged ECI YoY
    df["g_lag1"] = df["g"].shift(1)
    df = df.dropna()

    if df.empty:
        raise ValueError("Not enough overlapping data to estimate the ECI growth regression.")

    # 4) Estimate regression:
    #    g_t = alpha + rho*g_{t-1} + beta1*CPI_yoy_t + beta2*UNEMP_t + eps
    y = df["g"]
    X = df[["g_lag1", "cpi_yoy", "unemp"]]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    # 5) Forecast ECI YoY (g_t) out to FORECAST_END_YEAR Q4

    last_period = df.index[-1]  # last quarter with g and exog
    target_period = pd.Period(f"{FORECAST_END_YEAR}Q4", freq="Q")

    steps = (target_period.year - last_period.year) * 4 + \
            (target_period.quarter - last_period.quarter)

    # Historical g series (full, including pre-regression part)
    g_hist = eci_yoy.dropna().copy()

    if steps > 0:
        forecast_index = pd.period_range(start=last_period + 1, periods=steps, freq="Q")

        # Last observed macro values in the regression sample
        last_cpi_yoy = df["cpi_yoy"].iloc[-1]
        last_unemp   = df["unemp"].iloc[-1]
        prev_g       = df["g"].iloc[-1]

        params = model.params  # [const, g_lag1, cpi_yoy, unemp]

        g_forecast_vals = []

        for i, p in enumerate(forecast_index, start=1):
            # Linear convergence of CPI YoY to TARGET_CPI_YOY and UNEMP to TARGET_UNEMP
            w = i / steps  # fraction of the way to the horizon
            cpi_t   = last_cpi_yoy + w * (TARGET_CPI_YOY - last_cpi_yoy)
            unemp_t = last_unemp   + w * (TARGET_UNEMP   - last_unemp)

            # Predict g_t using regression:
            # g_t = const + rho*prev_g + beta1*cpi_t + beta2*unemp_t
            x_vec = np.array([1.0, prev_g, cpi_t, unemp_t])
            g_t = float(np.dot(params.values, x_vec))

            g_forecast_vals.append(g_t)
            prev_g = g_t  # update lagged g

        g_forecast = pd.Series(g_forecast_vals, index=forecast_index)
        g_full = pd.concat([g_hist, g_forecast]).sort_index()
    else:
        # No forecast needed; already beyond target
        g_full = g_hist

    # 6) Reconstruct ECI levels using ECI_t = ECI_{t-4} * (1 + g_t)

    # Use only levels up to the last period with g observed (for consistency)
    eci_base = eci_q[eci_q.index <= last_period].copy()
    eci_full = eci_base.copy()

    if steps > 0:
        for p in forecast_index:
            lag = p - 4  # t-4 quarter for YoY relation
            if lag not in eci_full.index:
                raise ValueError(f"Missing lag quarter {lag} for ECI reconstruction")
            eci_full[p] = eci_full[lag] * (1.0 + g_full[p])

    eci_full = eci_full.sort_index()

    # 7) Restrict to START_YEAR onward for output
    start_period = pd.Period(f"{START_YEAR}Q1", freq="Q")
    eci_out = eci_full[eci_full.index >= start_period]

    if eci_out.empty:
        raise ValueError(f"No ECI data at or after {START_YEAR}Q1 for output.")

    # 8) Build combined labels/values:
    #    For each year START_YEAR..FORECAST_END_YEAR:
    #    - add YYYY Q1..Q4 (if present)
    #    - add YYYY FY = YYYY Q4
    rows = []
    for year in range(START_YEAR, FORECAST_END_YEAR + 1):
        # Quarterlies
        for q in range(1, 5):
            period = pd.Period(f"{year}Q{q}", freq="Q")
            if period in eci_out.index:
                label = f"{year} Q{q}"
                value = float(eci_out.loc[period])
                rows.append((label, value))

        # FY = Q4 for that year (if Q4 exists)
        q4_period = pd.Period(f"{year}Q4", freq="Q")
        if q4_period in eci_out.index:
            fy_label = f"{year} FY"
            fy_value = float(eci_out.loc[q4_period])
            rows.append((fy_label, fy_value))

    if not rows:
        raise ValueError("No rows constructed for ECI index output.")

    labels = [lbl for (lbl, _) in rows]
    values = [val for (_, val) in rows]

    # 9) Build 2-row DataFrame: Row 1 = 'Year' + labels, Row 2 = 'Index' + values
    data = [
        ["Year"] + labels,
        ["Index"] + values,
    ]
    wide_df = pd.DataFrame(data)

    # 10) Write WITHOUT header (first row is literally "Year,2019 Q1,...")
    wide_df.to_csv(OUTPUT_CSV, index=False, header=False)

    # Optional: quick sanity print
    print(f"ECI index (YoY + macro regression, to {FORECAST_END_YEAR} FY) written to: {OUTPUT_CSV}")
    print("First few columns:")
    print(wide_df.iloc[:, :8].to_string(index=False))


if __name__ == "__main__":
    main()
