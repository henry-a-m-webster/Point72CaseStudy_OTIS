"""
eu_construction_index_from_pmi.py

Builds an EU Construction Production Index (quarterly + FY) from
EUConstructionPMI.csv and writes it to a 2-row wide CSV:

    EUConstructionIndex.csv

Input format (wide, 2 rows):

Row 0 (index 0): TIME, 2005-01, NaN, 2005-02, NaN, 2005-03, NaN, ...
Row 1 (index 1): Euro area – 20 countries (from 2023), 108.0, i, 107.6, i, 115.4, i, ...

Pattern:
- Every time label (YYYY-MM) is in row 0.
- The numeric value for that month is in row 1 in the same column.
- Interleaved columns contain "i" or NaN (we ignore those).

Steps:
1) Parse row 0 & row 1 to build a monthly Series: Date -> Value.
2) Rebase so that the 2016–2019 average = 100 (if available;
   otherwise use the sample mean).
3) Convert to quarterly (mean within each quarter).
4) Forecast to FORECAST_END_YEAR Q4 using Holt–Winters:
   - If >= 8 quarters of history: additive trend + additive seasonality (period=4).
   - Otherwise: additive trend only (no seasonality).
5) Build FY index as the average of the 4 quarters in each calendar year.
6) For each year from max(START_YEAR_CONFIG, first year in data)
   to FORECAST_END_YEAR, create columns:
       YYYY Q1, YYYY Q2, YYYY Q3, YYYY Q4, YYYY FY
7) Write a 2-row CSV (no header row) of the form:
       Row 1: "Year", labels...
       Row 2: "Index", values...

Requirements:
    pip install pandas statsmodels
"""

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

DATA_DIR = Path("/Users/henrywebster/Desktop/Cambridge/Recruiting/Lifts Model/Data for Regressions + Modelling/EU")  # folder where EUConstructionPMI.csv lives

FILE_EU_PMI = DATA_DIR / "EUConstructionPMI.csv"
OUTPUT_CSV  = DATA_DIR / "EUConstructionIndex.csv"

# Base period for rebasing index: 2016–2019 average = 100
BASE_START_YEAR = 2016
BASE_END_YEAR   = 2019

# Output horizon
START_YEAR_CONFIG = 2019      # earliest year you'd like if data exists
FORECAST_END_YEAR = 2028      # up to 2028 Q4 and 2028 FY


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_monthly_from_wide_pmi(path: Path) -> pd.Series:
    """
    Load EUConstructionPMI.csv in the wide, 2-row format and return a
    monthly Series indexed by DatetimeIndex.

    Assumes:
        - Row 0: TIME labels, including 'YYYY-MM' strings.
        - Row 1: values for 'Euro area – 20 countries (from 2023)'.
        - Interleaved columns with 'i' or NaN (ignored).
    """
    df = pd.read_csv(path, header=None)

    if df.shape[0] < 2:
        raise ValueError("Expected at least 2 rows in EUConstructionPMI.csv")

    time_row = df.iloc[0]
    val_row  = df.iloc[1]

    pairs = []
    # Start from column 1; col 0 is 'TIME'
    for j in range(1, df.shape[1]):
        t = time_row[j]
        # Look for strings of form 'YYYY-MM'
        if isinstance(t, str) and len(t) == 7 and t[4] == "-":
            v = pd.to_numeric(val_row[j], errors="coerce")
            if not pd.isna(v):
                pairs.append((t, v))

    if not pairs:
        raise ValueError("No (time, value) pairs parsed from EUConstructionPMI.csv")

    dates  = pd.to_datetime([t for t, _ in pairs])
    values = [v for _, v in pairs]

    series = pd.Series(values, index=dates).sort_index()
    return series


def rebase_index(series: pd.Series,
                 base_start_year: int,
                 base_end_year: int) -> pd.Series:
    """
    Rebase a monthly or quarterly series so that its average over
    base_start_year..base_end_year is 100.

    If that window isn't present in the data, fall back to the sample mean.
    """
    mask = (series.index.year >= base_start_year) & (series.index.year <= base_end_year)
    if mask.any():
        base_mean = series.loc[mask].mean()
    else:
        base_mean = series.mean()

    return series / base_mean * 100.0


def to_quarterly(series: pd.Series) -> pd.Series:
    """
    Convert a monthly series to quarterly by taking the mean within each quarter.
    Returns a Series with PeriodIndex (freq='Q').
    """
    quarterly = series.resample("Q").mean()
    quarterly.index = quarterly.index.to_period("Q")
    return quarterly.sort_index()


def forecast_quarterly(series_q: pd.Series,
                       end_year: int) -> pd.Series:
    """
    Forecast a quarterly series from the last observed quarter up to
    end_year Q4 (inclusive).

    Logic:
    - If we have enough data (>= 8 quarters), try Holt–Winters with
      additive trend + additive seasonality (period=4).
    - If that fails (or history is shorter), fall back to a simpler
      additive-trend-only model (no seasonality).
    """
    series_q = series_q.sort_index()
    last_period   = series_q.index[-1]
    target_period = pd.Period(f"{end_year}Q4", freq="Q")

    steps = ((target_period.year - last_period.year) * 4 +
             (target_period.quarter - last_period.quarter))
    if steps <= 0:
        # Already have data up to (or beyond) the target
        return series_q

    n_obs = len(series_q)
    fit = None

    # Try seasonal model if enough data
    if n_obs >= 8:
        try:
            model = ExponentialSmoothing(
                series_q.astype(float),
                trend="add",
                seasonal="add",
                seasonal_periods=4
            )
            fit = model.fit(optimized=True)
        except ValueError:
            fit = None

    # Fallback: trend-only (no seasonality)
    if fit is None:
        model = ExponentialSmoothing(
            series_q.astype(float),
            trend="add",
            seasonal=None
        )
        fit = model.fit(optimized=True)

    forecast = fit.forecast(steps=steps)
    forecast.index = pd.period_range(
        start=last_period + 1,
        periods=steps,
        freq="Q"
    )

    combined = pd.concat([series_q, forecast]).sort_index()
    return combined


def build_fy_from_quarterly(series_q: pd.Series) -> pd.Series:
    """
    Build FY index as the average of the 4 quarters in each calendar year.
    Returns a Series indexed by year (int).
    """
    df = series_q.to_frame("Index")
    df["Year"] = df.index.year
    fy = df.groupby("Year")["Index"].mean()
    return fy


# ---------------------------------------------------------------------
# Main: pipeline & write 2-row wide CSV
# ---------------------------------------------------------------------

def main():
    # 1) Load monthly EU construction series from wide PMI file
    eu_monthly = load_monthly_from_wide_pmi(FILE_EU_PMI)

    # 2) Rebase to 2016–2019 = 100 (or sample mean if that window is missing)
    eu_monthly_idx = rebase_index(eu_monthly, BASE_START_YEAR, BASE_END_YEAR)

    # 3) Convert to quarterly
    eu_quarterly = to_quarterly(eu_monthly_idx)
    if eu_quarterly.empty:
        raise ValueError("No quarterly data could be derived from EUConstructionPMI.csv")

    # 4) Forecast quarterly to FORECAST_END_YEAR Q4
    eu_quarterly_full = forecast_quarterly(eu_quarterly, FORECAST_END_YEAR)

    # 5) Build FY index (average of quarters)
    fy_full = build_fy_from_quarterly(eu_quarterly_full)

    # 6) Decide start year: max(configured, first year in data)
    first_year_in_data = int(eu_quarterly_full.index[0].year)
    start_year = max(START_YEAR_CONFIG, first_year_in_data)

    # 7) Restrict to start_year onward for output
    start_period = pd.Period(f"{start_year}Q1", freq="Q")
    eu_quarterly_out = eu_quarterly_full[eu_quarterly_full.index >= start_period]
    fy_out = fy_full[fy_full.index >= start_year]

    if eu_quarterly_out.empty:
        raise ValueError(f"No quarterly EU construction data at or after {start_year}Q1.")

    # 8) Build combined labels/values:
    #    For each year start_year..FORECAST_END_YEAR,
    #    add YYYY Q1..Q4 (if present), then YYYY FY (if present).
    rows = []
    for year in range(start_year, FORECAST_END_YEAR + 1):
        # Quarterly entries
        for q in range(1, 5):
            p = pd.Period(f"{year}Q{q}", freq="Q")
            if p in eu_quarterly_out.index:
                label = f"{year} Q{q}"
                val = float(eu_quarterly_out.loc[p])
                rows.append((label, val))

        # FY entry
        if year in fy_out.index:
            fy_label = f"{year} FY"
            fy_val = float(fy_out.loc[year])
            rows.append((fy_label, fy_val))

    if not rows:
        raise ValueError("No rows constructed for EU construction index output.")

    labels = [lbl for (lbl, _) in rows]
    values = [val for (_, val) in rows]

    # 9) Build 2-row DataFrame:
    #    Row 1 = 'Year' + labels
    #    Row 2 = 'Index' + values
    wide_df = pd.DataFrame(
        [
            ["Year"] + labels,
            ["Index"] + values,
        ]
    )

    # 10) Write WITHOUT header (first row is literally "Year,2019 Q1,...")
    wide_df.to_csv(OUTPUT_CSV, index=False, header=False)

    # Optional: quick sanity print
    print(f"EU Construction Index written (2-row wide) to: {OUTPUT_CSV}")
    print("First few columns:")
    print(wide_df.iloc[:, :8].to_string(index=False))


if __name__ == "__main__":
    main()
