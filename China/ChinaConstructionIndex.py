"""
china_construction_index_from_growth.py

Builds a China Construction Index (quarterly + FY) from **growth rates**
in:

    - RealEstateFloorSpace.csv
    - FloorSpaceCommercializedBuildings.csv
    - FloorSpaceofOfficeBuildings.csv

and writes a 2-row wide CSV:

    ChinaConstructionIndex.csv

Key idea:
    NBS floor space series are "Accumulated" (YTD). The more appropriate
    driver is the **Accumulated Growth Rate(%)** rows, not the raw
    accumulated levels. This script builds an index based on those growth
    rates.

We use:
    - Floor Space of Real Estate Started This Year, Accumulated Growth Rate(%)
    - Floor Space of Commercialized Residential Started This Year, Accumulated Growth Rate(%)
    - Floor Space of Office Buildings Started This Year, Accumulated Growth Rate(%)

Workflow:
    1) Parse the wide monthly layout (columns like 'Oct. 2025', 'Sep. 2025', ...).
    2) For each of the three CSVs, pull the "Started This Year, Accumulated Growth Rate(%)"
       row and build a monthly series of growth rates (%).
    3) Rebase each growth series to an index so that **2016–2019 average = 100**:
           index_t = 100 + (g_t - g_base_mean)
       where g_base_mean is the average growth in 2016–2019.
       So 100 = "normal" growth (2016–2019 average), >100 = stronger, <100 = weaker.
    4) Combine the three rebased series into a composite monthly index:
       default weights: 0.4 (Real estate), 0.4 (Commercial res), 0.2 (Office).
    5) Convert monthly → quarterly (mean within each quarter).
    6) Forecast quarterly index to FORECAST_END_YEAR Q4 (2028) with Holt–Winters:
         - additive trend, additive seasonality if >= 8 quarters
         - fallback to trend-only if not enough data or seasonal fit fails
    7) Build FY index as the average of the 4 quarters each year.
    8) Output 2-row CSV (no header):
         Row 1: "Year", YYYY Q1..Q4, YYYY FY, ...
         Row 2: "Index", values...

Requirements:
    pip install pandas statsmodels
"""

import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

DATA_DIR = Path("/Users/henrywebster/Desktop/Cambridge/Recruiting/Lifts Model/Data for Regressions + Modelling/China")  # folder where the CSVs live

FILE_RE   = DATA_DIR / "RealEstateFloorSpace.csv"
FILE_COM  = DATA_DIR / "FloorSpaceCommercializedBuildings.csv"
FILE_OFF  = DATA_DIR / "FloorSpaceofOfficeBuildings.csv"

OUTPUT_CSV = DATA_DIR / "ChinaConstructionIndex.csv"

# Base period to define "normal" growth
BASE_START_YEAR = 2016
BASE_END_YEAR   = 2019

# Output horizon
START_YEAR_CONFIG = 2019      # earliest year you'd like in the output (if data exists)
FORECAST_END_YEAR = 2028      # forecast to 2028 Q4 and 2028 FY

# Component weights for composite growth index
# [Real estate started, Commercialized res started, Office started]
COMP_WEIGHTS = np.array([0.4, 0.4, 0.2])

# Indicator labels (must match the 'Indicators' entries in the CSVs)
LABEL_RE_START_GROWTH = "Floor Space of Real Estate Started This Year, Accumulated Growth Rate(%)"
LABEL_COM_START_GROWTH = "Floor Space of Commercialized Residential Started This Year, Accumulated Growth Rate(%)"
LABEL_OFF_START_GROWTH = "Floor Space of Office Buildings Started This Year, Accumulated Growth Rate(%)"


# ---------------------------------------------------------------------
# Helpers to parse month columns & load growth series
# ---------------------------------------------------------------------

MONTH_MAP = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


def parse_month_col(col: str):
    """
    Parse wide-format monthly column names like 'Oct. 2025' or 'Oct 2005'
    into a datetime (YYYY-MM-01). Return None if it's not a month column.
    """
    col = str(col)
    if col == "Indicators":
        return None

    # Normalise 'Oct. 2025' -> 'Oct 2025'
    col_clean = col.replace(".", "").strip()
    parts = col_clean.split()
    if len(parts) == 2 and parts[0] in MONTH_MAP and parts[1].isdigit():
        year = int(parts[1])
        month = MONTH_MAP[parts[0]]
        try:
            return dt.datetime(year, month, 1)
        except ValueError:
            return None

    return None


def load_growth_indicator(path: Path, indicator_label: str) -> pd.Series:
    """
    Load one "Accumulated Growth Rate(%)" indicator from a wide China
    floor-space CSV (e.g. RealEstateFloorSpace.csv) and return a monthly
    Series of growth rates (%), indexed by DatetimeIndex, sorted.
    """
    df = pd.read_csv(path)

    if "Indicators" not in df.columns:
        raise ValueError(f"{path.name} must have an 'Indicators' column")

    row = df[df["Indicators"] == indicator_label]
    if row.empty:
        raise ValueError(f"Indicator '{indicator_label}' not found in {path.name}")

    row = row.iloc[0]

    data = {}
    for col, val in row.items():
        if col == "Indicators":
            continue
        date = parse_month_col(col)
        if date is not None and pd.notna(val):
            data[date] = float(val)  # growth rate in %

    if not data:
        raise ValueError(f"No valid monthly data parsed for '{indicator_label}' in {path.name}")

    series = pd.Series(data).sort_index()
    return series


def rebase_growth_to_index(series: pd.Series,
                           base_start_year: int,
                           base_end_year: int) -> pd.Series:
    """
    Take a growth-rate series (%), and convert it to an index where
    the 2016–2019 average growth is mapped to 100.

    index_t = 100 + (g_t - g_base_mean)

    So:
        - Index ~ 100 => growth is around the 2016–2019 average
        - Index > 100 => stronger growth than that period
        - Index < 100 => weaker growth
    """
    years = series.index.year
    mask = (years >= base_start_year) & (years <= base_end_year)
    if mask.any():
        g_base_mean = series.loc[mask].mean()
    else:
        g_base_mean = series.mean()

    index = 100.0 + (series - g_base_mean)
    return index


def forecast_quarterly(series_q: pd.Series, end_year: int) -> pd.Series:
    """
    Forecast a quarterly index series from the last observed quarter
    up to end_year Q4 (inclusive).

    - Uses Holt–Winters Exponential Smoothing.
    - If >= 8 quarters of history: additive trend + additive seasonality (period=4).
    - Otherwise (or if seasonal fit fails): additive trend only (no seasonality).
    """
    series_q = series_q.sort_index()
    last_period = series_q.index[-1]
    target_period = pd.Period(f"{end_year}Q4", freq="Q")

    steps = (target_period.year - last_period.year) * 4 + \
            (target_period.quarter - last_period.quarter)
    if steps <= 0:
        return series_q

    n_obs = len(series_q)
    fit = None

    if n_obs >= 8:
        try:
            model = ExponentialSmoothing(
                series_q.astype(float),
                trend="add",
                seasonal="add",
                seasonal_periods=4,
            )
            fit = model.fit(optimized=True)
        except ValueError:
            fit = None

    if fit is None:
        model = ExponentialSmoothing(
            series_q.astype(float),
            trend="add",
            seasonal=None,
        )
        fit = model.fit(optimized=True)

    forecast = fit.forecast(steps=steps)
    forecast.index = pd.period_range(
        start=last_period + 1,
        periods=steps,
        freq="Q",
    )

    combined = pd.concat([series_q, forecast]).sort_index()
    return combined


# ---------------------------------------------------------------------
# Main: build China Construction Index & write 2-row CSV
# ---------------------------------------------------------------------

def main():
    # 1) Load the three growth-rate series (%)
    re_start_g = load_growth_indicator(FILE_RE, LABEL_RE_START_GROWTH)
    com_start_g = load_growth_indicator(FILE_COM, LABEL_COM_START_GROWTH)
    off_start_g = load_growth_indicator(FILE_OFF, LABEL_OFF_START_GROWTH)

    # 2) Align to common monthly index
    components = pd.concat(
        [
            re_start_g.rename("re_start_g"),
            com_start_g.rename("com_res_start_g"),
            off_start_g.rename("off_start_g"),
        ],
        axis=1,
    ).dropna()

    if components.empty:
        raise ValueError("No overlapping monthly data across the three growth series.")

    # 3) Rebase each growth series to an index (2016–2019 avg = 100)
    rebased = pd.DataFrame(index=components.index)
    for col in components.columns:
        rebased[col] = rebase_growth_to_index(
            components[col],
            BASE_START_YEAR,
            BASE_END_YEAR,
        )

    # 4) Composite monthly index as weighted sum of rebased growth indices
    weights = COMP_WEIGHTS / COMP_WEIGHTS.sum()
    composite = (
        weights[0] * rebased["re_start_g"] +
        weights[1] * rebased["com_res_start_g"] +
        weights[2] * rebased["off_start_g"]
    )
    composite.name = "China_Construction_GrowthIndex"

    # 5) Convert monthly -> quarterly (mean)
    china_q = composite.resample("Q").mean()
    china_q.index = china_q.index.to_period("Q")
    china_q = china_q.sort_index()

    if china_q.empty:
        raise ValueError("No quarterly data could be derived from the composite growth index.")

    # 6) Forecast quarterly to FORECAST_END_YEAR Q4
    china_q_full = forecast_quarterly(china_q, FORECAST_END_YEAR)

    # 7) Build FY index (average of quarters)
    df_q = china_q_full.to_frame("Index")
    df_q["Year"] = df_q.index.year
    fy_full = df_q.groupby("Year")["Index"].mean()

    # 8) Decide start year for output
    first_year_in_data = int(china_q_full.index[0].year)
    start_year = max(START_YEAR_CONFIG, first_year_in_data)

    start_period = pd.Period(f"{start_year}Q1", freq="Q")
    china_q_out = china_q_full[china_q_full.index >= start_period]
    fy_out = fy_full[fy_full.index >= start_year]

    if china_q_out.empty:
        raise ValueError(f"No quarterly China construction data at or after {start_year}Q1.")

    # 9) Build labels & values: YYYY Q1..Q4, then YYYY FY
    rows = []
    for year in range(start_year, FORECAST_END_YEAR + 1):
        for q in range(1, 5):
            p = pd.Period(f"{year}Q{q}", freq="Q")
            if p in china_q_out.index:
                rows.append((f"{year} Q{q}", float(china_q_out.loc[p])))
        if year in fy_out.index:
            rows.append((f"{year} FY", float(fy_out.loc[year])))

    if not rows:
        raise ValueError("No rows constructed for China Construction Index output.")

    labels = [lbl for lbl, _ in rows]
    values = [val for _, val in rows]

    # 10) Build 2-row DataFrame: Row 1 = 'Year' + labels, Row 2 = 'Index' + values
    wide_df = pd.DataFrame(
        [
            ["Year"] + labels,
            ["Index"] + values,
        ]
    )

    # 11) Write WITHOUT header
    wide_df.to_csv(OUTPUT_CSV, index=False, header=False)

    # Optional quick print
    print(f"China Construction Index (growth-based) written to: {OUTPUT_CSV}")
    print("First few columns:")
    print(wide_df.iloc[:, :8].to_string(index=False))


if __name__ == "__main__":
    main()
