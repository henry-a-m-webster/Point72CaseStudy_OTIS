"""
us_construction_index_from_csv.py

Builds a US Construction Index (quarterly + FY) from your CSVs and
writes it to a wide CSV:

    USConstructionIndex.csv

Output CSV (2-row wide):
    Year,2019 Q1,2019 Q2,2019 Q3,2019 Q4,2019 FY,2020 Q1,...,2028 Q4,2028 FY
    Index,100.00,101.23,...

Data (CSV, monthly) expected in the SAME folder as this script:

1) ConstructionSpending.csv
   - Columns: 'Period', 'Value/$MM'
   - Total or residential construction spending

2) NonResidentialConstructionSpending.csv
   - Columns: 'observation_date', 'TLNRESCONS'
   - Non-residential construction spending

3) HousingUnitsCompleted.csv
   - Columns: 'Month', 'Number'
   - Completed housing units

4) NewResidentialConstruction.csv
   - Columns: 'Month', 'Number'
   - New housing units / starts

Base period for normalisation:
- 2016–2019 average = 100 for each component, then combined.

Weights (volume-heavy, with explicit non-res spending):
- NewResidentialConstruction (starts)           : 35%
- HousingUnitsCompleted                         : 30%
- ConstructionSpending (res/total)              : 15%
- NonResidentialConstructionSpending (non-res)  : 20%

Requirements:
    pip install pandas statsmodels
"""

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ---------------------------------------------------------------------
# CONFIG – adjust paths / weights here
# ---------------------------------------------------------------------

DATA_DIR = Path("/Users/henrywebster/Desktop/Cambridge/Recruiting/Lifts Model/Data for Regressions + Modelling/USA")  # folder where your CSVs live

FILE_CONSTRUCTION_SPENDING      = DATA_DIR / "ConstructionSpending.csv"
FILE_NONRES_CONSTRUCTION        = DATA_DIR / "NonResidentialConstructionSpending.csv"
FILE_HOUSING_COMPLETED          = DATA_DIR / "HousingUnitsCompleted.csv"
FILE_NEW_RES_CONSTR             = DATA_DIR / "NewResidentialConstruction.csv"

# Weights for each component (normalised to sum to 1)
WEIGHT_SPENDING_RES    = 0.15  # ConstructionSpending (res/total)
WEIGHT_SPENDING_NONRES = 0.25  # Non-residential construction spending
WEIGHT_COMPLETED       = 0.30  # Housing units completed
WEIGHT_STARTED         = 0.35  # New residential construction (starts)

# Base period for normalisation: 2016–2019 average
BASE_START_YEAR = 2016
BASE_END_YEAR   = 2019

FORECAST_END_QUARTER = "2028Q4"  # inclusive final quarter for forecast

OUTPUT_CSV = DATA_DIR / "USConstructionIndex.csv"


# ---------------------------------------------------------------------
# Helper functions to load and clean each series
# ---------------------------------------------------------------------

def load_spending_csv(path: Path) -> pd.Series:
    """
    Load a spending CSV with columns: 'Period', 'Value/$MM'.
    Returns a monthly Series indexed by DatetimeIndex.
    """
    df = pd.read_csv(path)

    if "Period" not in df.columns or "Value/$MM" not in df.columns:
        raise ValueError(f"{path.name} must have 'Period' and 'Value/$MM' columns")

    # Parse dates like 'Jan-2002'
    df["Date"] = pd.to_datetime(df["Period"].astype(str).str.strip(), errors="coerce")

    # Clean numeric values: remove commas and convert to float
    df["Value"] = (
        df["Value/$MM"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    df = df.dropna(subset=["Date", "Value"]).set_index("Date").sort_index()
    return df["Value"]


def load_nonres_construction_spending(path: Path) -> pd.Series:
    """
    Load NonResidentialConstructionSpending.csv

    Expected columns:
        - 'observation_date' (YYYY-MM-DD)
        - 'TLNRESCONS' (non-res construction spending level)

    Returns a monthly Series indexed by DatetimeIndex.
    """
    df = pd.read_csv(path)

    if "observation_date" not in df.columns or "TLNRESCONS" not in df.columns:
        raise ValueError(
            f"{path.name} must have 'observation_date' and 'TLNRESCONS' columns"
        )

    df["Date"] = pd.to_datetime(df["observation_date"], errors="coerce")
    df["Value"] = pd.to_numeric(df["TLNRESCONS"], errors="coerce")

    df = df.dropna(subset=["Date", "Value"]).set_index("Date").sort_index()
    return df["Value"]


def load_housing_units_completed(path: Path) -> pd.Series:
    """
    Load HousingUnitsCompleted.csv
    Expected columns: 'Month', 'Number'
    Returns a monthly Series (US total units) indexed by DatetimeIndex.
    """
    df = pd.read_csv(path)

    if "Month" not in df.columns or "Number" not in df.columns:
        raise ValueError("HousingUnitsCompleted.csv must have 'Month' and 'Number' columns")

    df["Date"] = pd.to_datetime(df["Month"].astype(str).str.strip(), errors="coerce")

    df["Units"] = (
        df["Number"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df["Units"] = pd.to_numeric(df["Units"], errors="coerce")

    df = df.dropna(subset=["Date", "Units"]).set_index("Date").sort_index()
    return df["Units"]


def load_new_residential_construction(path: Path) -> pd.Series:
    """
    Load NewResidentialConstruction.csv
    Expected columns: 'Month', 'Number'
    Returns a monthly Series (US total units) indexed by DatetimeIndex.

    Interpreted as 'new housing units / starts'.
    """
    df = pd.read_csv(path)

    if "Month" not in df.columns or "Number" not in df.columns:
        raise ValueError("NewResidentialConstruction.csv must have 'Month' and 'Number' columns")

    df["Date"] = pd.to_datetime(df["Month"].astype(str).str.strip(), errors="coerce")

    df["Units"] = (
        df["Number"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df["Units"] = pd.to_numeric(df["Units"], errors="coerce")

    df = df.dropna(subset=["Date", "Units"]).set_index("Date").sort_index()
    return df["Units"]


# ---------------------------------------------------------------------
# Build composite monthly index (base = 2016–2019 average)
# ---------------------------------------------------------------------

def build_monthly_composite_index() -> pd.Series:
    """
    Returns a monthly US Construction Index (rebased so that 2016–2019 average = 100).
    """

    # Load all four components
    spending_res    = load_spending_csv(FILE_CONSTRUCTION_SPENDING)        # total/res spending
    spending_nonres = load_nonres_construction_spending(FILE_NONRES_CONSTRUCTION)
    completed       = load_housing_units_completed(FILE_HOUSING_COMPLETED)
    started         = load_new_residential_construction(FILE_NEW_RES_CONSTR)

    # Combine into a single DataFrame
    df = pd.concat(
        [
            spending_res.rename("spending_res"),
            spending_nonres.rename("spending_nonres"),
            completed.rename("completed"),
            started.rename("started"),
        ],
        axis=1
    ).sort_index()

    # Keep only periods where we have all four series
    df = df.dropna()

    # Base period mask: 2016–2019 inclusive
    base_mask = (df.index.year >= BASE_START_YEAR) & (df.index.year <= BASE_END_YEAR)
    if not base_mask.any():
        raise ValueError(
            f"No overlapping data between {BASE_START_YEAR} and {BASE_END_YEAR} "
            f"to normalise index."
        )

    # Normalise each component so that its 2016–2019 average = 100
    for col in ["spending_res", "spending_nonres", "completed", "started"]:
        base_mean = df.loc[base_mask, col].mean()
        df[col + "_idx"] = df[col] / base_mean * 100.0

    # Weighted composite
    weights = np.array(
        [
            WEIGHT_SPENDING_RES,
            WEIGHT_SPENDING_NONRES,
            WEIGHT_COMPLETED,
            WEIGHT_STARTED,
        ],
        dtype=float
    )
    weights = weights / weights.sum()  # normalise to sum to 1

    df["US_CONSTRUCTION_INDEX"] = (
        weights[0] * df["spending_res_idx"] +
        weights[1] * df["spending_nonres_idx"] +
        weights[2] * df["completed_idx"] +
        weights[3] * df["started_idx"]
    )

    # Rebase composite so that its 2016–2019 average is exactly 100
    base_mean_comp = df.loc[base_mask, "US_CONSTRUCTION_INDEX"].mean()
    df["US_CONSTRUCTION_INDEX"] = df["US_CONSTRUCTION_INDEX"] * (100.0 / base_mean_comp)

    return df["US_CONSTRUCTION_INDEX"]


# ---------------------------------------------------------------------
# Quarterly aggregation, forecasting, FY build
# ---------------------------------------------------------------------

def to_quarterly(series: pd.Series) -> pd.Series:
    """
    Convert monthly series to quarterly by taking the mean within each quarter.
    Returns a Series with PeriodIndex (freq='Q').
    """
    quarterly = series.resample("Q").mean()
    quarterly.index = quarterly.index.to_period("Q")
    return quarterly.sort_index()


def forecast_quarterly_index(quarterly_series: pd.Series,
                             end_period: str) -> pd.Series:
    """
    Forecast quarterly index from last historical quarter up to end_period
    (inclusive), using Holt-Winters with additive trend & seasonality.
    """
    quarterly_series = quarterly_series.sort_index()

    model = ExponentialSmoothing(
        quarterly_series.astype(float),
        trend="add",
        seasonal="add",
        seasonal_periods=4
    )
    fit = model.fit(optimized=True)

    last_period = quarterly_series.index[-1]
    target_period = pd.Period(end_period, freq="Q")

    steps = ((target_period.year - last_period.year) * 4 +
             (target_period.quarter - last_period.quarter))
    if steps <= 0:
        return quarterly_series

    forecast = fit.forecast(steps=steps)
    forecast.index = pd.period_range(
        start=last_period + 1,
        periods=steps,
        freq="Q"
    )

    combined = pd.concat([quarterly_series, forecast]).sort_index()
    return combined


def build_fy_from_quarterly(q_series: pd.Series) -> pd.Series:
    """
    Build FY index as the average of the 4 quarters in each calendar year.
    Returns a Series indexed by year (int).
    """
    df = q_series.to_frame("index")
    df["year"] = df.index.year
    fy = df.groupby("year")["index"].mean()
    return fy


# ---------------------------------------------------------------------
# Main: run pipeline, print to terminal, write 2-row wide CSV
# ---------------------------------------------------------------------

def main():
    # 1) Monthly index (base 2016–2019 = 100)
    monthly_index = build_monthly_composite_index()

    # 2) Quarterly index (full history)
    quarterly_index = to_quarterly(monthly_index)

    # 3) Forecast quarterly to FORECAST_END_QUARTER
    quarterly_full = forecast_quarterly_index(
        quarterly_index,
        end_period=FORECAST_END_QUARTER
    )

    # 4) Restrict to 2019Q1+ for OTIS modelling / display
    start_period = pd.Period("2019Q1", freq="Q")
    quarterly_display = quarterly_full[quarterly_full.index >= start_period]

    # Mark Actual vs Forecast for printing
    last_hist = quarterly_index.index[-1]
    q_df = quarterly_display.to_frame("US_Construction_Index")
    q_df["Year"] = q_df.index.year
    q_df["Quarter"] = q_df.index.quarter
    q_df["Label"] = q_df["Year"].astype(str) + "Q" + q_df["Quarter"].astype(str)
    q_df["Type"] = np.where(q_df.index <= last_hist, "Actual", "Forecast")
    q_df = q_df[["Year", "Quarter", "Label", "US_Construction_Index", "Type"]]

    # 5) FY index (2019+)
    fy_index = build_fy_from_quarterly(quarterly_full)
    fy_index = fy_index[fy_index.index >= 2019]
    fy_df = fy_index.to_frame("US_Construction_Index_FY")
    fy_df["Year"] = fy_df.index
    fy_df["Label"] = fy_df["Year"].astype(str) + " FY"
    fy_df = fy_df[["Year", "Label", "US_Construction_Index_FY"]]

    # --- Print to terminal ---
    print("\n================ QUARTERLY US CONSTRUCTION INDEX (2019Q1 onward) ================")
    print(q_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    print("\n================   ANNUAL (FY) US CONSTRUCTION INDEX (2019FY onward)   ================")
    print(fy_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    # --- Build combined sequence (quarters then FY per year) for CSV ---

    # Prepare quarterly part
    q_out = q_df.copy()
    q_out = q_out.rename(columns={"US_Construction_Index": "Index"})
    q_out["Freq"] = "Q"

    # Prepare FY part
    fy_out = fy_df.copy()
    fy_out = fy_out.rename(columns={"US_Construction_Index_FY": "Index"})
    fy_out["Freq"] = "FY"
    fy_out["Type"] = np.where(
        fy_out["Year"] <= last_hist.year, "Actual", "Forecast"
    )

    # Interleave: for each year, Q1..Q4 then FY
    rows = []
    max_year = fy_out["Year"].max()
    for year in range(2019, int(max_year) + 1):
        subset_q = q_out[q_out["Year"] == year]
        if not subset_q.empty:
            rows.append(subset_q[["Label", "Index"]])
        subset_fy = fy_out[fy_out["Year"] == year]
        if not subset_fy.empty:
            rows.append(subset_fy[["Label", "Index"]])

    combined = pd.concat(rows)

    # Pretty column labels: "2019 Q1" instead of "2019Q1"
    time_labels = []
    for lbl in combined["Label"].tolist():
        if "Q" in lbl and " " not in lbl:
            year = lbl[:4]
            rest = lbl[4:]
            time_labels.append(f"{year} {rest}")
        else:
            time_labels.append(lbl)

    index_values = combined["Index"].tolist()

    # Build 2-row DataFrame: first row "Year", second row "Index"
    data = [
        ["Year"] + time_labels,
        ["Index"] + index_values,
    ]
    wide_df = pd.DataFrame(data)

    # Write WITHOUT header (first row is literally "Year,2019 Q1,...")
    wide_df.to_csv(OUTPUT_CSV, index=False, header=False)
    print(f"\n*** US Construction Index written (2-row wide) to CSV: {OUTPUT_CSV} ***")


if __name__ == "__main__":
    main()
