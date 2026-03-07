import pandas as pd
import numpy as np
from pathlib import Path


def read_csv_auto_sep(path: Path) -> pd.DataFrame:
    """Try ';' then ',' separator."""
    df = pd.read_csv(path, sep=";")
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=",")
    return df


def parse_order_date_to_day(s: pd.Series) -> pd.Series:
    """
    Robust date parsing for:
    - mm/dd/yy
    - mm/dd/yyyy
    - with optional time after space
    """

    s = s.astype(str).str.strip()

    # Keep only the date part (before space)
    date_part = s.str.split().str[0]

    # Let pandas auto-detect format (works in pandas >=2.0)
    dt = pd.to_datetime(date_part, errors="coerce")

    return dt.dt.date


def build_demand_features(input_csv: Path, tag: str, period: str = "D") -> None:
    """
    Compute mean demand, coefficient of variation, demand frequency per SKU.
    period: currently only "D" supported (daily). Kept as param for future extension.
    """
    print(f"\n==============================")
    print(f"Processing: {input_csv}  (tag={tag})")
    print(f"==============================")

    df = read_csv_auto_sep(input_csv)

    needed = {"item_code", "item_quantity", "order_date"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{input_csv.name} missing columns: {missing}. Found: {list(df.columns)}")

    df["item_code"] = df["item_code"].astype(str)
    df["item_quantity"] = pd.to_numeric(df["item_quantity"], errors="coerce").fillna(0)

    # Parse to day
    df["day"] = parse_order_date_to_day(df["order_date"])

    # Drop rows with unparseable date
    before = len(df)
    df = df.dropna(subset=["day"]).copy()
    after = len(df)
    if after < before:
        print(f"Dropped {before-after:,} rows due to invalid order_date parsing")

    # Aggregate to daily demand per SKU
    daily = (
        df.groupby(["item_code", "day"], as_index=False)["item_quantity"]
          .sum()
          .rename(columns={"item_quantity": "daily_demand"})
    )

    # Define T (number of periods)
    # Use the full date range across the dataset so frequency is comparable across SKUs
    min_day = pd.to_datetime(daily["day"]).min()
    max_day = pd.to_datetime(daily["day"]).max()
    all_days = pd.date_range(min_day, max_day, freq="D")
    T = len(all_days)

    print(f"Date range: {min_day.date()} to {max_day.date()}  => T={T} days")

    # Create a complete SKU x day grid (for correct mean/CV/frequency)
    skus = daily["item_code"].unique()
    full_index = pd.MultiIndex.from_product([skus, all_days.date], names=["item_code", "day"])

    daily_full = (
        daily.set_index(["item_code", "day"])
             .reindex(full_index, fill_value=0)
             .reset_index()
    )

    # Compute features per SKU
    g = daily_full.groupby("item_code")["daily_demand"]

    mean_demand = g.mean()
    std_demand = g.std(ddof=0)  # population std for stability
    nonzero_periods = g.apply(lambda x: (x > 0).sum())

    # CV handling: if mean == 0, CV undefined => set NaN
    cv = std_demand / mean_demand.replace(0, np.nan)

    demand_frequency = nonzero_periods / T

    features = pd.DataFrame({
        "item_code": mean_demand.index.astype(str),
        "mean_demand": mean_demand.values,
        "std_demand": std_demand.values,
        "cv_demand": cv.values,
        "nonzero_periods": nonzero_periods.values.astype(int),
        "T": T,
        "demand_frequency": demand_frequency.values,
    })

    out_csv = Path(f"demand_features_{tag}.csv")
    features.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv.resolve()}")

    # Quick preview
    print("\nPreview:")
    print(features.head(10).to_string(index=False))


def main():
    build_demand_features(Path("data_cleaned.csv"), tag="cleaned")
    build_demand_features(Path("data_cleaned_no_outlier.csv"), tag="no_outlier")


if __name__ == "__main__":
    main()