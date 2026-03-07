import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe for terminal runs
import matplotlib.pyplot as plt
from pathlib import Path


def read_csv_auto_sep(path: Path) -> pd.DataFrame:
    """Try ';' then ',' separator."""
    df = pd.read_csv(path, sep=";")
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=",")
    return df


def main():
    # -----------------------------
    # Paths
    # -----------------------------
    RAW_ORDER_PATH = Path("raw_order.csv")
    ITEMS_DICT_PATH = Path("items_dictionary.csv")

    OUT_CLEANED_RAW = Path("data_cleaned.csv")
    OUT_CLEANED_DICT = Path("items_dictionary_cleaned.csv")
    OUT_OUTLIERS = Path("outlier_skus.csv")

    PLOT_HIST = Path("sku_order_volume_distribution.png")
    PLOT_BOXPLOT = Path("sku_order_volume_boxplot.png")

    # -----------------------------
    # Load
    # -----------------------------
    raw = read_csv_auto_sep(RAW_ORDER_PATH)
    items = read_csv_auto_sep(ITEMS_DICT_PATH)

    print("\n=== Loaded Files ===")
    print("raw_order shape:", raw.shape)
    print("items_dictionary shape:", items.shape)

    # -----------------------------
    # Column checks
    # -----------------------------
    needed_raw = {"order_id", "item_code", "item_quantity"}
    missing_raw = needed_raw - set(raw.columns)
    if missing_raw:
        raise ValueError(f"raw_order.csv missing columns: {missing_raw}. Found: {list(raw.columns)}")

    if "item_code" not in items.columns:
        raise ValueError(f"items_dictionary.csv must contain 'item_code'. Found: {list(items.columns)}")

    # -----------------------------
    # Standardize dtypes
    # -----------------------------
    raw["order_id"] = raw["order_id"].astype(str)
    raw["item_code"] = raw["item_code"].astype(str)
    items["item_code"] = items["item_code"].astype(str)

    raw["item_quantity"] = pd.to_numeric(raw["item_quantity"], errors="coerce")

    # -----------------------------
    # SKU Sets
    # -----------------------------
    skus_in_orders = set(raw["item_code"].dropna().unique())
    skus_in_dict = set(items["item_code"].dropna().unique())

    skus_no_orders = sorted(list(skus_in_dict - skus_in_orders))      # in dict but not ordered
    skus_not_in_dict = sorted(list(skus_in_orders - skus_in_dict))    # ordered but not in dict

    print("\n=== SKU Matching Check ===")
    print("Unique SKUs in raw_order:", len(skus_in_orders))
    print("Unique SKUs in items_dictionary:", len(skus_in_dict))
    print("SKUs in dictionary with NO orders:", len(skus_no_orders))
    print("SKUs in orders but NOT in dictionary:", len(skus_not_in_dict))

    # -----------------------------
    # 1) Clean RAW: keep only SKUs that exist in dictionary
    # -----------------------------
    raw_clean = raw[raw["item_code"].isin(skus_in_dict)].copy()

    # Remove missing qty rows
    raw_clean = raw_clean.dropna(subset=["item_quantity"])

    # Remove negative qty (optional; keep if you want returns)
    raw_clean = raw_clean[raw_clean["item_quantity"] >= 0].copy()

    # Save cleaned raw
    raw_clean.to_csv(OUT_CLEANED_RAW, index=False)
    print(f"\nSaved cleaned raw to: {OUT_CLEANED_RAW.resolve()}")
    print("raw_clean shape:", raw_clean.shape)

    # -----------------------------
    # A) Clean DICTIONARY: remove SKUs with no orders
    # -----------------------------
    skus_after_clean_raw = set(raw_clean["item_code"].unique())

    items_clean = items[items["item_code"].isin(skus_after_clean_raw)].copy()
    items_clean.to_csv(OUT_CLEANED_DICT, index=False)

    print(f"Saved cleaned dictionary to: {OUT_CLEANED_DICT.resolve()}")
    print("items_clean shape:", items_clean.shape)

    # -----------------------------
    # 2) SKU order volume distribution + mean (outlier detection)
    # -----------------------------
    # Total demand volume per SKU
    sku_volume = (
        raw_clean.groupby("item_code")["item_quantity"]
        .sum()
        .sort_values(ascending=False)
    )

    # -----------------------------
    # Remove SKUs with total quantity = 0
    # -----------------------------
    zero_volume_skus = sku_volume[sku_volume == 0].index.tolist()

    print("\n=== Removing SKUs with total quantity = 0 ===")
    print("Number of zero-volume SKUs:", len(zero_volume_skus))

    # Remove them from raw_clean
    raw_clean = raw_clean[~raw_clean["item_code"].isin(zero_volume_skus)].copy()

    # Recalculate sku_volume after removal
    sku_volume = (
        raw_clean.groupby("item_code")["item_quantity"]
        .sum()
        .sort_values(ascending=False)
    )

    print("Remaining SKUs after zero-volume removal:", raw_clean["item_code"].nunique())

    mean_volume = sku_volume.mean()

    # Histogram
    plt.figure()
    plt.hist(sku_volume.values, bins=80)
    plt.axvline(mean_volume, linestyle="--", linewidth=2, label=f"Mean = {mean_volume:.2f}")
    plt.xlabel("Total item_quantity per SKU")
    plt.ylabel("Number of SKUs")
    plt.title("SKU Order Volume Distribution (Total item_quantity)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_HIST, dpi=200)
    plt.close()

    # Boxplot
    plt.figure()
    plt.boxplot(sku_volume.values, vert=False)
    plt.xlabel("Total item_quantity per SKU")
    plt.title("SKU Order Volume Boxplot (Outlier Detection)")
    plt.tight_layout()
    plt.savefig(PLOT_BOXPLOT, dpi=200)
    plt.close()

    print("\nSaved plots:")
    print("-", PLOT_HIST.resolve())
    print("-", PLOT_BOXPLOT.resolve())

    # -----------------------------
    # 3) Descriptive statistics
    # -----------------------------
    print("\n=== Descriptive Statistics: raw_clean (numeric) ===")
    print(raw_clean.describe(include=[np.number]).T)

    print("\n=== Descriptive Statistics: raw_clean (all columns) ===")
    print(raw_clean.describe(include="all").T)

    print("\n=== Descriptive Statistics: SKU total volume ===")
    print(sku_volume.describe())

    # -----------------------------
    # Outliers: export ALL outlier SKUs (IQR rule)
    # -----------------------------
    q1 = sku_volume.quantile(0.25)
    q3 = sku_volume.quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr

    outlier_series = sku_volume[sku_volume > upper_bound].sort_values(ascending=False)

    outlier_df = outlier_series.reset_index()
    outlier_df.columns = ["item_code", "total_volume"]

    # Add helpful columns
    outlier_df["mean_volume_all_skus"] = mean_volume
    outlier_df["upper_bound_iqr"] = upper_bound
    outlier_df["volume_minus_upper_bound"] = outlier_df["total_volume"] - upper_bound

    outlier_df.to_csv(OUT_OUTLIERS, index=False)

    print("\n=== IQR Outlier Detection (SKU total volume) ===")
    print(f"Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}, UpperBound={upper_bound:.2f}")
    print("Number of outlier SKUs:", len(outlier_df))
    print(f"Saved ALL outlier SKUs to: {OUT_OUTLIERS.resolve()}")

    # Optional: also print first 20 outliers for quick view
    print("\nTop 20 outliers by volume:")
    print(outlier_df.head(20))

    # Extra reporting for mismatch SKUs
    if len(skus_no_orders) > 0:
        print("\nExample SKUs in dictionary with NO orders (first 20):")
        print(skus_no_orders[:20])

    if len(skus_not_in_dict) > 0:
        print("\nExample SKUs in orders but NOT in dictionary (first 20):")
        print(skus_not_in_dict[:20])

    # -----------------------------
    # 4) REMOVE ONLY EXTREME SKUs (MANUAL LIST)
    # -----------------------------
    print("\n=== Removing ONLY extreme SKUs (manual list) ===")

    EXTREME_SKUS = {"10021025001", "10000011001"}  # edit/add if needed

    # Save full order history of extreme SKUs
    order_outlier = raw_clean[raw_clean["item_code"].isin(EXTREME_SKUS)].copy()
    order_outlier.to_csv("order_outlier.csv", index=False)
    print(f"Saved extreme SKU order history to: {Path('order_outlier.csv').resolve()}")
    print("Extreme SKU order rows:", len(order_outlier))
    print("Extreme SKUs removed:", sorted(list(EXTREME_SKUS)))

    # Remove extreme SKUs from cleaned dataset
    raw_no_outlier = raw_clean[~raw_clean["item_code"].isin(EXTREME_SKUS)].copy()
    raw_no_outlier.to_csv("data_cleaned_no_outlier.csv", index=False)
    print(f"Saved filtered cleaned data (without extreme SKUs) to: {Path('data_cleaned_no_outlier.csv').resolve()}")
    print("Remaining rows after removing extreme SKUs:", len(raw_no_outlier))

    # -----------------------------
    # 5) SKU COUNT SUMMARY
    # -----------------------------
    print("\n=== SKU COUNT SUMMARY ===")

    total_sku_raw = raw["item_code"].nunique()
    total_sku_dict = items["item_code"].nunique()
    total_sku_clean = raw_clean["item_code"].nunique()
    total_sku_no_outlier = raw_no_outlier["item_code"].nunique()
    total_sku_extreme = len(EXTREME_SKUS)

    print("Total unique SKUs in raw_order:", total_sku_raw)
    print("Total unique SKUs in items_dictionary:", total_sku_dict)
    print("Total unique SKUs after cleaning:", total_sku_clean)
    print("Total extreme SKUs removed:", total_sku_extreme)
    print("Total unique SKUs after removing extreme:", total_sku_no_outlier)

if __name__ == "__main__":
    main()