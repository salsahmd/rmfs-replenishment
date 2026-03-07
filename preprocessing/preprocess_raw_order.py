from pathlib import Path
import os

import pandas as pd
import numpy as np


# Folder lokasi file .py ini (bukan folder terminal)
BASE_DIR = Path(__file__).resolve().parent
RAW_FILE = BASE_DIR / "raw_order.csv"


def load_data(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep=";")
    print("Data loaded successfully.")
    print("Shape:", df.shape)
    print("Columns:", df.columns)
    return df


def check_missing_values(df: pd.DataFrame) -> None:
    print("\n=== Missing Values ===")
    missing = df.isnull().sum()
    print(missing)
    print("\nPercentage missing:")
    print((missing / len(df)) * 100)


def check_duplicates(df: pd.DataFrame) -> None:
    duplicates = df.duplicated().sum()
    print("\n=== Duplicate Rows ===")
    print("Number of duplicate rows:", duplicates)


def convert_date(df: pd.DataFrame) -> pd.DataFrame:
    # Sesuaikan format ini jika format tanggal kamu berbeda
    print("\n=== Converting order_date ===")
    if "order_date" not in df.columns:
        print("Column 'order_date' not found. Skipping date conversion.")
        return df

    df["order_date"] = pd.to_datetime(
        df["order_date"],
        format="%m/%d/%y %H.%M",
        errors="coerce",
    )
    print("Conversion done.")
    print("Missing dates after conversion:", df["order_date"].isnull().sum())
    return df


def basket_analysis(df: pd.DataFrame) -> None:
    print("\n=== Basket Size Analysis ===")
    if not {"order_id", "item_code"}.issubset(df.columns):
        print("Missing order_id or item_code. Skipping basket analysis.")
        return
    basket_sizes = df.groupby("order_id")["item_code"].nunique()
    print(basket_sizes.describe())
    print("\nTop basket sizes:")
    print(basket_sizes.value_counts().head(10))


def sku_analysis(df: pd.DataFrame) -> None:
    print("\n=== SKU Frequency Analysis ===")
    if "item_code" not in df.columns:
        print("Missing item_code. Skipping SKU analysis.")
        return
    sku_freq = df["item_code"].value_counts()
    print("Number of unique SKUs:", df["item_code"].nunique())
    print("\nTop 10 most ordered SKUs:")
    print(sku_freq.head(10))


def quantity_analysis(df: pd.DataFrame) -> None:
    print("\n=== Quantity Analysis ===")
    if "item_quantity" not in df.columns:
        print("Missing item_quantity. Skipping quantity analysis.")
        return
    print(df["item_quantity"].describe())
    print("\nUnique quantity values (first 20):")
    uniq = df["item_quantity"].dropna().unique()
    uniq_sorted = np.sort(uniq)[:20]
    print(uniq_sorted)


def main():
    df = load_data(RAW_FILE)
    check_missing_values(df)
    check_duplicates(df)
    df = convert_date(df)
    basket_analysis(df)
    sku_analysis(df)
    quantity_analysis(df)


if __name__ == "__main__":
    main()