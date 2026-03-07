from pathlib import Path
import pandas as pd


def parse_order_date_mmddyy(series: pd.Series) -> pd.Series:
    """
    Parse order_date in mm/dd/yy format.
    Supports:
      - '08/01/21'
      - '08/01/21 00.57'  (hour.minute)
    """
    s = series.astype(str).str.strip()

    # Try with time first: mm/dd/yy HH.MM
    dt = pd.to_datetime(s, format="%m/%d/%y %H.%M", errors="coerce")

    # For rows without time or failed parsing, try date-only: mm/dd/yy
    mask = dt.isna()
    if mask.any():
        dt2 = pd.to_datetime(s[mask], format="%m/%d/%y", errors="coerce")
        dt.loc[mask] = dt2

    # Final fallback: auto-parse but explicitly month-first
    mask = dt.isna()
    if mask.any():
        dt3 = pd.to_datetime(s[mask], dayfirst=False, errors="coerce")
        dt.loc[mask] = dt3

    return dt


def main():
    BASE_DIR = Path(__file__).resolve().parent
    INPUT_FILE = BASE_DIR / "raw_order.csv"
    OUTPUT_FILE = BASE_DIR / "daily_qty_by_item.csv"

    # Load (file kamu pakai separator ;)
    df = pd.read_csv(INPUT_FILE, sep=";")

    # Pastikan tipe data aman
    df["item_code"] = df["item_code"].astype(str).str.strip()
    df["item_quantity"] = pd.to_numeric(df["item_quantity"], errors="coerce").fillna(0)

    # Parse date & ambil harinya saja
    df["order_datetime"] = parse_order_date_mmddyy(df["order_date"])
    bad = df["order_datetime"].isna().sum()
    if bad > 0:
        print(f"WARNING: {bad} rows failed date parsing and will be dropped.")
    df = df.dropna(subset=["order_datetime"])

    # Ambil day saja (tanpa jam) dan formatkan kembali mm/dd/yy
    df["order_day"] = df["order_datetime"].dt.strftime("%m/%d/%y")

    # Aggregate: total quantity per item_code per day
    daily = (
        df.groupby(["item_code", "order_day"], as_index=False)["item_quantity"]
        .sum()
        .rename(columns={"item_quantity": "daily_qty"})
        .sort_values(["item_code", "order_day"])
    )

    daily.to_csv(OUTPUT_FILE, index=False)

    print("Saved:", OUTPUT_FILE)
    print("Rows:", len(daily))
    print(daily.head(20))

    # Contoh cek satu SKU (ubah SKU di sini kalau mau)
    example_sku = "10000005001"
    ex = daily[daily["item_code"] == example_sku]
    if len(ex) > 0:
        print(f"\nExample output for SKU {example_sku}:")
        print(ex.head(20))
    else:
        print(f"\nSKU {example_sku} not found in the dataset.")


if __name__ == "__main__":
    main()