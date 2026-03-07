from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    BASE_DIR = Path(__file__).resolve().parent

    # Input hasil agregasi harian
    INPUT_FILE = BASE_DIR / "daily_qty_by_item.csv"

    # Folder output gambar
    OUT_DIR = BASE_DIR / "plots_sku"
    OUT_DIR.mkdir(exist_ok=True)

    # Load
    daily = pd.read_csv(INPUT_FILE)

    # Pastikan tipe data benar
    daily["item_code"] = daily["item_code"].astype(str).str.strip()
    daily["daily_qty"] = pd.to_numeric(daily["daily_qty"], errors="coerce").fillna(0)

    # Parse tanggal (format mm/dd/yy)
    daily["order_day"] = pd.to_datetime(daily["order_day"], format="%m/%d/%y", errors="coerce")
    daily = daily.dropna(subset=["order_day"])

    # Pilih SKU untuk diplot:
    # Total demand per SKU
    total_per_sku = (
        daily.groupby("item_code")["daily_qty"]
        .sum()
        .sort_values(ascending=False)
    )

    # Top 10
    top_10 = total_per_sku.head(10).index.tolist()

    # Bottom 10 (exclude zero demand)
    bottom_10 = (
        total_per_sku[total_per_sku > 0]
        .sort_values(ascending=True)
        .head(10)
        .index
        .tolist()
    )

    print("Top 10 SKUs:", top_10)
    print("Bottom 10 SKUs:", bottom_10)

    # Gabungkan list
    selected_skus = top_10 + bottom_10

    # Plot 1: Trend harian per SKU (disimpan png)
    for sku in selected_skus:
        sku_df = daily[daily["item_code"] == sku].copy()
        sku_df = sku_df.sort_values("order_day").set_index("order_day")

        # Isi tanggal yang hilang jadi 0 supaya grafik tidak putus-putus
        sku_df = sku_df.asfreq("D").fillna(0)

        plt.figure()
        plt.plot(sku_df.index, sku_df["daily_qty"].values, linewidth=1)
        plt.title(f"Daily Demand Trend - SKU {sku}")
        plt.xlabel("Date")
        plt.ylabel("Daily Quantity")
        plt.xticks(rotation=45)
        plt.tight_layout()
        label = "TOP" if sku in top_10 else "BOTTOM"
        plt.savefig(OUT_DIR / f"{label}_trend_{sku}.png")
        plt.close()

        # Plot 2 (opsional): Pola demand per day-of-week
        sku_df2 = sku_df.copy()
        sku_df2["dow"] = sku_df2.index.day_name()
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow_avg = sku_df2.groupby("dow")["daily_qty"].mean().reindex(dow_order)

        plt.figure()
        dow_avg.plot(kind="bar")
        plt.title(f"Day-of-Week Seasonality - SKU {sku}")
        plt.xlabel("Day of Week")
        plt.ylabel("Avg Daily Quantity")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"seasonality_dow_{sku}.png")
        plt.close()

    print("Plots saved in:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()