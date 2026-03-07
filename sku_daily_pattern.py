from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Path setup
# =========================
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "raw_order.csv"
OUTPUT_DIR = BASE_DIR / "output_sku_daily"
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================
# Load & clean
# =========================
df = pd.read_csv(INPUT_FILE, sep=";")

# Pastikan tipe data konsisten
df["item_code"] = df["item_code"].astype(str)
df["item_quantity"] = pd.to_numeric(df["item_quantity"], errors="coerce").fillna(0)

# Parse tanggal (auto-detect)
df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

df = df.dropna(subset=["order_date"])

# Ambil tanggal saja (harian)
df["date"] = df["order_date"].dt.date
df["dow"] = df["order_date"].dt.day_name()
df["month"] = df["order_date"].dt.month


# =========================
# Daily aggregation per SKU
# =========================
daily_sku = (
    df.groupby(["item_code", "date"])["item_quantity"]
    .sum()
    .reset_index()
)

# Pilih top N SKU berdasarkan total quantity (biar fokus & cepat)
TOP_N = 5
top_skus = (
    daily_sku.groupby("item_code")["item_quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(TOP_N)
    .index
    .tolist()
)

print("Top SKUs selected:", top_skus)


# =========================
# Plot per SKU
# =========================
dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

for sku in top_skus:
    sku_daily = daily_sku[daily_sku["item_code"] == sku].copy()
    sku_daily["date"] = pd.to_datetime(sku_daily["date"])
    sku_daily = sku_daily.sort_values("date").set_index("date")

    # Isi tanggal yang hilang dengan 0 supaya grafik tidak putus-putus
    sku_daily = sku_daily.asfreq("D").fillna(0)

    # --- 1) Daily trend plot ---
    plt.figure()
    plt.plot(sku_daily.index, sku_daily["item_quantity"].values, linewidth=1)
    plt.title(f"Daily Demand Trend - SKU {sku}")
    plt.xlabel("Date")
    plt.ylabel("Quantity")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"daily_trend_{sku}.png")
    plt.close()

    # --- 2) Day-of-week seasonality (rata-rata demand per hari) ---
    tmp = df[df["item_code"] == sku].copy()
    dow_avg = tmp.groupby("dow")["item_quantity"].mean().reindex(dow_order)

    plt.figure()
    dow_avg.plot(kind="bar")
    plt.title(f"Day-of-Week Seasonality - SKU {sku}")
    plt.xlabel("Day of Week")
    plt.ylabel("Avg Quantity")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"dow_seasonality_{sku}.png")
    plt.close()

    # --- 3) Monthly seasonality (rata-rata demand per bulan) ---
    month_avg = tmp.groupby("month")["item_quantity"].mean().sort_index()

    plt.figure()
    month_avg.plot(kind="bar")
    plt.title(f"Monthly Seasonality - SKU {sku}")
    plt.xlabel("Month")
    plt.ylabel("Avg Quantity")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"monthly_seasonality_{sku}.png")
    plt.close()

    # --- 4) Save daily series as CSV (opsional) ---
    sku_daily.reset_index().to_csv(OUTPUT_DIR / f"daily_sku_{sku}.csv", index=False)

print("✅ Daily SKU plots saved to:", OUTPUT_DIR.resolve())