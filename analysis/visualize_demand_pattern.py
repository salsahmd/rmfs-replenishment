from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np


# ===== Setup Path =====
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "raw_order.csv"
OUTPUT_DIR = BASE_DIR / "output_sku"
OUTPUT_DIR.mkdir(exist_ok=True)

# ===== Load Data =====
df = pd.read_csv(INPUT_FILE, sep=";")

# Convert date
df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
df = df.dropna(subset=["order_date"])

# Create weekly aggregation
df["week"] = df["order_date"].dt.to_period("W")

weekly_sku = (
    df.groupby(["item_code", "week"])["item_quantity"]
    .sum()
    .reset_index()
)

# Convert week period to timestamp
weekly_sku["week"] = weekly_sku["week"].dt.to_timestamp()

# ===============================
# Select Top 5 SKUs by volume
# ===============================
top_skus = (
    weekly_sku.groupby("item_code")["item_quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
    .index
)

print("Top SKUs selected:", top_skus.tolist())


# ===============================
# Plot Trend per SKU
# ===============================
for sku in top_skus:
    sku_data = weekly_sku[weekly_sku["item_code"] == sku]
    sku_data = sku_data.set_index("week").asfreq("W").fillna(0)

    plt.figure()
    sku_data["item_quantity"].plot()
    plt.title(f"Weekly Demand - SKU {sku}")
    plt.xlabel("Week")
    plt.ylabel("Quantity")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"weekly_trend_{sku}.png")
    plt.close()


# ===============================
# Decomposition (Trend + Seasonality)
# ===============================
for sku in top_skus:
    sku_data = weekly_sku[weekly_sku["item_code"] == sku]
    sku_data = sku_data.set_index("week").asfreq("W").fillna(0)

    if len(sku_data) > 52:  # minimal 1 year data
        result = seasonal_decompose(
            sku_data["item_quantity"],
            model="additive",
            period=52
        )

        fig = result.plot()
        fig.set_size_inches(10, 6)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"decomposition_{sku}.png")
        plt.close(fig)

print("SKU-level trend & seasonality saved.")


feature_rows = []

for sku in top_skus:
    sku_data = weekly_sku[weekly_sku["item_code"] == sku]
    sku_data = sku_data.set_index("week").asfreq("W").fillna(0)

    if len(sku_data) > 52:
        result = seasonal_decompose(
            sku_data["item_quantity"],
            model="additive",
            period=52
        )

        # Trend slope (linear regression sederhana)
        y = sku_data["item_quantity"].values
        x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]

        # Seasonality strength
        var_total = np.var(result.observed)
        var_seasonal = np.var(result.seasonal)
        seasonality_strength = var_seasonal / var_total if var_total > 0 else 0

        feature_rows.append({
            "item_code": sku,
            "mean_demand": np.mean(y),
            "std_demand": np.std(y),
            "trend_slope": slope,
            "seasonality_strength": seasonality_strength
        })

# Save features
feature_df = pd.DataFrame(feature_rows)
feature_df.to_csv(OUTPUT_DIR / "sku_time_series_features.csv", index=False)

print("Feature extraction saved.")