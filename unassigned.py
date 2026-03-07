import pandas as pd
from pathlib import Path

# =========================
# File path
# =========================
INPUT_FILE = Path("unassigned_skus.csv")
OUTPUT_SUMMARY = Path("unassigned_summary.csv")
OUTPUT_DETAIL = Path("unassigned_detail_sorted.csv")

# =========================
# Load data
# =========================
df = pd.read_csv(INPUT_FILE)

# =========================
# Detect quantity column
# =========================
qty_col = None

if "unassigned_qty" in df.columns:
    qty_col = "unassigned_qty"
elif "unassigned_items_equivalent" in df.columns:
    qty_col = "unassigned_items_equivalent"
else:
    raise ValueError(
        "Tidak ditemukan kolom quantity unassigned. "
        "Kolom yang tersedia: "
        + ", ".join(df.columns)
    )

# Cek kolom wajib minimum
required_cols = {"item_code", "cluster"}
missing_cols = required_cols - set(df.columns)

if missing_cols:
    raise ValueError(f"Kolom berikut tidak ditemukan di unassigned_skus.csv: {missing_cols}")

# =========================
# 1. Total SKU unassigned
# =========================
total_unassigned_skus = df["item_code"].nunique()

# Kalau tahu total SKU semua, isi di sini
# total_sku_all = 7819
total_sku_all = None

if total_sku_all is not None:
    percent_unassigned = (total_unassigned_skus / total_sku_all) * 100
else:
    percent_unassigned = None

# =========================
# 2. Summary per cluster
# =========================
summary = df.groupby("cluster").agg(
    unassigned_sku_count=("item_code", "nunique"),
    total_unassigned_qty=(qty_col, "sum")
).reset_index()

# Tambahkan total row
total_row = pd.DataFrame({
    "cluster": ["TOTAL"],
    "unassigned_sku_count": [total_unassigned_skus],
    "total_unassigned_qty": [df[qty_col].sum()]
})

summary = pd.concat([summary, total_row], ignore_index=True)

# =========================
# 3. Sort detail
# =========================
detail_sorted = df.sort_values(
    by=["cluster", qty_col],
    ascending=[True, False]
).reset_index(drop=True)

# =========================
# 4. Save outputs
# =========================
summary.to_csv(OUTPUT_SUMMARY, index=False)
detail_sorted.to_csv(OUTPUT_DETAIL, index=False)

# =========================
# 5. Print results
# =========================
print("\n=== UNASSIGNED SKU SUMMARY ===")
print(f"Quantity column used: {qty_col}")
print(f"Total unique unassigned SKUs: {total_unassigned_skus}")

if percent_unassigned is not None:
    print(f"Percentage unassigned SKUs: {percent_unassigned:.2f}%")

print(f"Total unassigned quantity: {df[qty_col].sum()}")

if "unassigned_boxes" in df.columns:
    print(f"Total unassigned boxes: {df['unassigned_boxes'].sum()}")

print("\nUnassigned SKUs per cluster:")
print(summary)

print(f"\nSaved summary to: {OUTPUT_SUMMARY.resolve()}")
print(f"Saved sorted detail to: {OUTPUT_DETAIL.resolve()}")