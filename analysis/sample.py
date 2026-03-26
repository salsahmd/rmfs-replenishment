from pathlib import Path
import pandas as pd
from scipy.sparse import load_npz

# =========================
# 1. LOAD DATA
# =========================
raw_file = "raw_order.csv"
sku_file = "sku_sample.csv"
dict_file = "items_dictionary.csv"

df = pd.read_csv(raw_file, sep=";")
items_df = pd.read_csv(dict_file, sep=";", dtype=str)
items_df['item_code'] = items_df['item_code'].astype(str)

# =========================
# 2. FIX DATE FORMAT
# =========================
# raw_order.csv has two mixed formats:
#   '08/01/21 00.57'   → %m/%d/%y %H.%M  (rows 1–162k)
#   '8/21/2021 23:53'  → %m/%d/%Y %H:%M  (rows 162k–321k)
def parse_mixed_dates(series):
    formats = ['%m/%d/%y %H.%M', '%m/%d/%Y %H:%M']
    result = pd.Series([pd.NaT] * len(series), dtype='datetime64[ns]')
    remaining_mask = pd.Series([True] * len(series), index=series.index)
    for fmt in formats:
        if not remaining_mask.any():
            break
        parsed = pd.to_datetime(series[remaining_mask], format=fmt, errors='coerce')
        result[remaining_mask] = parsed.values
        remaining_mask = result.isna()
    return result

df['order_date'] = parse_mixed_dates(df['order_date'])
df = df.dropna(subset=['order_date'])
df['order_date'] = df['order_date'].dt.date

# =========================
# 3. DAILY DEMAND
# =========================
daily_demand = (
    df.groupby(['item_code', 'order_date'])['item_quantity']
    .sum()
    .reset_index()
)

# =========================
# 4. DATE RANGE
# =========================
all_dates = pd.date_range(
    start=min(daily_demand['order_date']),
    end=max(daily_demand['order_date'])
).date

T = len(all_dates)

# =========================
# 5. FEATURE CALCULATION
# =========================
features = []

for sku, group in daily_demand.groupby('item_code'):
    demand_map = dict(zip(group['order_date'], group['item_quantity']))
    demand_full = [demand_map.get(d, 0) for d in all_dates]
    demand_full = pd.Series(demand_full)

    mean_demand = demand_full.mean()
    std_demand = demand_full.std()

    cv = std_demand / mean_demand if mean_demand > 0 else 0
    nonzero = (demand_full > 0).sum()
    freq = nonzero / T

    features.append({
        'item_code': sku,
        'mean_demand': mean_demand,
        'std_demand': std_demand,
        'cv_demand': cv,
        'nonzero_periods': nonzero,
        'T': T,
        'demand_frequency': freq
    })

feature_df = pd.DataFrame(features)

# =========================
# 6. SELECT TOP 1000 BY MEAN DEMAND
# =========================
top_1000 = feature_df.sort_values(
    by='mean_demand', ascending=False
).head(1000)

# =========================
# 7. ADD MAX AFFINITY
# =========================
top_1000['item_code'] = top_1000['item_code'].astype(str)

affinity_dir = Path("clustering")
A_mat = load_npz(str(affinity_dir / "affinity_sparse.npz"))
sku_index = pd.read_csv(affinity_dir / "affinity_sku_index.csv")
sku_to_idx = {str(s): i for i, s in enumerate(sku_index["item_code"])}

def get_max_affinity(sku):
    idx = sku_to_idx.get(str(sku))
    if idx is None:
        return 0.0
    row = A_mat.getrow(idx)
    return float(row.max()) if row.nnz > 0 else 0.0

top_1000["max_affinity"] = top_1000["item_code"].apply(get_max_affinity)

# =========================
# 8. MERGE WITH ITEMS DICTIONARY
# =========================
final_df = pd.merge(top_1000, items_df, on='item_code', how='left')

# =========================
# 8. SAVE
# =========================
final_df.to_csv(sku_file, index=False)

print("✅ sku_sample.csv updated with new demand features (columns preserved!)")