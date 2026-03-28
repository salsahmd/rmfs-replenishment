from pathlib import Path
import numpy as np
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
# 6. STRATIFIED SAMPLING BY IMPACT SCORE
# =========================
# parse number_of_item_in_a_box (may have double-period format e.g. '24.00.00')
def parse_double_period(val):
    s = str(val).strip()
    if s.count('.') == 2:
        s = s.rsplit('.', 1)[0]  # strip trailing '.00'
    try:
        return float(s)
    except ValueError:
        return float('nan')

# Merge all SKUs with items_df to get number_of_item_in_a_box
feature_df['item_code'] = feature_df['item_code'].astype(str)
merged_all = pd.merge(feature_df, items_df, on='item_code', how='inner')
merged_all['number_of_item_in_a_box'] = merged_all['number_of_item_in_a_box'].apply(parse_double_period)

# Compute initial inventory for all SKUs
Z = 1.28   # 90% service level
LT = 1     # lead time in days
merged_all['initial_unit_inventory'] = (
    merged_all['mean_demand'] * 4
).round(2)
merged_all['initial_box_inventory'] = np.ceil(
    merged_all['initial_unit_inventory'] / merged_all['number_of_item_in_a_box']
).astype(int)

# Impact score: min-max normalize 4 features, then take mean
impact_features = ['mean_demand', 'initial_box_inventory', 'cv_demand', 'demand_frequency']
for feat in impact_features:
    col_min = merged_all[feat].min()
    col_max = merged_all[feat].max()
    denom = col_max - col_min if col_max != col_min else 1
    merged_all[f'{feat}_norm'] = (merged_all[feat] - col_min) / denom

merged_all['impact_score'] = merged_all[[f'{feat}_norm' for feat in impact_features]].mean(axis=1)

# Assign quartile and stratified sample
merged_all['quartile'] = pd.qcut(merged_all['impact_score'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
strata_n = {'Q1': 50, 'Q2': 100, 'Q3': 200, 'Q4': 650}

sampled_parts = []
for q, n in strata_n.items():
    stratum = merged_all[merged_all['quartile'] == q]
    take = min(n, len(stratum))
    sampled_parts.append(stratum.sample(n=take, random_state=42))

top_1000 = pd.concat(sampled_parts).reset_index(drop=True)

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
# 8. FINALIZE
# =========================
final_df = top_1000.copy()

# Drop temporary columns used for sampling
temp_cols = [f'{feat}_norm' for feat in impact_features] + ['quartile', 'impact_score']
final_df = final_df.drop(columns=[c for c in temp_cols if c in final_df.columns])

# =========================
# 9. SAVE
# =========================
final_df.to_csv(sku_file, index=False)

print("✅ sku_sample.csv updated with stratified sampling by impact score (1000 SKUs)")
print(f"   Strata: Q1=50, Q2=100, Q3=200, Q4=650")
