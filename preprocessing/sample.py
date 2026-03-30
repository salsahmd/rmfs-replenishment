import pandas as pd

# ── 1. Load data ──────────────────────────────────────────────────────────────
sku_dict = pd.read_csv("sku_dictionary.csv")
slots_cfg = pd.read_csv("items_slots_configuration.csv", sep=";")

# ── 1b. Recompute item_order_frequency from raw orders ────────────────────────
# Algorithm:
#   For each SKU, count how many distinct order_ids it appeared in.
#   This reflects how frequently an item was actually ordered (order-level),
#   as opposed to demand_frequency which measures on which days demand was nonzero.
#
#   Steps:
#   1. Load raw order lines (one row = one SKU in one order)
#   2. Group by item_code, count unique order_ids per SKU
#   3. Merge the result back into sku_dict, overwriting item_order_frequency
raw_orders = pd.read_csv("raw_order.csv", sep=";", usecols=["order_id", "item_code"])
raw_orders["item_code"] = raw_orders["item_code"].astype(str)
order_freq = (
    raw_orders.groupby("item_code")["order_id"]
    .nunique()
    .reset_index()
    .rename(columns={"order_id": "item_order_frequency"})
)
sku_dict["item_code"] = sku_dict["item_code"].astype(str)
sku_dict = sku_dict.drop(columns=["item_order_frequency"], errors="ignore")
sku_dict = sku_dict.merge(order_freq, on="item_code", how="left")
sku_dict["item_order_frequency"] = sku_dict["item_order_frequency"].fillna(0).astype(int)
print(f"item_order_frequency recomputed from raw_order.csv")

# ── 2. Feasibility filter ─────────────────────────────────────────────────────
# Feasible = item_code appears at least once in items_slots_configuration
feasible_codes = slots_cfg["item_code"].unique()
unfeasible_mask = ~sku_dict["item_code"].isin(feasible_codes)

print(f"Total SKUs          : {len(sku_dict)}")
print(f"Unfeasible SKUs     : {unfeasible_mask.sum()}")
print(f"Feasible SKUs       : {(~unfeasible_mask).sum()}")

sku_feasible = sku_dict[~unfeasible_mask].copy().reset_index(drop=True)

# ── 3. Impact score ───────────────────────────────────────────────────────────
# Normalise each driver to [0, 1] then take equal-weight average.
# Higher mean_demand, demand_frequency, avg_affinity, max_affinity → more impact
# Higher cv_demand → more variability / harder to manage → more impact
features = ["mean_demand", "cv_demand", "demand_frequency", "avg_affinity", "max_affinity"]

for f in features:
    col_min = sku_feasible[f].min()
    col_max = sku_feasible[f].max()
    denom = col_max - col_min if col_max != col_min else 1.0
    sku_feasible[f"{f}_norm"] = (sku_feasible[f] - col_min) / denom

norm_cols = [f"{f}_norm" for f in features]
sku_feasible["impact_score"] = sku_feasible[norm_cols].mean(axis=1)

# Drop the temporary normalised columns
sku_feasible.drop(columns=norm_cols, inplace=True)

# ── 4. Stratified sampling per cluster ───────────────────────────────────────
QUARTILE_FRACTIONS = {4: 0.40, 3: 0.30, 2: 0.20, 1: 0.10}

sampled_parts = []

for cluster_id, group in sku_feasible.groupby("cluster"):
    group = group.copy()

    # Assign quartile labels (1 = lowest impact, 4 = highest impact)
    group["quartile"] = pd.qcut(
        group["impact_score"],
        q=4,
        labels=[1, 2, 3, 4],
        duplicates="drop",
    ).astype(int)

    cluster_samples = []
    for q, frac in QUARTILE_FRACTIONS.items():
        q_group = group[group["quartile"] == q]
        n = max(1, round(len(q_group) * frac)) if len(q_group) > 0 else 0
        n = min(n, len(q_group))
        if n > 0:
            cluster_samples.append(q_group.sample(n=n, random_state=42))

    if cluster_samples:
        sampled_parts.append(pd.concat(cluster_samples))

    print(
        f"Cluster {cluster_id:>2} | total={len(group):>4} | "
        + " | ".join(
            f"Q{q}={len(group[group['quartile']==q]):>3}"
            for q in [1, 2, 3, 4]
        )
    )

sku_sample = pd.concat(sampled_parts).reset_index(drop=True)

# ── 5. Keep only original sku_dictionary columns + impact_score ───────────────
original_cols = list(sku_dict.columns)
sku_sample = sku_sample[original_cols + ["impact_score", "quartile"]]

# ── 6. Save ───────────────────────────────────────────────────────────────────
sku_sample.to_csv("sku_sample.csv", index=False)

print(f"\nSampled SKUs saved  : {len(sku_sample)}")
print(f"Quartile distribution:\n{sku_sample['quartile'].value_counts().sort_index()}")
print("\nsku_sample.csv written successfully.")
