"""
convert_to_pods.py
==================
Converts sku_assignment_detail.csv → pods.csv + items.csv
in the exact format expected by the netlogo.py simulation.

Inputs  (project root / analysis/)
------
  analysis/sku_assignment_detail.csv  — assignment result (one row per slot)
  sku_sample.csv                      — demand stats, box dims, cluster, inventory

Outputs (project root)
-------
  pods.csv   — consumed by assign_skus_to_pods_from_file() in netlogo.py
  items.csv  — consumed by order_generator.py (item_class = cluster int 0/1/2/3)

Key decisions
-------------
  item_id    : sequential integer 0..N-1, sorted by cluster then item_code
  item_class : cluster integer (0, 1, 2, 3) — matches clustering_kmeans.py output
               NOTE: netlogo.py items_orders_class_configuration must also use
               integer keys {0: ..., 1: ..., 2: ..., 3: ...}

  Inventory levels by cluster:
    cluster 0 (slow/erratic) : pod=0.6  warehouse=0.5
    cluster 1 (regular)      : pod=0.5  warehouse=0.4
    cluster 2 (top fast)     : pod=0.4  warehouse=0.3
    cluster 3 (high-vol)     : pod=0.4  warehouse=0.3
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE         = Path(__file__).parent
PROJECT_ROOT = BASE.parent

DETAIL_FILE = BASE / "sku_assignment_detail.csv"
SKU_SAMPLE  = PROJECT_ROOT / "sku_sample.csv"
OUT_PODS    = PROJECT_ROOT / "pods.csv"
OUT_ITEMS   = PROJECT_ROOT / "items.csv"
OUT_MAPPING = BASE / "item_id_mapping.csv"

POD_INV_LEVEL = {0: 0.6, 1: 0.5, 2: 0.4, 3: 0.4}
WH_INV_LEVEL  = {0: 0.5, 1: 0.4, 2: 0.3, 3: 0.3}


def main():
    print("=" * 60)
    print("convert_to_pods  –  assignment → pods.csv + items.csv")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load inputs
    # ------------------------------------------------------------------
    detail = pd.read_csv(DETAIL_FILE)
    detail["item_code"] = detail["item_code"].astype(str)

    sku = pd.read_csv(SKU_SAMPLE)
    sku["item_code"] = sku["item_code"].astype(str)

    print(f"  Assignment rows : {len(detail)}  "
          f"({detail['pod_id'].nunique()} pods, "
          f"{detail['item_code'].nunique()} SKUs)")

    # ------------------------------------------------------------------
    # 2. Build item_code → item_id mapping
    #    Sorted by cluster priority order then item_code for reproducibility.
    # ------------------------------------------------------------------
    cluster_order = {0: 0, 2: 1, 3: 2, 1: 3}
    assigned_codes = sorted(detail["item_code"].unique())
    sku_assigned = sku[sku["item_code"].isin(assigned_codes)].copy()
    sku_assigned["_co"] = sku_assigned["cluster"].map(cluster_order)
    sku_assigned = (
        sku_assigned.sort_values(["_co", "item_code"])
        .drop(columns=["_co"])
        .reset_index(drop=True)
    )
    sku_assigned["item_id"] = sku_assigned.index   # 0..N-1

    mapping = sku_assigned[["item_id", "item_code", "cluster"]].copy()
    mapping.to_csv(OUT_MAPPING, index=False)
    code_to_id = dict(zip(mapping["item_code"], mapping["item_id"]))
    print(f"  item_id mapping : {len(mapping)} SKUs  →  {OUT_MAPPING.name}")

    # ------------------------------------------------------------------
    # 3. Build items.csv
    #    item_class = cluster integer (0, 1, 2, 3)
    #    item_order_frequency from nonzero_periods (demand frequency proxy)
    # ------------------------------------------------------------------
    items_df = sku_assigned.copy()
    items_df["item_class"]  = items_df["cluster"].astype(int)   # <-- cluster as class
    items_df["item_weight"] = (
        items_df["box_weight"] / items_df["number_of_item_in_a_box"]
    ).round(3)
    items_df["item_pod_inventory_level"]       = items_df["cluster"].map(POD_INV_LEVEL)
    items_df["item_warehouse_inventory_level"] = items_df["cluster"].map(WH_INV_LEVEL)
    items_df["item_order_frequency"] = (
        items_df.get("nonzero_periods", items_df["demand_frequency"] * 100)
        .fillna(0).astype(int)
    )

    # use initial_unit_inventory as the initial qty for items.csv
    if "initial_unit_inventory" in items_df.columns:
        items_df["item_initial_quantity_inventory"] = (
            items_df["initial_unit_inventory"].fillna(0).round(0).astype(int)
        )

    items_out = items_df[[
        "item_id", "item_code", "item_class", "item_order_frequency",
        "item_initial_quantity_inventory",
        "box_length", "box_width", "box_height", "box_volume",
        "box_weight", "number_of_item_in_a_box",
        "item_volume", "item_weight", "item_unit",
        "item_pod_inventory_level", "item_warehouse_inventory_level",
    ]].copy()

    items_out.to_csv(OUT_ITEMS, index=False)
    print(f"  items.csv       : {len(items_out)} rows  →  {OUT_ITEMS}")
    print(f"    item_class distribution (= cluster):")
    for cl in sorted(items_out["item_class"].unique()):
        n = (items_out["item_class"] == cl).sum()
        print(f"      cluster {cl}: {n} SKUs")

    # ------------------------------------------------------------------
    # 4. Merge item_id + sku metadata into detail
    # ------------------------------------------------------------------
    # drop columns already in detail that we'll get from sku
    detail = detail.drop(columns=["box_weight", "cluster"], errors="ignore")
    detail = detail.merge(
        sku[["item_code", "cluster", "box_weight",
             "number_of_item_in_a_box", "initial_box_inventory"]],
        on="item_code", how="left"
    )
    detail["item_id"]       = detail["item_code"].map(code_to_id)
    detail["units_per_box"] = detail["number_of_item_in_a_box"].fillna(1).clip(lower=1)
    detail["item_weight"]   = (detail["box_weight"] / detail["units_per_box"]).round(3)
    detail["item_pod_inv_level"] = detail["cluster"].map(POD_INV_LEVEL)
    detail["item_wh_inv_level"]  = detail["cluster"].map(WH_INV_LEVEL)
    detail["max_qty"] = (detail["max_boxes_in_slot"] * detail["units_per_box"]).astype(int)
    detail["qty"]     = detail["qty_items_assigned"].astype(int)
    detail["total_item_weight"] = (detail["item_weight"] * detail["qty"]).round(3)

    # ------------------------------------------------------------------
    # 5. Build pods.csv
    # ------------------------------------------------------------------
    pods_df = pd.DataFrame({
        "pod_id":                         detail["pod_id"].astype(int),
        "pod_type":                       0,
        "slot_id":                        detail["slot_idx"].astype(int),
        "slot_type":                      detail["slot_type"].astype(int),
        "item":                           detail["item_id"].astype(int),
        "unusedColumn1":                  0,
        "unusedColumn2":                  0,
        "unusedColumn3":                  0,
        "qty":                            detail["qty"],
        "max_qty":                        detail["max_qty"],
        "due_date":                       99999,
        "facing":                         0,
        "pick_ind":                       0,
        "slot_sequence":                  detail["slot_idx"].astype(int),
        "item_weight":                    detail["item_weight"],
        "total_item_weight":              detail["total_item_weight"],
        "item_pod_inventory_level":       detail["item_pod_inv_level"],
        "item_warehouse_inventory_level": detail["item_wh_inv_level"],
    })

    pods_df = pods_df.sort_values(["pod_id", "slot_id"]).reset_index(drop=True)
    pods_df.to_csv(OUT_PODS, index=False)
    print(f"  pods.csv        : {len(pods_df)} rows, "
          f"{pods_df['pod_id'].nunique()} pods  →  {OUT_PODS}")

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print(f"\n  {'cluster':>7}  {'skus':>5}  {'slots':>6}  "
          f"{'qty_placed':>10}  {'pod_inv':>7}  {'wh_inv':>6}")
    for cl in [0, 2, 3, 1]:
        ids = set(mapping[mapping["cluster"] == cl]["item_id"])
        sub = pods_df[pods_df["item"].isin(ids)]
        print(f"  {cl:>7}  {len(ids):>5}  {len(sub):>6}  "
              f"{sub['qty'].sum():>10,}  "
              f"{POD_INV_LEVEL[cl]:>7.1f}  {WH_INV_LEVEL[cl]:>6.1f}")

    print(f"\n  NOTE: Update netlogo.py  items_orders_class_configuration  to:")
    print(f"    {{0: 0.10, 1: 0.30, 2: 0.30, 3: 0.30}}")
    print(f"  (cluster 0 = slow, gets fewer orders;")
    print(f"   clusters 1/2/3 = regular/fast, get more orders)")
    print(f"\nDone. You can now run: python3 profile_netlogo.py")


if __name__ == "__main__":
    main()
