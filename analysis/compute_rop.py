"""
compute_rop.py
==============
Computes Reorder Point (ROP) per SKU and per pod for SKUs assigned to pods
in the simulation.

Inputs (project root / analysis/)
------
  pods.csv                       — simulation pods (item column = item_id)
  analysis/item_id_mapping.csv   — item_id ↔ item_code mapping
  sku_sample.csv                 — demand statistics (mean_demand, std_demand)
  analysis/sku_assignment_detail.csv — pod_id per item_code (to count pods)

Formula
-------
  lead_time    = 0.125
  Z            = 1.645  (95% service level, all clusters)
  safety_stock = Z x std_demand x sqrt(lead_time)
  rop_global   = mean_demand x lead_time + safety_stock
  rop_per_pod  = rop_global / num_pods

Output
------
  analysis/rop_per_sku.csv
"""

import math
import pandas as pd
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
BASE         = Path(__file__).parent
PROJECT_ROOT = BASE.parent

PODS_FILE    = PROJECT_ROOT / "pods.csv"
MAPPING_FILE = BASE / "item_id_mapping.csv"
SKU_SAMPLE   = PROJECT_ROOT / "sku_sample.csv"
DETAIL_FILE  = BASE / "sku_assignment_detail.csv"
OUT_FILE     = BASE / "rop_per_sku.csv"

# =============================================================================
# PARAMETERS
# =============================================================================
LEAD_TIME     = 0.125
Z_95          = 1.645   # 95% service level


def main():
    print("=" * 60)
    print("compute_rop  –  ROP global and ROP per pod per SKU")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Get item_codes used in simulation via pods.csv → item_id_mapping
    # ------------------------------------------------------------------
    pods_df    = pd.read_csv(PODS_FILE)
    mapping_df = pd.read_csv(MAPPING_FILE)
    mapping_df["item_code"] = mapping_df["item_code"].astype(str)

    sim_item_ids   = pods_df["item"].unique()
    sim_mapping    = mapping_df[mapping_df["item_id"].isin(sim_item_ids)]
    sim_item_codes = set(sim_mapping["item_code"])

    print(f"\n  SKUs in simulation (pods.csv): {len(sim_item_codes)}")

    # ------------------------------------------------------------------
    # 2. Load demand stats from sku_sample.csv, filter to sim SKUs
    # ------------------------------------------------------------------
    sku_df = pd.read_csv(SKU_SAMPLE)
    sku_df["item_code"] = sku_df["item_code"].astype(str)
    sku_df = sku_df[sku_df["item_code"].isin(sim_item_codes)].copy()

    print(f"  SKUs matched in sku_sample.csv: {len(sku_df)}")

    # ------------------------------------------------------------------
    # 3. Count number of pods per SKU from sku_assignment_detail.csv
    # ------------------------------------------------------------------
    detail_df = pd.read_csv(DETAIL_FILE)
    detail_df["item_code"] = detail_df["item_code"].astype(str)
    detail_df = detail_df[detail_df["item_code"].isin(sim_item_codes)]

    pods_per_sku = (
        detail_df.groupby("item_code")["pod_id"]
        .nunique()
        .reset_index()
        .rename(columns={"pod_id": "num_pods"})
    )

    # Total max_qty across all pods per SKU (S total) from pods.csv
    max_qty_per_sku = (
        pods_df.merge(sim_mapping[["item_id", "item_code"]], left_on="item", right_on="item_id", how="inner")
        .groupby("item_code")["max_qty"]
        .sum()
        .reset_index()
        .rename(columns={"max_qty": "s_total"})
    )

    # ------------------------------------------------------------------
    # 4. Merge and compute ROP
    # ------------------------------------------------------------------
    result = sku_df[["item_code", "cluster", "mean_demand", "std_demand"]].copy()
    result = result.merge(pods_per_sku, on="item_code", how="left")
    result = result.merge(max_qty_per_sku, on="item_code", how="left")
    result["num_pods"] = result["num_pods"].fillna(1).astype(int)
    result["s_total"]  = result["s_total"].fillna(0).astype(int)

    result["lead_time"]     = LEAD_TIME
    result["safety_stock"]  = Z_95 * result["std_demand"] * math.sqrt(LEAD_TIME)
    result["rop_global"]    = (result["mean_demand"] * LEAD_TIME + result["safety_stock"]).apply(math.ceil)
    result["rop_per_pod"]   = (result["rop_global"] / result["num_pods"]).apply(math.ceil)

    # ------------------------------------------------------------------
    # 5. Save rop_per_sku.csv
    # ------------------------------------------------------------------
    out_cols = [
        "item_code", "cluster", "mean_demand", "std_demand",
        "lead_time", "safety_stock", "rop_global", "num_pods", "rop_per_pod", "s_total"
    ]
    result[out_cols].to_csv(OUT_FILE, index=False)

    print(f"\n  Output saved → {OUT_FILE}")

    # ------------------------------------------------------------------
    # 6. Merge rop_global and rop_per_pod into pods.csv
    # ------------------------------------------------------------------
    # Drop existing ROP columns to avoid duplicates on re-run
    pods_df = pods_df.drop(columns=[c for c in ["rop_global", "rop_per_pod"] if c in pods_df.columns])

    rop_cols = result[["item_code", "rop_global", "rop_per_pod"]]
    pods_df = pods_df.merge(
        sim_mapping[["item_id", "item_code"]], left_on="item", right_on="item_id", how="left"
    )
    pods_df = pods_df.merge(rop_cols, on="item_code", how="left")
    pods_df = pods_df.drop(columns=["item_id", "item_code"])

    pods_df.to_csv(PODS_FILE, index=False)
    print(f"  pods.csv updated with rop_global and rop_per_pod → {PODS_FILE}")
    print(f"\n  Summary:")
    print(f"    Total SKUs      : {len(result)}")
    print(f"    Lead time       : {LEAD_TIME}")
    print(f"    Z (95% SL)      : {Z_95}")
    print(f"\n  ROP global stats:")
    print(f"    min  : {result['rop_global'].min():.4f}")
    print(f"    mean : {result['rop_global'].mean():.4f}")
    print(f"    max  : {result['rop_global'].max():.4f}")
    print(f"\n  ROP per pod stats:")
    print(f"    min  : {result['rop_per_pod'].min():.4f}")
    print(f"    mean : {result['rop_per_pod'].mean():.4f}")
    print(f"    max  : {result['rop_per_pod'].max():.4f}")
    print(f"\n  Preview (top 5):")
    print(result[out_cols].head(5).to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
