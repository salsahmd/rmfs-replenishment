"""
Compute ROP (Reorder Point) per slot for each SKU.

Formula:
    rop_global   = ceil(mean_demand * L + z * std_demand * sqrt(L))
    rop_per_slot = ceil(rop_global / n_slots)

where:
    L       = 1/8 day  (lead time = 1 hour out of 8-hour working day)
    n_slots = total number of slots storing that SKU across ALL pods (from pods.csv)
    z       = service-level z-score per cluster

Parameters (adjustable for sensitivity analysis):
    L              — lead time in days
    Z_BY_CLUSTER   — z-score per cluster (service level differentiation)

Outputs sku_rop.csv with columns:
    item_code, rop_global, rop_per_slot, n_slots, S_total

Usage:
    python3 analysis/compute_rop.py
    python3 analysis/compute_rop.py --lead-time 0.25
"""

import argparse
import math
import os
import sys
from pathlib import Path

import pandas as pd

# ── Parameters ──────────────────────────────────────────────────────────────

L = 1 / 8  # lead time = 1 hour / 8-hour working day

Z_BY_CLUSTER = {
    0: 1.65,   # ~95% service level
    1: 1.65,   # ~95% service level
    2: 1.65,   # ~95% service level
    3: 1.65,   # ~95% service level
}
Z_DEFAULT = 1.65  

# ── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent

SKU_SAMPLE_PATH = REPO_ROOT / "sku_sample.csv"
PODS_CSV_PATH = REPO_ROOT / "netlogo" / "pods.csv"
OUTPUT_PATH = REPO_ROOT / "sku_rop.csv"


# ── Core computation ─────────────────────────────────────────────────────────

def compute_rop(sku_sample_path: Path, pods_csv_path: Path,
                lead_time: float = L) -> pd.DataFrame:
    """
    Compute rop_global and rop_per_slot for each assigned SKU.

    Returns DataFrame with columns:
        item_code, rop_global, rop_per_slot, n_slots, S_total
    """
    # Load sku_sample
    sku_df = pd.read_csv(sku_sample_path, sep=None, engine="python")
    sku_df.columns = sku_df.columns.str.strip().str.lower().str.replace(" ", "_")
    sku_df["item_code"] = sku_df["item_code"].astype(str).str.strip()

    # Resolve mean_demand column
    mean_col = next(
        (c for c in ["mean_daily_demand", "mean_demand", "average_daily_demand",
                     "avg_daily_demand", "demand_mean"]
         if c in sku_df.columns), None
    )
    if mean_col is None:
        raise ValueError(f"Cannot find mean demand column in {sku_sample_path}. "
                         f"Columns present: {list(sku_df.columns)}")

    # Resolve std_demand column (derive from cv × mean if needed)
    std_col = next(
        (c for c in ["std_daily_demand", "std_demand", "demand_std", "stddev_demand",
                     "standard_deviation_demand"]
         if c in sku_df.columns), None
    )
    if std_col is None and "cv_demand" in sku_df.columns:
        sku_df["_std_demand"] = sku_df["cv_demand"] * sku_df[mean_col]
        std_col = "_std_demand"
    if std_col is None:
        raise ValueError(f"Cannot find std demand column in {sku_sample_path}. "
                         f"Columns present: {list(sku_df.columns)}")

    # Resolve cluster column
    cluster_col = next(
        (c for c in ["cluster", "item_class", "sku_cluster"] if c in sku_df.columns), None
    )
    if cluster_col is None:
        raise ValueError(f"Cannot find cluster column in {sku_sample_path}. "
                         f"Columns present: {list(sku_df.columns)}")

    sku_df["mean_demand"] = pd.to_numeric(sku_df[mean_col], errors="coerce").fillna(0.0)
    sku_df["std_demand"] = pd.to_numeric(sku_df[std_col], errors="coerce").fillna(0.0)
    sku_df["cluster"] = pd.to_numeric(sku_df[cluster_col], errors="coerce").fillna(0).astype(int)

    # z-score per cluster
    sku_df["z"] = sku_df["cluster"].map(Z_BY_CLUSTER).fillna(Z_DEFAULT)

    # rop_global = ceil(mean * L + z * std * sqrt(L))
    sku_df["rop_global"] = (
        sku_df["mean_demand"] * lead_time
        + sku_df["z"] * sku_df["std_demand"] * math.sqrt(lead_time)
    ).apply(math.ceil).clip(lower=1)

    # Load pods.csv
    pods_df = pd.read_csv(pods_csv_path)
    pods_df.columns = pods_df.columns.str.strip().str.lower()

    # pods.csv already has item_code column (from convert_to_sim.py)
    if "item_code" not in pods_df.columns:
        # fallback: map item_id → item_code via sorted sku_df
        item_col = "item" if "item" in pods_df.columns else "item_id"
        sorted_skus = sku_df.sort_values("item_code").reset_index(drop=True)
        sorted_skus["item_id"] = sorted_skus.index
        id_to_code = sorted_skus.set_index("item_id")["item_code"].to_dict()
        pods_df["item_code"] = pods_df[item_col].map(id_to_code).astype(str)

    pods_df["item_code"] = pods_df["item_code"].astype(str).str.strip()

    # n_slots: total number of slots (rows) for each item_code across all pods
    n_slots_series = pods_df.groupby("item_code").size().rename("n_slots")

    # S_total: sum of max_qty per item_code across all slots
    qty_col = "max_qty" if "max_qty" in pods_df.columns else "qty"
    s_total_series = pods_df.groupby("item_code")[qty_col].sum().rename("S_total")

    # Inner join — keeps only assigned SKUs (matches pods.csv)
    result = sku_df[["item_code", "cluster", "mean_demand", "std_demand",
                      "z", "rop_global"]].copy()
    result = result.merge(n_slots_series, on="item_code", how="inner")
    result = result.merge(s_total_series, on="item_code", how="left")
    result["n_slots"] = result["n_slots"].fillna(1).astype(int)
    result["S_total"] = result["S_total"].fillna(0).astype(int)

    # rop_per_slot = ceil(rop_global / n_slots), minimum 1
    result["rop_per_slot"] = (
        result["rop_global"] / result["n_slots"]
    ).apply(math.ceil).clip(lower=1).astype(int)

    return result[["item_code", "rop_global", "rop_per_slot", "n_slots", "S_total"]]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compute ROP per slot for each SKU")
    parser.add_argument("--lead-time", type=float, default=L,
                        help=f"Lead time in days (default: {L} = 1/8 day = 1 hour)")
    parser.add_argument("--sku-sample", type=str, default=str(SKU_SAMPLE_PATH),
                        help="Path to sku_sample.csv")
    parser.add_argument("--pods-csv", type=str, default=str(PODS_CSV_PATH),
                        help="Path to netlogo/pods.csv")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH),
                        help="Output path for sku_rop.csv")
    args = parser.parse_args()

    print(f"Lead time L = {args.lead_time} day(s)  ({args.lead_time * 24:.1f} hours)")
    print(f"Z by cluster: {Z_BY_CLUSTER}")
    print(f"  Reading sku_sample: {args.sku_sample}")
    print(f"  Reading pods.csv:   {args.pods_csv}")

    if not os.path.exists(args.pods_csv):
        print(f"\nERROR: pods.csv not found at {args.pods_csv}")
        print("Please run Phase 4 of the notebook first to generate pods.csv.")
        sys.exit(1)

    result = compute_rop(
        sku_sample_path=Path(args.sku_sample),
        pods_csv_path=Path(args.pods_csv),
        lead_time=args.lead_time,
    )

    result.to_csv(args.output, index=False)
    print(f"\n  sku_rop.csv: {len(result)} SKUs written to {args.output}")
    print(f"  rop_global   — min: {result['rop_global'].min()}, "
          f"max: {result['rop_global'].max()}, "
          f"mean: {result['rop_global'].mean():.1f}")
    print(f"  rop_per_slot — min: {result['rop_per_slot'].min()}, "
          f"max: {result['rop_per_slot'].max()}, "
          f"mean: {result['rop_per_slot'].mean():.1f}")
    print(f"  n_slots      — min: {result['n_slots'].min()}, "
          f"max: {result['n_slots'].max()}")
    exceed = (result["rop_global"] > result["S_total"]).sum()
    print(f"  rop_global > S_total: {exceed} SKUs (should be 0)")


if __name__ == "__main__":
    main()
