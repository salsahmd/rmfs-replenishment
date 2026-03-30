"""
Convert SKU assignment output → NetLogo simulation input (items.csv, pods.csv).

This bridges the thesis-based FFD algorithm (sku_assignment.py) with the
RMFS simulator by producing the exact CSV formats that PodGenerator would
normally create.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

from assignment.sku_assignment import run_assignment, normalize_cols, read_csv_auto_sep


# Default inventory-level mappings (same as PodGenerator defaults)
DEFAULT_POD_INVENTORY_LEVELS = {0: 0.4, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.6}
DEFAULT_WAREHOUSE_INVENTORY_LEVELS = {0: 0.4, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.6}


def generate_items_csv(sku_sample_path, items_dict_path, output_path,
                       pod_inventory_levels=None, warehouse_inventory_levels=None):
    """
    Generate items.csv from sku_sample + items_dictionary_cleaned.

    Sorts by item_code for deterministic item_id assignment (critical so that
    the same SKU always gets the same item_id across phases).

    Returns the items DataFrame (with item_id as index).
    """
    pod_inv = pod_inventory_levels or DEFAULT_POD_INVENTORY_LEVELS
    wh_inv = warehouse_inventory_levels or DEFAULT_WAREHOUSE_INVENTORY_LEVELS

    # Load sku_sample (has cluster, demand stats, box dims)
    sku_df = normalize_cols(read_csv_auto_sep(Path(sku_sample_path)))
    sku_df["item_code"] = sku_df["item_code"].astype(str)

    # Only merge items_dict if sku_sample is missing columns we need
    extra_cols = ["item_order_frequency", "item_initial_quantity_inventory",
                  "item_volume", "item_unit", "item_quantity_order_unique"]
    missing_cols = [c for c in extra_cols if c not in sku_df.columns]
    if missing_cols and items_dict_path is not None:
        idict = normalize_cols(read_csv_auto_sep(Path(items_dict_path)))
        idict["item_code"] = idict["item_code"].astype(str)
        merged = sku_df.merge(idict[["item_code"] + missing_cols], on="item_code", how="left")
    else:
        merged = sku_df.copy()

    # Sort by item_code for deterministic item_id
    merged = merged.sort_values("item_code").reset_index(drop=True)

    # Compute item_weight
    merged["number_of_item_in_a_box"] = pd.to_numeric(
        merged["number_of_item_in_a_box"], errors="coerce"
    ).fillna(1).clip(lower=1).astype(int)
    merged["box_weight"] = pd.to_numeric(merged["box_weight"], errors="coerce").fillna(0.0)
    merged["item_weight"] = (merged["box_weight"] / merged["number_of_item_in_a_box"]).round(3)

    # Map cluster → inventory levels
    cluster_col = "cluster"
    merged["item_pod_inventory_level"] = merged[cluster_col].map(pod_inv).fillna(0.5)
    merged["item_warehouse_inventory_level"] = merged[cluster_col].map(wh_inv).fillna(0.5)

    # Fill missing values
    for col in ["item_order_frequency", "item_initial_quantity_inventory"]:
        src = merged[col] if col in merged.columns else pd.Series(0, index=merged.index)
        merged[col] = pd.to_numeric(src, errors="coerce").fillna(0).astype(int)
    for col in ["box_length", "box_width", "box_height", "box_volume", "item_volume", "cv_demand"]:
        src = merged[col] if col in merged.columns else pd.Series(0.0, index=merged.index)
        merged[col] = pd.to_numeric(src, errors="coerce").fillna(0.0)
    merged["item_unit"] = (merged["item_unit"] if "item_unit" in merged.columns else pd.Series("PCS", index=merged.index)).fillna("PCS")
    merged["item_quantity_order_unique"] = (merged["item_quantity_order_unique"] if "item_quantity_order_unique" in merged.columns else pd.Series("[]", index=merged.index)).fillna("[]")

    # Use cluster as item_class (matches how the pipeline updates items_dictionary)
    merged["item_class"] = merged[cluster_col].astype(int)

    # Build final items DataFrame in exact PodGenerator column order
    items = merged[[
        "item_code", "item_class", "item_order_frequency", "cv_demand",
        "item_initial_quantity_inventory",
        "box_length", "box_width", "box_height", "box_volume", "box_weight",
        "number_of_item_in_a_box", "item_volume", "item_weight", "item_unit",
        "item_quantity_order_unique",
        "item_pod_inventory_level", "item_warehouse_inventory_level",
    ]].copy()

    items.index.name = "item_id"
    items.to_csv(output_path, index=True)
    print(f"  items.csv: {len(items)} items written to {output_path}")
    return items


def convert_detail_to_pods_csv(detail_df, items_df, output_path,
                               pods_dict_path=None, rop_path=None):
    """
    Convert sku_assignment_detail → pods.csv in PodGenerator format.

    Maps item_code → item_id via items_df (sorted by item_code).
    Only includes assigned slots (simulation handles missing slots correctly).
    """
    if detail_df.empty:
        print("  WARNING: detail_df is empty — no pods to write")
        return pd.DataFrame()

    # Load pods_dictionary for slot structure reference
    if pods_dict_path and os.path.exists(pods_dict_path):
        pods_dict = pd.read_csv(pods_dict_path)
        default_facing = int(pods_dict["pod_face"].iloc[0]) if "pod_face" in pods_dict.columns else 0
    else:
        default_facing = 0

    # Build lookup table from items_df (index = item_id)
    lookup = items_df[["item_code", "item_weight",
                        "item_pod_inventory_level",
                        "item_warehouse_inventory_level"]].copy()
    lookup["item_code"] = lookup["item_code"].astype(str)
    lookup["item_id"] = lookup.index  # preserve original index as item_id

    # Normalise detail item_code to string for join
    df = detail_df.copy()
    df["item_code"] = df["item_code"].astype(str)

    # Merge — drops rows with no matching item_id (infeasible SKUs)
    pods_df = df.merge(lookup, on="item_code", how="inner")

    if pods_df.empty:
        print("  WARNING: no rows matched between detail_df and items_df — pods.csv will be empty")
        pods_df.to_csv(output_path, index=False)
        return pods_df

    # Merge rop_per_slot if provided
    if rop_path is not None and os.path.exists(str(rop_path)):
        rop_df = pd.read_csv(rop_path)
        rop_df["item_code"] = rop_df["item_code"].astype(str)
        pods_df = pods_df.merge(rop_df[["item_code", "rop_per_slot"]], on="item_code", how="left")
        pods_df["rop_per_slot"] = pd.to_numeric(pods_df["rop_per_slot"], errors="coerce").fillna(0).astype(int)
        # If rop_per_slot >= max_qty (item too large, only 1 unit fits per slot),
        # set to 0 so fallback ratio check is used (trigger only when slot is empty)
        qty_assigned = pd.to_numeric(pods_df["qty_items_assigned"], errors="coerce").fillna(0).astype(int)
        pods_df.loc[pods_df["rop_per_slot"] >= qty_assigned, "rop_per_slot"] = 0
    else:
        pods_df["rop_per_slot"] = 0

    pods_df["total_item_weight"] = (pods_df["item_weight"] *
                                    pd.to_numeric(pods_df["qty_items_assigned"], errors="coerce").fillna(0)).round(3)
    pods_df["slot_sequence"] = range(len(pods_df))
    pods_df["pod_type"] = 0
    pods_df["unusedColumn1"] = 0
    pods_df["unusedColumn2"] = 0
    pods_df["unusedColumn3"] = 0
    pods_df["due_date"] = 99999
    pods_df["facing"] = default_facing
    pods_df["pick_ind"] = 0

    pods_df = pods_df.rename(columns={
        "slot_idx": "slot_id",
        "qty_items_assigned": "qty",
        "item_id": "item",
    })
    pods_df["max_qty"] = pods_df["qty"]

    int_cols = ["pod_id", "pod_type", "slot_id", "slot_type", "item",
                "unusedColumn1", "unusedColumn2", "unusedColumn3",
                "qty", "max_qty", "due_date", "facing", "pick_ind", "slot_sequence",
                "rop_per_slot"]
    for col in int_cols:
        pods_df[col] = pd.to_numeric(pods_df[col], errors="coerce").fillna(0).astype(int)

    pods_df.to_csv(output_path, index=False)
    n_pods = pods_df["pod_id"].nunique()
    print(f"  pods.csv: {len(pods_df)} slots across {n_pods} pods written to {output_path}")
    return pods_df


def run_full_conversion(base_dir, netlogo_dir,
                        sku_sample_path=None, items_dict_path=None,
                        items_slots_config_path=None,
                        pod_inventory_levels=None, warehouse_inventory_levels=None,
                        rop_path=None):
    """
    Orchestrator: run assignment → generate items.csv → generate pods.csv.

    Parameters
    ----------
    base_dir : str or Path
        Project root (contains sku_sample.csv, items_dictionary_cleaned.csv, etc.)
    netlogo_dir : str or Path
        NetLogo directory where items.csv and pods.csv should be written.
    """
    base = Path(base_dir)
    netlogo = Path(netlogo_dir)

    sku_path = Path(sku_sample_path) if sku_sample_path else base / "sku_sample.csv"
    idict_path = Path(items_dict_path) if items_dict_path else base / "items_dictionary_cleaned.csv"
    islots_path = Path(items_slots_config_path) if items_slots_config_path else base / "items_slots_configuration.csv"

    print("=" * 60)
    print("SKU ASSIGNMENT → SIMULATION CONVERSION")
    print("=" * 60)

    # Step 1: Run FFD assignment
    detail_df = run_assignment(
        base_dir=base,
        sku_sample_path=sku_path,
        items_dict_path=idict_path,
        items_slots_config_path=islots_path,
    )

    # Step 2: Generate items.csv
    items_df = generate_items_csv(
        sku_sample_path=sku_path,
        items_dict_path=idict_path,
        output_path=netlogo / "items.csv",
        pod_inventory_levels=pod_inventory_levels,
        warehouse_inventory_levels=warehouse_inventory_levels,
    )

    # Step 3: Generate pods.csv
    pods_dict_path = netlogo / "pods_dictionary.csv"
    pods_df = convert_detail_to_pods_csv(
        detail_df=detail_df,
        items_df=items_df,
        output_path=netlogo / "pods.csv",
        pods_dict_path=pods_dict_path,
        rop_path=rop_path,
    )

    print("=" * 60)
    return detail_df, items_df, pods_df
