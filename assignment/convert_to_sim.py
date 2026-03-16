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

    # Load items_dictionary_cleaned (has item_initial_quantity_inventory, item_order_frequency, etc.)
    idict = normalize_cols(read_csv_auto_sep(Path(items_dict_path)))
    idict["item_code"] = idict["item_code"].astype(str)

    # Merge — sku_sample is the authority on which SKUs are included
    merged = sku_df.merge(
        idict[["item_code", "item_order_frequency", "item_initial_quantity_inventory",
               "item_volume", "item_unit", "item_quantity_order_unique"]],
        on="item_code", how="left",
    )

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
        merged[col] = pd.to_numeric(merged.get(col, 0), errors="coerce").fillna(0).astype(int)
    for col in ["box_length", "box_width", "box_height", "box_volume", "item_volume"]:
        merged[col] = pd.to_numeric(merged.get(col, 0), errors="coerce").fillna(0.0)
    merged["item_unit"] = merged.get("item_unit", "PCS").fillna("PCS")
    merged["item_quantity_order_unique"] = merged.get("item_quantity_order_unique", "[]").fillna("[]")

    # Use cluster as item_class (matches how the pipeline updates items_dictionary)
    merged["item_class"] = merged[cluster_col].astype(int)

    # Build final items DataFrame in exact PodGenerator column order
    items = merged[[
        "item_code", "item_class", "item_order_frequency",
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
                               pods_dict_path=None):
    """
    Convert sku_assignment_detail → pods.csv in PodGenerator format.

    Maps item_code → item_id via items_df (sorted by item_code).
    Only includes assigned slots (simulation handles missing slots correctly).
    """
    if detail_df.empty:
        print("  WARNING: detail_df is empty — no pods to write")
        return pd.DataFrame()

    # Build item_code → item_id mapping from items_df
    code_to_id = {}
    for item_id, row in items_df.iterrows():
        code_to_id[str(row["item_code"])] = item_id

    # Build item_code → inventory levels from items_df
    code_to_pod_inv = {}
    code_to_wh_inv = {}
    for _, row in items_df.iterrows():
        ic = str(row["item_code"])
        code_to_pod_inv[ic] = float(row["item_pod_inventory_level"])
        code_to_wh_inv[ic] = float(row["item_warehouse_inventory_level"])

    # Build item_code → item_weight from items_df
    code_to_weight = {}
    for _, row in items_df.iterrows():
        code_to_weight[str(row["item_code"])] = float(row["item_weight"])

    # Load pods_dictionary for slot structure reference
    if pods_dict_path and os.path.exists(pods_dict_path):
        pods_dict = pd.read_csv(pods_dict_path)
        # Get facing from pods_dictionary (typically all 0 for pod_type 0)
        default_facing = int(pods_dict["pod_face"].iloc[0]) if "pod_face" in pods_dict.columns else 0
    else:
        default_facing = 0

    rows = []
    slot_seq = 0
    for _, d in detail_df.iterrows():
        item_code = str(d["item_code"])
        item_id = code_to_id.get(item_code)
        if item_id is None:
            continue

        item_weight = code_to_weight.get(item_code, 0.0)
        qty = int(d["qty_items_assigned"])
        total_weight = round(item_weight * qty, 3)

        rows.append({
            "pod_id": int(d["pod_id"]),
            "pod_type": 0,
            "slot_id": int(d["slot_idx"]),
            "slot_type": int(d["slot_type"]),
            "item": item_id,
            "unusedColumn1": 0,
            "unusedColumn2": 0,
            "unusedColumn3": 0,
            "qty": qty,
            "max_qty": qty,
            "due_date": 99999,
            "facing": default_facing,
            "pick_ind": 0,
            "slot_sequence": slot_seq,
            "item_weight": item_weight,
            "total_item_weight": total_weight,
            "item_pod_inventory_level": code_to_pod_inv.get(item_code, 0.5),
            "item_warehouse_inventory_level": code_to_wh_inv.get(item_code, 0.5),
        })
        slot_seq += 1

    pods_df = pd.DataFrame(rows)

    # Ensure integer types for columns that must be int
    for col in ["pod_id", "pod_type", "slot_id", "slot_type", "item",
                "unusedColumn1", "unusedColumn2", "unusedColumn3",
                "qty", "max_qty", "due_date", "facing", "pick_ind", "slot_sequence"]:
        pods_df[col] = pods_df[col].astype(int)

    pods_df.to_csv(output_path, index=False)
    n_pods = pods_df["pod_id"].nunique()
    print(f"  pods.csv: {len(pods_df)} slots across {n_pods} pods written to {output_path}")
    return pods_df


def run_full_conversion(base_dir, netlogo_dir,
                        sku_sample_path=None, items_dict_path=None,
                        items_slots_config_path=None,
                        pod_inventory_levels=None, warehouse_inventory_levels=None):
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
    )

    print("=" * 60)
    return detail_df, items_df, pods_df
