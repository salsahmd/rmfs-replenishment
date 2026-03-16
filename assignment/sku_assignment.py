"""
SKU-to-Pod Assignment  —  DC6 (ABC/XYZ Dimension-Constrained)
==============================================================
Implements the greedy FFD-inspired algorithm from Table 3.6 of the thesis.

Inputs
------
  items_slots_configuration.csv  — feasible (item, slot_type, orientation) rows
                                   produced by item_pod_config.py
  sku_kmeans_clusters_cleaned.csv — cluster label + demand stats per SKU
  sku_dictionary.csv             — box_weight, units_per_box,
                                   item_initial_inventory (items),
                                   item_initial_quantity_inventory (actual day-0 stock)

Parameters
----------
  MAX_POD_WEIGHT   = 1300 kg
  CLUSTER_PRIORITY = [4, 1, 3, 0, 2]  — class 4 is highest priority (high demand)

Pod structure (from paper Table 3.6 line 2)
-------------------------------------------
  POD_STRUCTURE = [0,0,0,1,1,1,2,2,2,2,2,2,2,2,3,3,4]
  Every new pod is created with exactly these 17 slots (by slot_type).
  Pods are created on demand — no hard cap.

Assignment flow (paper Table 3.6)
----------------------------------
  1. Build flexibility metadata per SKU (num_compatible_slots, min_slot_type).
  2. Sort: [cluster_priority ASC, num_compatible_slots ASC,
             min_slot_type ASC, initial_inventory_boxes DESC]
  3. Pass 1 — constrained items (num_compatible_slots ≤ 2): first-fit greedy.
  4. Pass 2 — flexible items   (num_compatible_slots >  2): first-fit greedy.
     Both passes: while remaining_boxes > 0 →
       • Scan existing pods in order for a free compatible slot.
       • If none found → create a new pod.
  5. Pass 3 — secondary fill: scan every empty slot across all pods and
     place any item that still has unassigned boxes.

Outputs
-------
  sku_assignment_detail.csv  — one row per (pod, slot) assignment
  sku_assignment.csv         — one row per SKU with totals
  pod_summary.csv            — one row per pod
  unassigned_skus.csv        — items that could not be fully assigned
"""

import time
import pandas as pd
import numpy as np
from pathlib import Path
from math import ceil, sqrt
from collections import Counter

# =============================================================================
# PARAMETERS
# =============================================================================
TOTAL_PODS      = 300
MAX_POD_WEIGHT  = 1300.0   # kg
LEAD_TIME       = 1.0      # days
SERVICE_LEVEL_Z = 1.2816   # 90 % service level

# Cluster priority: index 0 = highest priority
# Class 4 = high-demand (ABC/XYZ A-class), assigned first and dispersed
CLUSTER_PRIORITY = [4, 1, 3, 0, 2]

# Pod slot structure (Table 3.6, line 2 of paper)
# Each new pod is created with exactly these slot_types in this order.
POD_STRUCTURE = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4]

# Constrained = num_compatible_slots <= this threshold (paper line 18)
CONSTRAINED_THRESHOLD = 2

# =============================================================================
# I/O PATHS
# =============================================================================
ITEMS_SLOTS_CONFIG  = Path("items_slots_configuration.csv")
SKU_SAMPLE_FILE     = Path("sku_sample.csv")                # sampled SKUs (primary source)
ITEMS_DICT_FILE     = Path("items_dictionary_cleaned.csv")  # item_initial_quantity_inventory

OUT_DETAIL          = Path("sku_assignment_detail.csv")
OUT_SKU_ASSIGNMENT  = Path("sku_assignment.csv")
OUT_POD_SUMMARY     = Path("pod_summary.csv")
OUT_UNASSIGNED      = Path("unassigned_skus.csv")


# =============================================================================
# HELPERS
# =============================================================================
def read_csv_auto_sep(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=",")
    return df


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: c.strip().replace("\ufeff", "") for c in df.columns})


def cluster_priority_rank(c) -> int:
    """Lower rank = processed earlier (higher priority)."""
    try:
        return CLUSTER_PRIORITY.index(int(c))
    except ValueError:
        return len(CLUSTER_PRIORITY)


def parse_dot_comma_float(val) -> float:
    """Handles '15.07.00' and '0,395' locale formats (legacy parser, kept for safety)."""
    s = str(val).strip().replace(",", ".")
    parts = s.split(".")
    try:
        return float(parts[0] + ("." + parts[1] if len(parts) > 1 else ""))
    except (ValueError, IndexError):
        return np.nan


def load_sku_dict(path: Path) -> pd.DataFrame:
    """
    Load sku_dictionary.csv — box_weight and units_per_box only.
    (item_initial_inventory here is the safety-stock formula result, not used.)
    """
    df = normalize_cols(read_csv_auto_sep(path))
    df["item_code"]    = df["item_code"].astype(str)
    df["box_weight"]   = pd.to_numeric(df["box_weight"],               errors="coerce").fillna(0.0)
    df["units_per_box"] = pd.to_numeric(df["number_of_item_in_a_box"], errors="coerce").fillna(1.0).clip(lower=1.0)
    return df[["item_code", "box_weight", "units_per_box"]]


def load_items_dict(path: Path) -> pd.DataFrame:
    """
    Load items_dictionary_cleaned.csv.
    Extracts item_initial_quantity_inventory (actual warehouse stock on day 0,
    in item units) and converts to boxes:
        initial_inventory_boxes = ceil(item_initial_quantity_inventory / units_per_box)
    Total: ~12,906 boxes — the value the literature used.
    """
    df = normalize_cols(read_csv_auto_sep(path))
    df["item_code"]    = df["item_code"].astype(str)
    df["units_per_box"] = pd.to_numeric(df["number_of_item_in_a_box"],       errors="coerce").fillna(1.0).clip(lower=1.0)
    df["qty_inv_items"] = pd.to_numeric(df["item_initial_quantity_inventory"], errors="coerce").fillna(0.0).clip(lower=0)
    df["initial_inventory_boxes"] = np.where(
        df["qty_inv_items"] > 0,
        np.ceil(df["qty_inv_items"] / df["units_per_box"]).astype(int),
        0,
    ).astype(int)
    return df[["item_code", "qty_inv_items", "initial_inventory_boxes"]]


# =============================================================================
# LOAD & PREPARE DATA
# =============================================================================
def load_config(path: Path) -> pd.DataFrame:
    """
    Load items_slots_configuration.csv.
    Keeps only the BEST orientation per (item_code, slot_type):
      primary   : max_boxes_in_slot DESC
      secondary : vol_util_slot DESC
    This mirrors build_config_lookup() but without pod_type (the new config
    has no pod_type column — one flat set of slot types for all pods).
    """
    df = normalize_cols(read_csv_auto_sep(path))

    needed = {"item_code", "slot_type", "max_boxes_in_slot"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"items_slots_configuration.csv missing: {missing}")

    df["item_code"]         = df["item_code"].astype(str)
    df["slot_type"]         = pd.to_numeric(df["slot_type"],         errors="coerce")
    df["max_boxes_in_slot"] = pd.to_numeric(df["max_boxes_in_slot"], errors="coerce").fillna(0)
    df["vol_util_slot"]     = pd.to_numeric(df.get("vol_util_slot",  pd.Series(0.0, index=df.index)),
                                            errors="coerce").fillna(0.0)
    df["orientation"]       = pd.to_numeric(df.get("orientation",    pd.Series(-1,  index=df.index)),
                                            errors="coerce").fillna(-1).astype(int)
    df["box_weight"]        = pd.to_numeric(df.get("box_weight",     pd.Series(0.0, index=df.index)),
                                            errors="coerce").fillna(0.0)
    df["units_per_box"]     = pd.to_numeric(df.get("units_per_box",  pd.Series(1.0, index=df.index)),
                                            errors="coerce").fillna(1.0)

    # used_volume_in_slot and slot_volume for dynamic vol_util computation
    for col in ("used_volume_in_slot", "slot_length", "slot_width", "slot_height"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    df = df.dropna(subset=["slot_type"]).copy()
    df["slot_type"] = df["slot_type"].astype(int)
    df = df[df["max_boxes_in_slot"] > 0].copy()

    # Best orientation per (item_code, slot_type)
    best = (
        df.sort_values(
            ["item_code", "slot_type", "max_boxes_in_slot", "vol_util_slot"],
            ascending=[True, True, False, False],
            kind="mergesort",
        )
        .groupby(["item_code", "slot_type"], as_index=False)
        .head(1)
        .copy()
    )
    return best


def load_sku_sample(path: Path) -> pd.DataFrame:
    """Load sku_sample.csv — primary source for SKUs, demand stats, and box dimensions."""
    df = normalize_cols(read_csv_auto_sep(path))
    df["item_code"] = df["item_code"].astype(str)
    for c in ["mean_demand", "std_demand", "cv_demand"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    if "std_demand" not in df.columns:
        df["std_demand"] = df.get("cv_demand", 0.0) * df["mean_demand"]
    df["box_weight"]   = pd.to_numeric(df["box_weight"],               errors="coerce").fillna(0.0)
    df["units_per_box"] = pd.to_numeric(df["number_of_item_in_a_box"], errors="coerce").fillna(1.0).clip(lower=1.0)
    return df


def build_item_table(cluster_df: pd.DataFrame, config_df: pd.DataFrame,
                     items_dict_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge all sources into one item table.

    Data sources
    ------------
    box_weight, units_per_box       → sku_sample.csv (already in cluster_df)
    qty_inv_items,
    initial_inventory_boxes         → items_dictionary_cleaned.csv
                                      ceil(item_initial_QUANTITY_inventory / units_per_box)
                                      = actual day-0 stock
    slot compatibility metadata     → items_slots_configuration.csv
    """
    df = cluster_df.copy()
    df["box_weight"]   = df["box_weight"].fillna(0.0)
    df["units_per_box"] = df["units_per_box"].fillna(1.0).clip(lower=1.0)

    # Step 2: actual initial inventory from items_dictionary_cleaned
    df = df.merge(
        items_dict_df[["item_code", "qty_inv_items", "initial_inventory_boxes"]],
        on="item_code", how="left"
    )
    df["qty_inv_items"]           = df["qty_inv_items"].fillna(0.0)
    df["initial_inventory_boxes"] = df["initial_inventory_boxes"].fillna(0).astype(int)

    # Step 3: slot compatibility metadata (Table 3.6 lines 6-14)
    compat = config_df.groupby("item_code")["slot_type"].agg(list).reset_index()
    compat.columns = ["item_code", "compatible_slot_types"]
    compat["num_compatible_slots"] = compat["compatible_slot_types"].apply(len)
    compat["min_slot_type"]        = compat["compatible_slot_types"].apply(min)
    compat["max_slot_type"]        = compat["compatible_slot_types"].apply(max)
    compat["compatible_slot_types"] = compat["compatible_slot_types"].apply(set)

    df = df.merge(compat, on="item_code", how="left")
    df["num_compatible_slots"] = df["num_compatible_slots"].fillna(0).astype(int)
    df["min_slot_type"]        = df["min_slot_type"].fillna(0).astype(int)
    df["max_slot_type"]        = df["max_slot_type"].fillna(0).astype(int)
    df["compatible_slot_types"] = df["compatible_slot_types"].apply(
        lambda x: x if isinstance(x, set) else set()
    )

    df["cluster_priority_rank"] = df["cluster"].apply(cluster_priority_rank)
    return df


# =============================================================================
# CONFIG LOOKUP:  (item_code, slot_type)  ->  config dict
# =============================================================================
def build_lookup(config_df: pd.DataFrame) -> dict:
    """
    Returns a dict keyed by (item_code, slot_type) with the best config row.
    """
    lookup = {}
    for _, r in config_df.iterrows():
        key = (str(r["item_code"]), int(r["slot_type"]))
        slot_vol = float(r["slot_length"]) * float(r["slot_width"]) * float(r["slot_height"])
        lookup[key] = {
            "max_boxes_in_slot":   int(r["max_boxes_in_slot"]),
            "max_items_in_slot":   int(r.get("max_items_in_slot", r["max_boxes_in_slot"])),
            "orientation":         int(r["orientation"]),
            "box_weight":          float(r["box_weight"]),
            "units_per_box":       float(r["units_per_box"]),
            "vol_util_slot":       float(r["vol_util_slot"]),
            "used_volume_in_slot": float(r["used_volume_in_slot"]),
            "slot_volume":         slot_vol,
        }
    return lookup


# =============================================================================
# POD MANAGEMENT
# =============================================================================
def make_pod(pod_id: int) -> dict:
    """
    Creates a new pod with the standard slot structure from the paper.
    POD_STRUCTURE defines the sequence of slot_types in one pod.
    """
    slots = [
        {
            "slot_idx":  idx,
            "slot_type": st,
            "item_code": None,    # None = free
            "qty_boxes": 0,
            "qty_items": 0,
            "orientation": None,
            "cluster":   None,
        }
        for idx, st in enumerate(POD_STRUCTURE)
    ]
    return {
        "pod_id":       pod_id,
        "total_weight": 0.0,
        "slots":        slots,
        "class_set":    set(),   # clusters present in this pod
    }


def select_best_config(cfg: dict, assign_boxes: int) -> dict:
    """Compute dynamic vol_util for assign_boxes using stored config."""
    max_b    = cfg["max_boxes_in_slot"]
    used_vol = cfg["used_volume_in_slot"]
    slot_vol = cfg["slot_volume"]
    if max_b > 0 and slot_vol > 0 and used_vol > 0:
        dyn_vol_util = min((used_vol / max_b * assign_boxes) / slot_vol, 1.0)
    else:
        dyn_vol_util = cfg["vol_util_slot"]
    return {**cfg, "dyn_vol_util": dyn_vol_util}


# =============================================================================
# CORE ASSIGNMENT
# =============================================================================
def assign_item(item_code: str, cluster: int, remaining_boxes: int,
                units_per_box: int, box_weight: float,
                compatible_slot_types: set,
                pods: list, lookup: dict,
                next_pod_id_ref: list) -> tuple:
    """
    Pure first-fit greedy — faithful to paper Table 3.6.

    Tracks remaining inventory in BOXES (not item units), because:
      - Slots physically hold boxes (max_boxes_in_slot is the capacity unit).
      - item_initial_quantity_inventory is already converted to boxes before
        this function is called.
      - assign_boxes = min(remaining_boxes, max_boxes_in_slot) — clean integer
        arithmetic with no overshoot.

    Slot selection within a pod: tries LARGEST compatible slot_type first.
    This preserves small slots (type 0/1, only 3 per pod) for constrained items
    that cannot fit anywhere else, while flexible items (can use type 2/3) are
    directed to the more abundant larger slots.

    Returns (remaining_boxes_after, detail_rows_added).
    """
    detail_rows = []
    # Sort compatible slot types largest-first, used when picking within a pod
    preferred_types = sorted(compatible_slot_types, reverse=True)

    while remaining_boxes > 0:
        chosen_pod    = None
        chosen_slot   = None
        chosen_cfg    = None
        chosen_boxes  = 0
        chosen_weight = 0.0

        # Scan pods in REVERSE order (most recently opened first = "last-fit").
        # Classic first-fit (pod 0,1,2,...) spreads each item type thinly across
        # every pod: e.g. type-2 items end up 1-per-pod across 6388 pods instead
        # of 8-per-pod in 799 pods.  Last-fit fills the current open pod to
        # capacity before advancing — dramatically reducing pod count.
        for pod in reversed(pods):
            for st in preferred_types:
                for slot in pod["slots"]:
                    if slot["item_code"] is not None:
                        continue
                    if slot["slot_type"] != st:
                        continue

                    cfg = lookup.get((item_code, st))
                    if cfg is None:
                        continue

                    assign_boxes = min(remaining_boxes, cfg["max_boxes_in_slot"])
                    if assign_boxes <= 0:
                        continue

                    added_weight = assign_boxes * box_weight
                    if pod["total_weight"] + added_weight > MAX_POD_WEIGHT:
                        continue

                    chosen_pod    = pod
                    chosen_slot   = slot
                    chosen_cfg    = select_best_config(cfg, assign_boxes)
                    chosen_boxes  = assign_boxes
                    chosen_weight = added_weight
                    break

                if chosen_slot is not None:
                    break

            if chosen_slot is not None:
                break

        # Paper lines 34-37: no slot found → open a new pod
        if chosen_slot is None:
            new_pod = make_pod(next_pod_id_ref[0])
            next_pod_id_ref[0] += 1
            pods.append(new_pod)

            for st in preferred_types:
                for slot in new_pod["slots"]:
                    if slot["slot_type"] != st:
                        continue
                    cfg = lookup.get((item_code, st))
                    if cfg is None:
                        continue
                    assign_boxes = min(remaining_boxes, cfg["max_boxes_in_slot"])
                    if assign_boxes <= 0:
                        continue
                    added_weight = assign_boxes * box_weight
                    if new_pod["total_weight"] + added_weight > MAX_POD_WEIGHT:
                        continue

                    chosen_pod    = new_pod
                    chosen_slot   = slot
                    chosen_cfg    = select_best_config(cfg, assign_boxes)
                    chosen_boxes  = assign_boxes
                    chosen_weight = added_weight
                    break

                if chosen_slot is not None:
                    break

            if chosen_slot is None:
                break   # truly unassignable

        # Commit
        assign_items = chosen_boxes * units_per_box

        chosen_slot["item_code"]   = item_code
        chosen_slot["qty_boxes"]   = chosen_boxes
        chosen_slot["qty_items"]   = assign_items
        chosen_slot["orientation"] = chosen_cfg["orientation"]
        chosen_slot["cluster"]     = cluster

        chosen_pod["total_weight"] += chosen_weight
        chosen_pod["class_set"].add(cluster)

        remaining_boxes -= chosen_boxes

        detail_rows.append({
            "pod_id":             chosen_pod["pod_id"],
            "slot_idx":           chosen_slot["slot_idx"],
            "slot_type":          chosen_slot["slot_type"],
            "item_code":          item_code,
            "cluster":            cluster,
            "orientation":        chosen_cfg["orientation"],
            "qty_boxes_assigned": chosen_boxes,
            "qty_items_assigned": assign_items,
            "box_weight":         box_weight,
            "weight_assigned":    chosen_weight,
            "pod_weight_after":   chosen_pod["total_weight"],
            "vol_util_slot":      chosen_cfg["dyn_vol_util"],
            "max_boxes_in_slot":  chosen_cfg["max_boxes_in_slot"],
        })

    return remaining_boxes, detail_rows


# =============================================================================
# SECONDARY FILL PASS  (paper lines 43-52)
# =============================================================================
def secondary_fill(pods: list, items_df: pd.DataFrame,
                   lookup: dict) -> list:
    """
    Paper lines 43-52: for every empty slot in every pod, fill it with
    the highest-priority item that physically fits — regardless of whether
    that item is already assigned elsewhere.

    The paper places items REDUNDANTLY: the same item_code can appear in
    multiple pods. This is intentional — it maximises slot utilisation and
    minimises total pod count. The warehouse simply stocks the item at each
    pod location it is assigned to.

    Selection priority (per paper): cluster priority rank ASC, then
    vol_util DESC (most space-efficient orientation).
    """
    extra_rows = []

    # --- DIAGNOSTIC ---
    total_empty = sum(1 for pod in pods for slot in pod["slots"] if slot["item_code"] is None)
    print(f"    [secondary_fill] pods={len(pods)}, total_empty_slots={total_empty}")
    cand_counts = {st: len(v) for st, v in slot_type_candidates.items()} if 'slot_type_candidates' in dir() else {}
    # Build a fast per-slot-type candidate list, sorted by priority
    # so we can quickly find the best item for each empty slot
    slot_type_candidates: dict = {}  # slot_type -> [(rank, vol_util, item_code, info, cfg)]
    for _, row in items_df.iterrows():
        item_code = str(row["item_code"])
        compat    = row["compatible_slot_types"]
        if not compat:
            continue
        rank = cluster_priority_rank(int(row["cluster"]))
        for st in compat:
            cfg = lookup.get((item_code, st))
            if cfg is None:
                continue
            if st not in slot_type_candidates:
                slot_type_candidates[st] = []
            slot_type_candidates[st].append(
                (rank, -cfg["vol_util_slot"], item_code, row, cfg)
            )

    # Sort each list: lower rank first, then higher vol_util first
    for st in slot_type_candidates:
        slot_type_candidates[st].sort(key=lambda x: (x[0], x[1]))

    # --- DIAGNOSTIC ---
    print(f"    [secondary_fill] candidate slot types: { {st: len(v) for st,v in slot_type_candidates.items()} }")

    for pod in pods:
        for slot in pod["slots"]:
            if slot["item_code"] is not None:
                continue

            st = slot["slot_type"]
            candidates = slot_type_candidates.get(st, [])

            chosen = None
            for rank, neg_vu, item_code, info, cfg in candidates:
                assign_boxes = cfg["max_boxes_in_slot"]   # fill slot completely
                assign_boxes = min(assign_boxes, int(info["initial_inventory_boxes"]) or assign_boxes)
                assign_boxes = max(assign_boxes, 1)

                added_weight = assign_boxes * float(info["box_weight"])
                if pod["total_weight"] + added_weight > MAX_POD_WEIGHT:
                    # Try with fewer boxes
                    max_by_weight = int((MAX_POD_WEIGHT - pod["total_weight"]) / max(float(info["box_weight"]), 0.001))
                    if max_by_weight <= 0:
                        continue
                    assign_boxes = min(assign_boxes, max_by_weight)
                    added_weight = assign_boxes * float(info["box_weight"])

                chosen = (item_code, info, cfg, assign_boxes, added_weight)
                break

            if chosen is None:
                continue

            item_code, info, cfg, assign_boxes, added_weight = chosen
            assign_items = assign_boxes * int(info["units_per_box"])
            cfg_ex       = select_best_config(cfg, assign_boxes)

            slot["item_code"]   = item_code
            slot["qty_boxes"]   = assign_boxes
            slot["qty_items"]   = assign_items
            slot["orientation"] = cfg_ex["orientation"]
            slot["cluster"]     = int(info["cluster"])

            pod["total_weight"] += added_weight
            pod["class_set"].add(int(info["cluster"]))

            extra_rows.append({
                "pod_id":             pod["pod_id"],
                "slot_idx":           slot["slot_idx"],
                "slot_type":          st,
                "item_code":          item_code,
                "cluster":            int(info["cluster"]),
                "orientation":        cfg_ex["orientation"],
                "qty_boxes_assigned": assign_boxes,
                "qty_items_assigned": assign_items,
                "box_weight":         float(info["box_weight"]),
                "weight_assigned":    added_weight,
                "pod_weight_after":   pod["total_weight"],
                "vol_util_slot":      cfg_ex["dyn_vol_util"],
                "max_boxes_in_slot":  cfg["max_boxes_in_slot"],
            })

    return extra_rows


# =============================================================================
# BUILD OUTPUT TABLES
# =============================================================================
def build_outputs(detail_rows: list, pods: list,
                  items_df: pd.DataFrame, unassigned: list):

    detail_df = pd.DataFrame(detail_rows) if detail_rows else pd.DataFrame()

    # ── pod_summary ──────────────────────────────────────────────────────────
    pod_rows = []
    for pod in pods:
        used   = [s for s in pod["slots"] if s["item_code"] is not None]
        total  = len(pod["slots"])
        pod_rows.append({
            "pod_id":        pod["pod_id"],
            "used_slots":    len(used),
            "total_slots":   total,
            "empty_slots":   total - len(used),
            "total_boxes":   sum(s["qty_boxes"] for s in used),
            "total_items":   sum(s["qty_items"] for s in used),
            "total_weight":  pod["total_weight"],
            "num_classes":   len(pod["class_set"]),
            "classes":       "|".join(str(c) for c in sorted(pod["class_set"])),
        })
    pod_summary_df = pd.DataFrame(pod_rows)

    # ── sku_assignment (one row per SKU) ─────────────────────────────────────
    if not detail_df.empty:
        sku_stats = (
            detail_df.groupby("item_code", as_index=False).agg(
                total_boxes_assigned=("qty_boxes_assigned", "sum"),
                total_items_assigned=("qty_items_assigned", "sum"),
                total_weight_assigned=("weight_assigned", "sum"),
                pods_used=("pod_id", "nunique"),
                slots_used=("slot_idx", "count"),
            )
        )
        sku_df = items_df[[
            "item_code", "cluster", "mean_demand", "std_demand", "cv_demand",
            "initial_inventory_boxes", "qty_inv_items",
            "num_compatible_slots", "min_slot_type",
        ]].merge(sku_stats, on="item_code", how="left")
    else:
        sku_df = items_df.copy()
        for c in ["total_boxes_assigned","total_items_assigned",
                  "total_weight_assigned","pods_used","slots_used"]:
            sku_df[c] = 0

    for c in ["total_boxes_assigned","total_items_assigned",
              "total_weight_assigned","pods_used","slots_used"]:
        sku_df[c] = sku_df[c].fillna(0)

    unassigned_df = pd.DataFrame(unassigned) if unassigned else pd.DataFrame()

    return detail_df, sku_df, pod_summary_df, unassigned_df


# =============================================================================
# MAIN
# =============================================================================
def run_assignment(base_dir=None, sku_sample_path=None,
                   items_dict_path=None, items_slots_config_path=None):
    """
    Run the FFD assignment algorithm with configurable paths.

    Parameters
    ----------
    base_dir : str or Path, optional
        Base directory for default file resolution. Defaults to CWD.
    sku_sample_path : str or Path, optional
        Path to sku_sample.csv (overrides base_dir default).
    items_dict_path : str or Path, optional
        Path to items_dictionary_cleaned.csv (overrides base_dir default).
    items_slots_config_path : str or Path, optional
        Path to items_slots_configuration.csv (overrides base_dir default).

    Returns
    -------
    detail_df : pd.DataFrame
        The sku_assignment_detail table (one row per pod/slot assignment).
    """
    base = Path(base_dir) if base_dir else Path.cwd()
    sku_sample_p = Path(sku_sample_path) if sku_sample_path else base / "sku_sample.csv"
    items_dict_p = Path(items_dict_path) if items_dict_path else base / "items_dictionary_cleaned.csv"
    items_slots_p = Path(items_slots_config_path) if items_slots_config_path else base / "items_slots_configuration.csv"

    _t0 = time.perf_counter()
    print("=" * 62)
    print("SKU-to-Pod Assignment  (DC6 / ABC-XYZ Greedy FFD)")
    print("=" * 62)

    config_df = load_config(items_slots_p)
    print(f"  {len(config_df)} best-config rows  ({config_df['item_code'].nunique()} unique SKUs)")

    cluster_df = load_sku_sample(sku_sample_p)
    sample_codes = set(cluster_df["item_code"].unique())
    print(f"  {len(cluster_df)} SKUs in sample")

    config_df = config_df[config_df["item_code"].isin(sample_codes)].copy()

    items_dict_df = load_items_dict(items_dict_p)
    items_dict_df = items_dict_df[items_dict_df["item_code"].isin(sample_codes)].copy()
    total_boxes = items_dict_df["initial_inventory_boxes"].sum()
    print(f"  Total initial inventory: {total_boxes:,} boxes")

    items_df = build_item_table(cluster_df, config_df, items_dict_df)

    no_config = items_df[items_df["num_compatible_slots"] == 0]
    if len(no_config) > 0:
        print(f"  [WARN] {len(no_config)} SKUs have no config entry (no slot fits their dimensions).")

    no_inv = items_df[items_df["initial_inventory_boxes"] == 0]
    if len(no_inv) > 0:
        print(f"  [WARN] {len(no_inv)} SKUs have zero initial inventory — skipped.")

    priority_order = {cl: i + 1 for i, cl in enumerate(CLUSTER_PRIORITY)}
    print(f"\nCluster distribution (processed in order: {CLUSTER_PRIORITY}):")
    for cl in sorted(items_df["cluster"].unique()):
        n     = (items_df["cluster"] == cl).sum()
        order = priority_order.get(cl, "?")
        print(f"  cluster {cl}: {n:>5} SKUs  (processed {order}{['st','nd','rd','th','th'][min(order-1,4)]})")

    # ── Sort (Table 3.6, line 15) ────────────────────────────────────────────
    items_df = items_df.sort_values(
        by=["cluster_priority_rank", "num_compatible_slots",
            "min_slot_type", "initial_inventory_boxes"],
        ascending=[True, True, True, False],
        kind="mergesort",
    ).reset_index(drop=True)

    constrained = items_df[items_df["num_compatible_slots"] <= CONSTRAINED_THRESHOLD].copy()
    flexible    = items_df[items_df["num_compatible_slots"] >  CONSTRAINED_THRESHOLD].copy()
    print(f"\nConstrained items (≤{CONSTRAINED_THRESHOLD} slot types): {len(constrained)}")
    print(f"Flexible items    (>{CONSTRAINED_THRESHOLD} slot types): {len(flexible)}")

    # ── Build lookup ─────────────────────────────────────────────────────────
    lookup = build_lookup(config_df)

    # ── Assignment state ─────────────────────────────────────────────────────
    pods            = []
    next_pod_id_ref = [0]
    all_detail_rows = []
    unassigned_list = []
    assigned_qty    = {}

    def run_pass(subset: pd.DataFrame):
        for _, row in subset.iterrows():
            item_code  = str(row["item_code"])
            cluster    = int(row["cluster"])
            box_weight = float(row["box_weight"])
            units_pb   = int(max(row["units_per_box"], 1))
            inv_boxes  = int(row["initial_inventory_boxes"])
            compat     = row["compatible_slot_types"]

            if inv_boxes <= 0 or not compat:
                continue

            remaining, rows = assign_item(
                item_code             = item_code,
                cluster               = cluster,
                remaining_boxes       = inv_boxes,
                units_per_box         = units_pb,
                box_weight            = box_weight,
                compatible_slot_types = compat,
                pods                  = pods,
                lookup                = lookup,
                next_pod_id_ref       = next_pod_id_ref,
            )

            all_detail_rows.extend(rows)
            assigned_qty[item_code] = assigned_qty.get(item_code, 0) + (inv_boxes - remaining)

            if remaining > 0:
                unassigned_list.append({
                    "item_code":               item_code,
                    "cluster":                 cluster,
                    "initial_inventory_boxes": inv_boxes,
                    "unassigned_boxes":        remaining,
                    "reason":                  "no_compatible_slot",
                })

    # ── Single combined pass: constrained items first (via sort), then flexible ─
    # WHY single pass beats two separate passes:
    #   Two-pass: Pass 1 opens 389 pods for constrained items, leaving ~11 large
    #   slots empty per pod. Pass 2 flexible items use first-fit and SPREAD across
    #   all 389 pods (1 item per pod per scan), never filling a pod completely
    #   before moving to the next. Result: 205 extra pods opened unnecessarily.
    #
    #   Single pass: items are already sorted constrained-first (num_compatible_slots
    #   ASC). As each new pod opens, the very next flexible item in the sorted queue
    #   fills its remaining large slots — each pod reaches ~17/17 before a new one
    #   opens. Expected result: ~370 pods (matching literature).
    print("\n[Pass 1+2] Assigning all items (constrained first, then flexible) …")
    run_pass(items_df)
    print(f"  Pods created       : {next_pod_id_ref[0]}")

    # ── Secondary fill (paper lines 43-52) ──────────────────────────────────
    print("[Pass 3] Secondary fill of empty slots …")
    extra = secondary_fill(pods, items_df, lookup)
    all_detail_rows.extend(extra)
    print(f"  Extra assignments   : {len(extra)}")

    # ── Build outputs ────────────────────────────────────────────────────────
    detail_df, sku_df, pod_summary_df, unassigned_df = build_outputs(
        all_detail_rows, pods, items_df, unassigned_list
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    detail_df.to_csv(OUT_DETAIL, index=False)
    sku_df.to_csv(OUT_SKU_ASSIGNMENT, index=False)
    pod_summary_df.to_csv(OUT_POD_SUMMARY, index=False)
    unassigned_df.to_csv(OUT_UNASSIGNED, index=False)

    runtime_sec = time.perf_counter() - _t0

    # ── Statistics ───────────────────────────────────────────────────────────
    eligible            = items_df[
        (items_df["num_compatible_slots"] > 0) &
        (items_df["initial_inventory_boxes"] > 0)
    ]
    total_skus_eligible = len(eligible)
    assigned_skus       = int((sku_df["total_boxes_assigned"] > 0).sum())
    unassigned_n        = len(unassigned_list)
    active_pods         = len(pods)

    # Space utilisation
    slot_vols = {st: 0.0 for st in set(POD_STRUCTURE)}
    for st in POD_STRUCTURE:
        sample = config_df[config_df["slot_type"] == st]
        if not sample.empty:
            r = sample.iloc[0]
            slot_vols[st] = float(r["slot_length"]) * float(r["slot_width"]) * float(r["slot_height"])

    total_slot_volume = sum(slot_vols[st] for st in POD_STRUCTURE) * active_pods
    used_volume = 0.0
    if not detail_df.empty and "vol_util_slot" in detail_df.columns:
        for _, dr in detail_df.iterrows():
            st  = int(dr["slot_type"])
            vu  = float(dr["vol_util_slot"])
            used_volume += vu * slot_vols.get(st, 0.0)
    space_util_pct = (used_volume / total_slot_volume * 100) if total_slot_volume > 0 else 0.0

    if not detail_df.empty:
        skus_per_pod    = detail_df.groupby("pod_id")["item_code"].nunique()
        avg_skus_per_pod = skus_per_pod.mean()
    else:
        avg_skus_per_pod = 0.0

    avg_classes_per_pod = pod_summary_df["num_classes"].mean() if not pod_summary_df.empty else 0.0

    if not detail_df.empty:
        pods_per_sku     = detail_df.groupby("item_code")["pod_id"].nunique()
        avg_pods_per_sku = pods_per_sku.mean()
    else:
        avg_pods_per_sku = 0.0

    print(f"\n  Pods: {active_pods}, SKUs assigned: {assigned_skus}/{total_skus_eligible}, "
          f"space util: {space_util_pct:.1f}%, runtime: {runtime_sec:.1f}s")

    if unassigned_n > 0:
        print(f"  Unassigned SKUs: {unassigned_n}")

    return detail_df


def main():
    """CLI entry point — calls run_assignment() with default CWD-relative paths."""
    run_assignment()


if __name__ == "__main__":
    main()