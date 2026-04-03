"""
SKU-to-Pod Assignment
=====================
Greedy slot-filling algorithm with 1-item-per-pod constraint.

Inputs
------
  items_slots_configuration.csv – feasible (item, slot_type) rows (project root)
  sku_sample.csv                – cluster label, demand stats, box dimensions,
                                  initial inventory (boxes)
  pods_dictionary.csv           – slot dimensions per slot_type (project root)

Parameters
----------
  TOTAL_PODS       = 300        – hard cap; no new pods created beyond this
  MAX_POD_WEIGHT   = 1300 kg
  CLUSTER_PRIORITY = [0, 2, 3, 1]  – clusters 0 and 2 (low demand) placed first
                                      to ensure they get space; then 3 and 1

Pod structure
-------------
  POD_STRUCTURE = [0] * 10  – pod_type 0 from pods_dictionary.csv:
                               10 identical slots, each 60×50×40 mm (slot_type 0)

Constraints
-----------
  1. Each slot holds exactly 1 item_code.
  2. Each item_code may appear at most once per pod (1 slot per item per pod).
     If an item's initial inventory exceeds one slot's capacity, it is dispersed
     across multiple pods – 1 slot each.
  3. Total pods are capped at TOTAL_PODS = 300. Any inventory that cannot be
     placed within 300 pods is flagged as unplaced in the output.

Assignment flow
---------------
  1. Build slot compatibility metadata per SKU (num_compatible_slots).
  2. Sort by: [cluster_priority ASC, num_compatible_slots ASC,
               min_slot_type ASC, initial_inventory_boxes DESC]
  3. Pass 1+2 – single greedy pass (constrained items first via sort order):
       while remaining_boxes > 0:
         • Scan existing pods in order; skip pods already containing this item.
         • If a free compatible slot is found → assign min(remaining, capacity).
         • If no slot found and pods < TOTAL_PODS → open a new pod.
         • If pod cap reached → stop; remaining inventory is unplaced.
  4. Pass 3 – secondary fill: for every empty slot across all pods, place the
     highest-priority item not already present in that pod.

Outputs
-------
  sku_assignment_detail.csv – one row per (pod, slot) assignment
  sku_assignment.csv        – one row per SKU with totals + boxes_unplaced
  pod_summary.csv           – one row per pod
  unassigned_skus.csv       – items whose inventory could not be fully placed
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
TOTAL_PODS      = 420
MAX_POD_WEIGHT  = 1300.0   # kg
LEAD_TIME       = 1.0      # days
SERVICE_LEVEL_Z = 1.2816   # 90 % service level

# Cluster priority: index 0 = highest priority
# Cluster 3 = high-demand, assigned first and dispersed
CLUSTER_PRIORITY = [0, 2, 3, 1]  # low-demand (0, 2) first, then high-demand (3, 1)

# Pod slot structure (Table 3.6, line 2 of paper)
# Each new pod is created with exactly these slot_types in this order.
POD_STRUCTURE = [0] * 10  # pod_type=0: 10 slots, all slot_type=0

# Constrained = num_compatible_slots <= this threshold (paper line 18)
CONSTRAINED_THRESHOLD = 2

# =============================================================================
# I/O PATHS
# =============================================================================
BASE        = Path(__file__).parent
PROJECT_ROOT = BASE.parent

ITEMS_SLOTS_CONFIG  = PROJECT_ROOT / "items_slots_configuration.csv"
PODS_DICT_FILE      = PROJECT_ROOT / "pods_dictionary.csv"
SKU_SAMPLE_FILE     = PROJECT_ROOT / "sku_sample.csv"
ITEMS_DICT_FILE     = PROJECT_ROOT / "items_dictionary.csv"

OUT_DETAIL          = BASE / "sku_assignment_detail.csv"
OUT_SKU_ASSIGNMENT  = BASE / "sku_assignment.csv"
OUT_POD_SUMMARY     = BASE / "pod_summary.csv"
OUT_UNASSIGNED      = BASE / "unassigned_skus.csv"


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


def load_items_dict(path: Path) -> pd.DataFrame:
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
def load_config(path: Path, pods_dict_path: Path = None) -> pd.DataFrame:
    """
    Load items_slots_configuration.csv.
    Keeps only the BEST orientation per (item_code, slot_type).

    Handles the project's existing column naming:
      max_box_in_slot  → renamed to max_boxes_in_slot
      max_item_in_slot → renamed to max_items_in_slot

    Slot dimensions (slot_length/width/height) are joined from
    pods_dictionary.csv when available; otherwise default to 0.
    """
    df = normalize_cols(read_csv_auto_sep(path))

    # Normalise column names from the project's existing file
    df = df.rename(columns={
        "max_box_in_slot":  "max_boxes_in_slot",
        "max_item_in_slot": "max_items_in_slot",
    })

    needed = {"item_code", "slot_type", "max_boxes_in_slot"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"items_slots_configuration.csv missing columns: {missing}")

    df["item_code"]         = df["item_code"].astype(str)
    df["slot_type"]         = pd.to_numeric(df["slot_type"],         errors="coerce")
    df["max_boxes_in_slot"] = pd.to_numeric(df["max_boxes_in_slot"], errors="coerce").fillna(0)
    df["vol_util_slot"]     = pd.to_numeric(df.get("vol_util_slot",  pd.Series(0.0, index=df.index)),
                                            errors="coerce").fillna(0.0)
    df["orientation"]       = pd.to_numeric(df.get("orientation",    pd.Series(-1,  index=df.index)),
                                            errors="coerce").fillna(-1).astype(int)
    df["box_weight"]        = pd.to_numeric(df.get("box_weight",     pd.Series(0.0, index=df.index)),
                                            errors="coerce").fillna(0.0)
    df["units_per_box"]     = pd.to_numeric(df.get("number_of_item_in_a_box", pd.Series(1.0, index=df.index)),
                                            errors="coerce").fillna(1.0)

    # Join slot dimensions from pods_dictionary for vol_util accuracy
    _pods_dict_p = pods_dict_path if pods_dict_path else path.parent / "pods_dictionary.csv"
    if _pods_dict_p.exists():
        pods_dict = normalize_cols(read_csv_auto_sep(_pods_dict_p))
        pods_dict = pods_dict[["slot_type", "slot_length", "slot_width", "slot_height", "slot_volume"]].drop_duplicates("slot_type")
        pods_dict["slot_type"] = pd.to_numeric(pods_dict["slot_type"], errors="coerce").astype(int)
        # Drop existing slot_volume from config (it may conflict with pods_dict join)
        for col in ("slot_length", "slot_width", "slot_height", "slot_volume"):
            if col in df.columns:
                df = df.drop(columns=[col])
        df = df.merge(pods_dict, on="slot_type", how="left")
        for col in ("slot_length", "slot_width", "slot_height", "slot_volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        # Compute vol_util_slot = (max_boxes * box_volume) / slot_volume
        if "box_volume" in df.columns:
            mask = df["slot_volume"] > 0
            df.loc[mask, "vol_util_slot"] = (
                df.loc[mask, "max_boxes_in_slot"] * df.loc[mask, "box_volume"]
            ) / df.loc[mask, "slot_volume"]
            df["vol_util_slot"] = df["vol_util_slot"].clip(upper=1.0).fillna(0.0)
        used_vol_col = "box_volume"
    else:
        for col in ("slot_length", "slot_width", "slot_height"):
            df[col] = 0.0

    # used_volume_in_slot: how much volume one box occupies * max_boxes
    if "used_volume_in_slot" not in df.columns:
        if "box_volume" in df.columns:
            df["used_volume_in_slot"] = df["max_boxes_in_slot"] * df["box_volume"]
        else:
            df["used_volume_in_slot"] = 0.0
    df["used_volume_in_slot"] = pd.to_numeric(df["used_volume_in_slot"], errors="coerce").fillna(0.0)

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
    """Load sku_sample.csv – primary source for SKUs, demand stats, and box dimensions."""
    df = normalize_cols(read_csv_auto_sep(path))
    df["item_code"] = df["item_code"].astype(str)
    for c in ["mean_demand", "std_demand", "cv_demand"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    if "std_demand" not in df.columns:
        df["std_demand"] = df.get("cv_demand", 0.0) * df["mean_demand"]
    df["box_weight"]    = pd.to_numeric(df["box_weight"],               errors="coerce").fillna(0.0)
    df["units_per_box"] = pd.to_numeric(df["number_of_item_in_a_box"],  errors="coerce").fillna(1.0).clip(lower=1.0)
    if "initial_box_inventory" in df.columns:
        df["initial_inventory_boxes"] = pd.to_numeric(df["initial_box_inventory"], errors="coerce").fillna(0).clip(lower=0).astype(int)
    else:
        df["initial_inventory_boxes"] = 0
    return df


def build_item_table(cluster_df: pd.DataFrame, config_df: pd.DataFrame,
                     items_dict_df: pd.DataFrame) -> pd.DataFrame:
    df = cluster_df.copy()
    df["box_weight"]    = df["box_weight"].fillna(0.0)
    df["units_per_box"] = df["units_per_box"].fillna(1.0).clip(lower=1.0)

    if "initial_inventory_boxes" in df.columns:
        df = df.drop(columns=["initial_inventory_boxes"])
    df = df.merge(
        items_dict_df[["item_code", "qty_inv_items", "initial_inventory_boxes"]],
        on="item_code", how="left"
    )
    df["qty_inv_items"]           = df["qty_inv_items"].fillna(0.0)
    df["initial_inventory_boxes"] = df["initial_inventory_boxes"].fillna(0).astype(int)

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
            "slot_volume":         slot_vol if slot_vol > 0 else float(r.get("slot_volume", 0)),
        }
    return lookup


# =============================================================================
# POD MANAGEMENT
# =============================================================================
def make_pod(pod_id: int) -> dict:
    slots = [
        {
            "slot_idx":  idx,
            "slot_type": st,
            "item_code": None,
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
        "class_set":    set(),
    }


def select_best_config(cfg: dict, assign_boxes: int) -> dict:
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
    detail_rows = []
    preferred_types = sorted(compatible_slot_types, reverse=True)

    while remaining_boxes > 0:
        chosen_pod    = None
        chosen_slot   = None
        chosen_cfg    = None
        chosen_boxes  = 0
        chosen_weight = 0.0

        for pod in pods:
            items_in_pod = {s["item_code"] for s in pod["slots"] if s["item_code"] is not None}
            if item_code in items_in_pod:
                continue
            for st in preferred_types:
                for slot in pod["slots"]:
                    if slot["item_code"] is not None:
                        continue
                    if slot["slot_type"] != st:
                        continue
                    cfg = lookup.get((item_code, st))
                    if cfg is None:
                        continue
                    assign_boxes = cfg["max_boxes_in_slot"]
                    if assign_boxes <= 0:
                        continue
                    added_weight = assign_boxes * box_weight
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

        if chosen_slot is None:
            if next_pod_id_ref[0] >= TOTAL_PODS:
                break
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
                    assign_boxes = cfg["max_boxes_in_slot"]
                    if assign_boxes <= 0:
                        continue
                    added_weight = assign_boxes * box_weight
                    chosen_pod    = new_pod
                    chosen_slot   = slot
                    chosen_cfg    = select_best_config(cfg, assign_boxes)
                    chosen_boxes  = assign_boxes
                    chosen_weight = added_weight
                    break
                if chosen_slot is not None:
                    break

            if chosen_slot is None:
                break

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
    extra_rows = []

    total_empty = sum(1 for pod in pods for slot in pod["slots"] if slot["item_code"] is None)
    print(f"    [secondary_fill] pods={len(pods)}, total_empty_slots={total_empty}")

    slot_type_candidates: dict = {}
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

    for st in slot_type_candidates:
        slot_type_candidates[st].sort(key=lambda x: (x[0], x[1]))

    print(f"    [secondary_fill] candidate slot types: { {st: len(v) for st,v in slot_type_candidates.items()} }")

    for pod in pods:
        items_in_pod = {s["item_code"] for s in pod["slots"] if s["item_code"] is not None}
        for slot in pod["slots"]:
            if slot["item_code"] is not None:
                continue

            st = slot["slot_type"]
            candidates = slot_type_candidates.get(st, [])

            chosen = None
            for rank, neg_vu, item_code, info, cfg in candidates:
                if item_code in items_in_pod:
                    continue
                assign_boxes = max(cfg["max_boxes_in_slot"], 1)
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
            items_in_pod.add(item_code)

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

    pod_rows = []
    for pod in pods:
        used  = [s for s in pod["slots"] if s["item_code"] is not None]
        total = len(pod["slots"])
        pod_rows.append({
            "pod_id":       pod["pod_id"],
            "used_slots":   len(used),
            "total_slots":  total,
            "empty_slots":  total - len(used),
            "total_boxes":  sum(s["qty_boxes"] for s in used),
            "total_items":  sum(s["qty_items"] for s in used),
            "total_weight": pod["total_weight"],
            "num_classes":  len(pod["class_set"]),
            "classes":      "|".join(str(c) for c in sorted(pod["class_set"])),
        })
    pod_summary_df = pd.DataFrame(pod_rows)

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

    sku_df["boxes_unplaced"] = (
        sku_df["initial_inventory_boxes"] - sku_df["total_boxes_assigned"]
    ).clip(lower=0).astype(int)

    unassigned_df = pd.DataFrame(unassigned) if unassigned else pd.DataFrame()

    return detail_df, sku_df, pod_summary_df, unassigned_df


# =============================================================================
# MAIN
# =============================================================================
def run_assignment(base_dir=None, sku_sample_path=None,
                   items_dict_path=None, items_slots_config_path=None):
    base         = Path(base_dir) if base_dir else BASE
    project_root = base.parent
    sku_sample_p  = Path(sku_sample_path)  if sku_sample_path  else project_root / "sku_sample.csv"
    items_dict_p  = Path(items_dict_path)  if items_dict_path  else project_root / "items_dictionary.csv"
    items_slots_p = Path(items_slots_config_path) if items_slots_config_path else project_root / "items_slots_configuration.csv"
    pods_dict_p   = project_root / "pods_dictionary.csv"

    _t0 = time.perf_counter()
    print("=" * 62)
    print("SKU-to-Pod Assignment  (DC6 / ABC-XYZ Greedy FFD)")
    print("=" * 62)

    config_df = load_config(items_slots_p, pods_dict_path=pods_dict_p)
    print(f"  {len(config_df)} best-config rows  ({config_df['item_code'].nunique()} unique SKUs)")

    cluster_df   = load_sku_sample(sku_sample_p)
    sample_codes = set(cluster_df["item_code"].unique())
    print(f"  {len(cluster_df)} SKUs in sample")

    config_df = config_df[config_df["item_code"].isin(sample_codes)].copy()

    items_dict_df = cluster_df[["item_code", "initial_inventory_boxes"]].copy()
    items_dict_df = items_dict_df[items_dict_df["item_code"].isin(sample_codes)].copy()
    items_dict_df["qty_inv_items"] = 0
    total_boxes = items_dict_df["initial_inventory_boxes"].sum()
    print(f"  Total initial inventory: {total_boxes:,} boxes")

    items_df = build_item_table(cluster_df, config_df, items_dict_df)

    # Exclude SKUs that physically cannot fit in any slot (box too large)
    no_config = items_df[items_df["num_compatible_slots"] == 0].copy()
    if len(no_config) > 0:
        print(f"  Excluded {len(no_config)} SKUs — box dimensions exceed all slot sizes.")
        excluded_path = BASE / "excluded_skus.csv"
        no_config[["item_code", "cluster", "initial_inventory_boxes",
                    "box_weight", "units_per_box"]].to_csv(excluded_path, index=False)
        print(f"  Saved to {excluded_path.name}")
        items_df = items_df[items_df["num_compatible_slots"] > 0].copy()

    no_inv = items_df[items_df["initial_inventory_boxes"] == 0]
    if len(no_inv) > 0:
        print(f"  [WARN] {len(no_inv)} SKUs have zero initial inventory – skipped.")

    priority_order = {cl: i + 1 for i, cl in enumerate(CLUSTER_PRIORITY)}
    print(f"\nCluster distribution (processed in order: {CLUSTER_PRIORITY}):")
    for cl in sorted(items_df["cluster"].unique()):
        n     = (items_df["cluster"] == cl).sum()
        order = priority_order.get(cl, "?")
        print(f"  cluster {cl}: {n:>5} SKUs  (processed {order}{['st','nd','rd','th','th'][min(order-1,4)]})")

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

    lookup = build_lookup(config_df)

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

    print("\n[Pass 1+2] Assigning all items (constrained first, then flexible) …")
    run_pass(items_df)
    print(f"  Pods created       : {next_pod_id_ref[0]}")

    print("[Pass 3] Secondary fill of empty slots …")
    extra = secondary_fill(pods, items_df, lookup)
    all_detail_rows.extend(extra)
    print(f"  Extra assignments   : {len(extra)}")

    detail_df, sku_df, pod_summary_df, unassigned_df = build_outputs(
        all_detail_rows, pods, items_df, unassigned_list
    )

    detail_df.to_csv(OUT_DETAIL, index=False)
    sku_df.to_csv(OUT_SKU_ASSIGNMENT, index=False)
    pod_summary_df.to_csv(OUT_POD_SUMMARY, index=False)
    unassigned_df.to_csv(OUT_UNASSIGNED, index=False)

    runtime_sec = time.perf_counter() - _t0

    eligible            = items_df[
        (items_df["num_compatible_slots"] > 0) &
        (items_df["initial_inventory_boxes"] > 0)
    ]
    total_skus_eligible = len(eligible)
    assigned_skus       = int((sku_df["total_boxes_assigned"] > 0).sum())
    active_pods         = len(pods)

    slot_vols = {st: 0.0 for st in set(POD_STRUCTURE)}
    for st in POD_STRUCTURE:
        sample = config_df[config_df["slot_type"] == st]
        if not sample.empty:
            r = sample.iloc[0]
            sv = float(r["slot_length"]) * float(r["slot_width"]) * float(r["slot_height"])
            slot_vols[st] = sv if sv > 0 else float(r.get("slot_volume", 0))

    total_slot_volume = sum(slot_vols[st] for st in POD_STRUCTURE) * active_pods
    used_volume = 0.0
    if not detail_df.empty and "vol_util_slot" in detail_df.columns:
        for _, dr in detail_df.iterrows():
            st  = int(dr["slot_type"])
            vu  = float(dr["vol_util_slot"])
            used_volume += vu * slot_vols.get(st, 0.0)
    space_util_pct = (used_volume / total_slot_volume * 100) if total_slot_volume > 0 else 0.0

    avg_skus_per_pod    = detail_df.groupby("pod_id")["item_code"].nunique().mean() if not detail_df.empty else 0.0
    avg_classes_per_pod = pod_summary_df["num_classes"].mean() if not pod_summary_df.empty else 0.0
    items_with_unplaced = int((sku_df["boxes_unplaced"] > 0).sum())
    total_slots_used    = int(sku_df["slots_used"].sum())
    total_slots_avail   = TOTAL_PODS * len(POD_STRUCTURE)

    print(f"\n  Pods: {active_pods}/{TOTAL_PODS}, SKUs assigned: {assigned_skus}/{total_skus_eligible}, "
          f"space util: {space_util_pct:.1f}%, runtime: {runtime_sec:.1f}s")
    print(f"  Slots used: {total_slots_used} / {total_slots_avail} total")

    if items_with_unplaced > 0:
        print(f"  Items with unplaced inventory: {items_with_unplaced}")

    print(f"\n  {'cluster':>7}  {'items':>6}  {'pods_used':>9}  {'slots_used':>10}  "
          f"{'boxes_placed':>12}  {'boxes_total':>11}  {'boxes_unplaced':>14}")
    for cl in CLUSTER_PRIORITY:
        sub = sku_df[sku_df["cluster"] == cl]
        if sub.empty:
            continue
        print(f"  {cl:>7}  {len(sub):>6}  {int(sub['pods_used'].sum()):>9}  "
              f"{int(sub['slots_used'].sum()):>10}  "
              f"{int(sub['total_boxes_assigned'].sum()):>12}  "
              f"{int(sub['initial_inventory_boxes'].sum()):>11}  "
              f"{int(sub['boxes_unplaced'].sum()):>14}")

    # Per-SKU assignment summary (item_code level)
    print(f"\n  Per-SKU summary (saved to {OUT_SKU_ASSIGNMENT.name}):")
    print(f"  {'item_code':>15}  {'cluster':>7}  {'pods_used':>9}  {'slots_used':>10}  "
          f"{'boxes_placed':>12}  {'boxes_total':>11}  {'unplaced':>8}")
    display_df = sku_df[["item_code", "cluster", "pods_used", "slots_used",
                          "total_boxes_assigned", "initial_inventory_boxes",
                          "boxes_unplaced"]].copy()
    display_df = display_df.sort_values(["cluster", "boxes_unplaced"],
                                        ascending=[True, False])
    for _, r in display_df.head(20).iterrows():
        flag = " !" if r["boxes_unplaced"] > 0 else ""
        print(f"  {str(r['item_code']):>15}  {int(r['cluster']):>7}  "
              f"{int(r['pods_used']):>9}  {int(r['slots_used']):>10}  "
              f"{int(r['total_boxes_assigned']):>12}  "
              f"{int(r['initial_inventory_boxes']):>11}  "
              f"{int(r['boxes_unplaced']):>8}{flag}")
    if len(display_df) > 20:
        print(f"  … ({len(display_df) - 20} more rows in {OUT_SKU_ASSIGNMENT.name})")

    return detail_df


def main():
    run_assignment()


if __name__ == "__main__":
    main()
