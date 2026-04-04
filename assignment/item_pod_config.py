import pandas as pd
import numpy as np
from pathlib import Path
from math import ceil


# =============================================================================
# I/O PATHS
# =============================================================================
BASE = Path(__file__).parent

POD_SIZE_PATH   = BASE / "../pod_size.csv"       # slot_type, slot_length/width/height, slots_per_pod
ITEMS_DICT_PATH = BASE / "../sku_sample.csv"

OUT_ALL_CONFIG  = BASE / "items_slots_configuration.csv"
OUT_BEST_CONFIG = BASE / "best_config_per_sku.csv"


# =============================================================================
# HELPERS
# =============================================================================
def read_csv_auto_sep(path: Path) -> pd.DataFrame:
    """Try semicolon first, fall back to comma."""
    df = pd.read_csv(path, sep=";")
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=",")
    return df


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and BOM characters from column names."""
    return df.rename(columns={c: c.strip().replace("\ufeff", "") for c in df.columns})


def parse_dot_comma_float(val) -> float:
    """
    Robustly parses item dimension values that arrive in locale-dependent or
    multi-dot formats from items_dictionary_cleaned.csv.

    Handled formats
    ---------------
    "45.00.00"      -> 45.0   (multi-dot, keep first two segments)
    "0,395138889"   -> 0.395  (comma decimal separator)
    "24.00"         -> 24.0   (normal float string)
    "45"            -> 45.0   (plain integer string)
    """
    s = str(val).strip().replace(",", ".")
    parts = s.split(".")
    if len(parts) == 1:
        try:
            return float(parts[0])
        except ValueError:
            return np.nan
    try:
        return float(parts[0] + "." + parts[1])
    except ValueError:
        return np.nan


def oriented_dims(l0: float, w0: float, h0: float, d: int):
    """
    Returns the (length, width, height) of the box after applying orientation d.

    Six orientations as defined in the paper (Fig 3.6 / Eq 3.1):
        d=0  (a): l, w, h   — default
        d=1  (b): l, h, w
        d=2  (c): h, l, w
        d=3  (d): h, w, l
        d=4  (e): w, h, l
        d=5  (f): w, l, h
    """
    if d == 0: return l0, w0, h0
    if d == 1: return l0, h0, w0
    if d == 2: return h0, l0, w0
    if d == 3: return h0, w0, l0
    if d == 4: return w0, h0, l0
    if d == 5: return w0, l0, h0
    raise ValueError(f"Orientation d must be 0..5, got {d}")


# =============================================================================
# LOAD & VALIDATE POD SIZE
# =============================================================================
def load_pod_size(path: Path, pod_type: int = None) -> pd.DataFrame:
    """
    Loads slot size configuration.

    Two modes
    ---------
    pod_type=None  : legacy pod_size.csv — each row is one slot type
    pod_type=<int> : pods_dictionary.csv — filter to the given pod_type,
                     derive unique slot types and slots_per_pod from row counts

    Output columns (both modes)
    ---------------------------
    slot_type, slot_length, slot_width, slot_height, slot_volume, slots_per_pod
    """
    df = read_csv_auto_sep(path)
    df = normalize_cols(df)

    if pod_type is not None:
        # pods_dictionary.csv mode: filter to requested pod_type then collapse
        if "pod_type" not in df.columns:
            raise ValueError(f"pods_dictionary missing 'pod_type' column. Found: {list(df.columns)}")
        df["pod_type"] = pd.to_numeric(df["pod_type"], errors="coerce")
        df = df[df["pod_type"] == pod_type].copy()
        if df.empty:
            raise ValueError(f"No rows found for pod_type={pod_type} in {path}")
        # slots_per_pod = count of each slot_type within this pod_type
        slots_per_pod = df.groupby("slot_type").size().rename("slots_per_pod").reset_index()
        df = df.drop_duplicates("slot_type").merge(slots_per_pod, on="slot_type")

    required = {"slot_type", "slot_length", "slot_width", "slot_height"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}. Found: {list(df.columns)}")

    for c in ["slot_length", "slot_width", "slot_height"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["slot_type"] = pd.to_numeric(df["slot_type"], errors="coerce").astype("Int64")

    if "slots_per_pod" in df.columns:
        df["slots_per_pod"] = pd.to_numeric(df["slots_per_pod"], errors="coerce")
    else:
        df["slots_per_pod"] = np.nan

    df = df.dropna(subset=["slot_type", "slot_length", "slot_width", "slot_height"]).copy()
    df["slot_volume"] = df["slot_length"] * df["slot_width"] * df["slot_height"]

    print(f"[pod_size] Loaded {len(df)} slot type(s):")
    for _, r in df.iterrows():
        spp = f"{int(r['slots_per_pod'])}" if pd.notna(r["slots_per_pod"]) else "N/A"
        print(
            f"  slot_type {int(r['slot_type'])}: "
            f"{r['slot_length']} x {r['slot_width']} x {r['slot_height']} "
            f"(vol={r['slot_volume']:.1f})  slots_per_pod={spp}"
        )
    return df


# =============================================================================
# LOAD & VALIDATE ITEMS
# =============================================================================
def load_items(path: Path) -> pd.DataFrame:
    """
    Loads items_dictionary_cleaned.csv and applies robust dimension parsing.

    Required columns : item_code, box_length, box_width, box_height
    Optional columns : box_weight, box_volume, number_of_item_in_a_box
    """
    df = read_csv_auto_sep(path)
    df = normalize_cols(df)

    required = {"item_code", "box_length", "box_width", "box_height"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"items_dictionary_cleaned missing columns: {missing}. Found: {list(df.columns)}"
        )

    df["item_code"] = df["item_code"].astype(str)

    # Dimension columns use robust parser (handles "45.00.00" and "0,395" formats)
    for c in ["box_length", "box_width", "box_height"]:
        df[c] = df[c].apply(parse_dot_comma_float)

    # Optional numeric columns
    if "box_weight" not in df.columns:
        df["box_weight"] = 0.0
    else:
        df["box_weight"] = df["box_weight"].apply(parse_dot_comma_float).fillna(0.0)

    if "number_of_item_in_a_box" not in df.columns:
        df["number_of_item_in_a_box"] = 1.0
    else:
        df["number_of_item_in_a_box"] = (
            df["number_of_item_in_a_box"].apply(parse_dot_comma_float).fillna(1.0)
        )

    # Compute box_volume from cleaned dimensions (authoritative)
    df["box_volume"] = df["box_length"] * df["box_width"] * df["box_height"]

    before = len(df)
    df = df.dropna(subset=["box_length", "box_width", "box_height"]).copy()
    df = df[(df["box_length"] > 0) & (df["box_width"] > 0) & (df["box_height"] > 0)].copy()
    after = len(df)

    if before != after:
        print(f"[items] Dropped {before - after} rows with invalid/zero dimensions.")
    print(f"[items] {after} items loaded with valid dimensions.")
    return df


# =============================================================================
# GENERATE FEASIBLE CONFIGURATIONS  (Table 3.4 in paper)
# =============================================================================
def generate_configs(items_df: pd.DataFrame, slot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Brute-force grid placement for every (item, slot_type, orientation) triple.

    For each combination the box is tested in all 6 orientations.  If it fits
    (oriented dimensions <= slot dimensions on every axis), the maximum number
    of boxes that can be packed into the slot without overlap is computed as:

        max_boxes_in_slot = floor(Lc / li) * floor(Wc / wi) * floor(Hc / hi)

    Only combinations with max_boxes_in_slot >= 1 are kept.

    If slots_per_pod is available in pod_size.csv, the pod-level capacity is
    also computed:

        max_boxes_in_pod  = max_boxes_in_slot  * slots_per_pod
        max_items_in_pod  = max_items_in_slot  * slots_per_pod

    When slots_per_pod is NaN these two columns will be NaN.
    """
    slot_rows = slot_df.to_dict(orient="records")
    orientations = list(range(6))
    records = []

    for _, it in items_df.iterrows():
        item_code    = str(it["item_code"])
        l0           = float(it["box_length"])
        w0           = float(it["box_width"])
        h0           = float(it["box_height"])
        v0           = float(it["box_volume"])
        wt0          = float(it["box_weight"])
        units_per_box = float(it["number_of_item_in_a_box"])

        if l0 <= 0 or w0 <= 0 or h0 <= 0:
            continue

        for s in slot_rows:
            slot_type    = int(s["slot_type"])
            Lc           = float(s["slot_length"])
            Wc           = float(s["slot_width"])
            Hc           = float(s["slot_height"])
            slot_vol     = float(s["slot_volume"])
            slots_per_pod = s["slots_per_pod"]  # may be NaN

            for d in orientations:
                li, wi, hi = oriented_dims(l0, w0, h0, d)

                # Constraint 3.10: oriented box must fit entirely inside slot
                if not (li <= Lc and wi <= Wc and hi <= Hc):
                    continue

                # Grid placement (Table 3.4, lines 9-11)
                max_x = int(np.floor(Lc / li)) if li > 0 else 0
                max_y = int(np.floor(Wc / wi)) if wi > 0 else 0
                max_z = int(np.floor(Hc / hi)) if hi > 0 else 0

                max_boxes_in_slot = max_x * max_y * max_z
                if max_boxes_in_slot <= 0:
                    continue

                max_items_in_slot = max_boxes_in_slot * units_per_box

                # Pod-level capacity (only if slots_per_pod is known)
                if pd.notna(slots_per_pod) and slots_per_pod > 0:
                    spp = int(slots_per_pod)
                    max_boxes_in_pod = max_boxes_in_slot * spp
                    max_items_in_pod = max_items_in_slot * spp
                else:
                    max_boxes_in_pod = np.nan
                    max_items_in_pod = np.nan

                used_volume_in_slot  = max_boxes_in_slot * v0
                used_weight_in_slot  = max_boxes_in_slot * wt0
                vol_util_slot        = used_volume_in_slot / slot_vol if slot_vol > 0 else 0.0

                records.append({
                    "item_code":   item_code,
                    "slot_type":   slot_type,
                    "orientation": d,

                    # Slot dimensions
                    "slot_length":   Lc,
                    "slot_width":    Wc,
                    "slot_height":   Hc,
                    "slot_volume":   slot_vol,
                    "slots_per_pod": spp if pd.notna(slots_per_pod) else np.nan,

                    # Original box dimensions
                    "box_l0":     l0,
                    "box_w0":     w0,
                    "box_h0":     h0,
                    "box_weight": wt0,
                    "box_volume": v0,
                    "units_per_box": units_per_box,

                    # Oriented box dimensions (after rotation d)
                    "oriented_l": li,
                    "oriented_w": wi,
                    "oriented_h": hi,

                    # Grid counts per axis
                    "max_x": max_x,
                    "max_y": max_y,
                    "max_z": max_z,

                    # Slot-level capacity
                    "max_boxes_in_slot": max_boxes_in_slot,
                    "max_items_in_slot": max_items_in_slot,

                    # Pod-level capacity (NaN if slots_per_pod unknown)
                    "max_boxes_in_pod": max_boxes_in_pod,
                    "max_items_in_pod": max_items_in_pod,

                    # Volume / weight utilisation
                    "used_volume_in_slot":  used_volume_in_slot,
                    "used_weight_in_slot":  used_weight_in_slot,
                    "vol_util_slot":        vol_util_slot,
                })

    return pd.DataFrame(records)


# =============================================================================
# BEST CONFIGURATION PER SKU
# =============================================================================
def pick_best_config(config_df: pd.DataFrame) -> pd.DataFrame:
    """
    Selects the single best (item_code, slot_type, orientation) row per SKU.

    Ranking criteria (descending):
      1. max_items_in_pod  — maximise total unit capacity across the pod
                             (falls back to max_items_in_slot when slots_per_pod
                              is NaN, since they are proportional)
      2. vol_util_slot     — prefer orientations that fill the slot more completely
      3. max_boxes_in_slot — tiebreaker on raw box count
    """
    sort_cols = ["max_boxes_in_slot", "vol_util_slot"]

    # Use pod-level capacity as primary sort only when it is available
    if config_df["max_items_in_pod"].notna().any():
        sort_cols = ["max_items_in_pod"] + sort_cols
    else:
        sort_cols = ["max_items_in_slot"] + sort_cols

    best_df = (
        config_df.sort_values(
            by=["item_code"] + sort_cols,
            ascending=[True] + [False] * len(sort_cols),
        )
        .groupby("item_code", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    return best_df


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("item_pod_config  —  items-to-slots configuration generator")
    print("=" * 60)

    # ── Load inputs ───────────────────────────────────────────────
    slot_df  = load_pod_size(POD_SIZE_PATH)
    items_df = load_items(ITEMS_DICT_PATH)

    print(f"\nSlot types  : {sorted(slot_df['slot_type'].astype(int).tolist())}")
    print(f"SKUs        : {len(items_df)}")
    print(f"Orientations: 6 (d=0..5 per paper Fig 3.6)")

    # ── Generate all feasible configs ────────────────────────────
    print("\nGenerating feasible (SKU × slot_type × orientation) configurations …")
    config_df = generate_configs(items_df, slot_df)

    if config_df.empty:
        print(
            "\n[ERROR] No feasible configurations found.\n"
            "Check that box dimensions are smaller than at least one slot dimension."
        )
        return

    # ── Coverage report ──────────────────────────────────────────
    covered_skus   = config_df["item_code"].nunique()
    uncovered_skus = len(items_df) - covered_skus
    print(f"\nCoverage summary:")
    print(f"  Total SKUs            : {len(items_df)}")
    print(f"  SKUs with ≥1 config   : {covered_skus}")
    print(f"  SKUs with NO config   : {uncovered_skus}  "
          f"(box too large for every slot in every orientation)")
    print(f"  Total config rows     : {len(config_df)}")

    print("\nFeasible rows per slot_type:")
    per_slot = config_df.groupby("slot_type").agg(
        rows=("item_code", "count"),
        skus=("item_code", "nunique"),
        avg_max_boxes=("max_boxes_in_slot", "mean"),
        avg_vol_util=("vol_util_slot", "mean"),
    )
    print(per_slot.to_string())

    # ── Save full config ─────────────────────────────────────────
    config_df.to_csv(OUT_ALL_CONFIG, index=False)
    print(f"\nSaved all feasible configurations → {OUT_ALL_CONFIG.resolve()}")

    # ── Pick & save best config per SKU ─────────────────────────
    best_df = pick_best_config(config_df)
    best_df.to_csv(OUT_BEST_CONFIG, index=False)
    print(f"Saved best config per SKU         → {OUT_BEST_CONFIG.resolve()}")
    print(f"Best configs: {len(best_df)}")

    print("\nPreview — best_config_per_sku (top 10):")
    preview_cols = [
        "item_code", "slot_type", "orientation",
        "max_boxes_in_slot", "max_items_in_slot",
        "max_boxes_in_pod",  "max_items_in_pod",
        "vol_util_slot",
    ]
    print(best_df[preview_cols].head(10).to_string(index=False))

    # ── Slot-type distribution in best config ────────────────────
    print("\nBest slot_type distribution across all SKUs:")
    print(best_df["slot_type"].value_counts().sort_index().to_string())

    # ── Warn about uncovered SKUs ────────────────────────────────
    if uncovered_skus > 0:
        covered_set  = set(config_df["item_code"])
        all_items    = set(items_df["item_code"])
        uncovered    = all_items - covered_set
        print(f"\n[WARNING] {uncovered_skus} SKUs could not fit in any slot/orientation:")
        sample = list(uncovered)[:20]
        for code in sample:
            r = items_df[items_df["item_code"] == code].iloc[0]
            print(
                f"  {code}  box=({r['box_length']:.1f} x {r['box_width']:.1f} x {r['box_height']:.1f})"
            )
        if uncovered_skus > 20:
            print(f"  … and {uncovered_skus - 20} more.")

    print("\nDone.")


if __name__ == "__main__":
    main()