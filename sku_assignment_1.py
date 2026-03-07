import pandas as pd
import numpy as np
from pathlib import Path
from math import ceil, sqrt

# =====================================
# PARAMETERS
# =====================================
TOTAL_PODS = 300
MAX_POD_WEIGHT = 1300.0
LEAD_TIME = 1.0
SERVICE_LEVEL_Z = 1.2816  # 90%
CLUSTER_PRIORITY = [4, 1, 3, 0, 2]


# =====================================
# HELPERS
# =====================================
def read_csv_auto_sep(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=",")
    return df


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: c.strip().replace("\ufeff", "") for c in df.columns})


def parse_dot_comma_float(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def compute_initial_inventory_items(mu, sigma, L=1.0, z=1.2816):
    mu = 0.0 if pd.isna(mu) else float(mu)
    sigma = 0.0 if pd.isna(sigma) else float(sigma)
    return int(ceil(max(mu * L + z * sigma * sqrt(L), 0.0)))


def compute_initial_inventory_boxes(initial_inventory_items, items_per_box):

    if pd.isna(items_per_box) or items_per_box <= 0:
        items_per_box = 1

    if initial_inventory_items <= 0:
        return 0

    boxes = int(ceil(float(initial_inventory_items) / float(items_per_box)))
    return boxes

def cluster_rank(cluster_value):
    try:
        return CLUSTER_PRIORITY.index(int(cluster_value))
    except Exception:
        return 999


# =====================================
# POD TEMPLATE FROM pod_size.csv
# =====================================
def build_generic_pod_template(pod_size_df: pd.DataFrame):
    """
    Build generic pod slots from pod_size.csv.
    Required:
    - slot_type
    - slots_per_pod
    """
    pod_size_df = normalize_cols(pod_size_df)

    required = {"slot_type", "slots_per_pod"}
    missing = required - set(pod_size_df.columns)
    if missing:
        raise ValueError(f"pod_size.csv missing columns: {missing}. Found: {list(pod_size_df.columns)}")

    pod_size_df["slot_type"] = pd.to_numeric(pod_size_df["slot_type"], errors="coerce")
    pod_size_df["slots_per_pod"] = pd.to_numeric(pod_size_df["slots_per_pod"], errors="coerce")

    pod_size_df = pod_size_df.dropna(subset=["slot_type", "slots_per_pod"]).copy()
    pod_size_df = pod_size_df.sort_values("slot_type")

    template_slots = []
    slot_counter = 0

    for _, row in pod_size_df.iterrows():
        slot_type = int(row["slot_type"])
        count = int(row["slots_per_pod"])

        for _ in range(count):
            template_slots.append({
                "slot_index": slot_counter,
                "slot_sequence": slot_counter + 1,
                "slot_type": slot_type,
                "item_code": None,
                "qty_boxes": 0,
                "qty_items": 0,
                "orientation": None,
                "cluster": None,
            })
            slot_counter += 1

    return template_slots


def create_pod_pool(template_slots, total_pods=300):
    pods = []
    for pod_id in range(total_pods):
        slots = []
        for s in template_slots:
            slots.append({
                "slot_index": s["slot_index"],
                "slot_sequence": s["slot_sequence"],
                "slot_type": s["slot_type"],
                "item_code": None,
                "qty_boxes": 0,
                "qty_items": 0,
                "orientation": None,
                "cluster": None,
            })

        pods.append({
            "pod_id": pod_id,
            "total_weight": 0.0,
            "slots": slots,
            "class_set": set()
        })
    return pods


# =====================================
# CONFIG LOOKUP FROM items_slots_configuration.csv
# =====================================
def build_config_lookup(config_df: pd.DataFrame):
    """
    Required:
    - item_code
    - slot_type
    - max_boxes_in_slot
    Optional:
    - orientation
    """
    config_df = normalize_cols(config_df)

    required = {"item_code", "slot_type", "max_boxes_in_slot"}
    missing = required - set(config_df.columns)
    if missing:
        raise ValueError(f"items_slots_configuration.csv missing columns: {missing}. Found: {list(config_df.columns)}")

    config_df["item_code"] = config_df["item_code"].astype(str)
    config_df["slot_type"] = pd.to_numeric(config_df["slot_type"], errors="coerce")
    config_df["max_boxes_in_slot"] = pd.to_numeric(config_df["max_boxes_in_slot"], errors="coerce")

    if "orientation" in config_df.columns:
        config_df["orientation"] = pd.to_numeric(config_df["orientation"], errors="coerce").fillna(-1).astype(int)
    else:
        config_df["orientation"] = -1

    config_df = config_df.dropna(subset=["slot_type", "max_boxes_in_slot"]).copy()
    config_df["slot_type"] = config_df["slot_type"].astype(int)
    config_df["max_boxes_in_slot"] = config_df["max_boxes_in_slot"].fillna(0).astype(int)

    # Best config per item_code x slot_type
    best_rows = (
        config_df.sort_values(
            by=["item_code", "slot_type", "max_boxes_in_slot"],
            ascending=[True, True, False],
            kind="mergesort"
        )
        .groupby(["item_code", "slot_type"], as_index=False)
        .head(1)
        .copy()
    )

    config_best = {}
    compatible_slot_types = {}

    for _, r in best_rows.iterrows():
        key = (r["item_code"], int(r["slot_type"]))
        config_best[key] = {
            "max_boxes_in_slot": int(r["max_boxes_in_slot"]),
            "orientation": int(r["orientation"])
        }

    for item_code, g in best_rows.groupby("item_code"):
        compatible_slot_types[str(item_code)] = set(int(x) for x in g["slot_type"].unique())

    return config_best, compatible_slot_types


# =====================================
# FLEXIBILITY / SORTING PREP
# =====================================
def build_flexibility_data(items_df, compatible_slot_types, config_best):
    rows = []

    for _, row in items_df.iterrows():
        item_code = str(row["item_code"])
        compatible = sorted(list(compatible_slot_types.get(item_code, set())))

        if len(compatible) == 0:
            continue

        total_max_capacity = 0
        for st in compatible:
            cfg = config_best.get((item_code, st))
            if cfg:
                total_max_capacity += cfg["max_boxes_in_slot"]

        rows.append({
            "item_code": item_code,
            "num_compatible_slots": len(compatible),
            "min_slot_type": min(compatible),
            "max_slot_type": max(compatible),
            "total_max_capacity": total_max_capacity
        })

    return pd.DataFrame(rows)


# =====================================
# GREEDY PRODUCTS-TO-POD ASSIGNMENT
# =====================================
def greedy_assignment(items_df, pods, config_best, compatible_slot_types):
    """
    Greedy first-fit:
    - iterate sorted SKU
    - try existing pods first
    - assign to first free compatible slot
    - if not found, continue to next pod
    - no new pod creation beyond TOTAL_PODS
    """
    detail_rows = []
    unassigned_rows = []

    # First pass
    for _, row in items_df.iterrows():
        item_code = str(row["item_code"])
        cluster = int(row["cluster"])
        initial_items = int(row["initial_inventory_items"])
        initial_boxes = int(row["initial_inventory_boxes"])
        remaining_boxes = initial_boxes
        remaining_items_equiv = initial_boxes * int(row["number_of_item_in_a_box"])
        box_weight = float(row["box_weight"])
        items_per_box = int(row["number_of_item_in_a_box"])
        mean_demand = float(row["mean_demand"])
        std_demand = float(row["std_demand"])
        cv_demand = float(row["cv_demand"])

        compatible = compatible_slot_types.get(item_code, set())
        if len(compatible) == 0:
            # should not happen because we already filter feasible SKU
            unassigned_rows.append({
                "item_code": item_code,
                "cluster": cluster,
                "initial_inventory_items": initial_items,
                "initial_inventory_boxes": initial_boxes,
                "unassigned_boxes": remaining_boxes,
                "unassigned_items_equivalent": remaining_boxes * items_per_box,
                "reason": "no_config"
            })
            continue

        while remaining_boxes > 0:
            slot_found = False

            for pod in pods:
                if remaining_boxes <= 0:
                    break

                for slot in pod["slots"]:
                    if slot["item_code"] is not None:
                        continue

                    slot_type = int(slot["slot_type"])
                    if slot_type not in compatible:
                        continue

                    cfg = config_best.get((item_code, slot_type))
                    if cfg is None:
                        continue

                    max_boxes_in_slot = int(cfg["max_boxes_in_slot"])
                    if max_boxes_in_slot <= 0:
                        continue

                    assign_boxes = min(remaining_boxes, max_boxes_in_slot)
                    assign_items = assign_boxes * items_per_box
                    added_weight = assign_boxes * box_weight

                    if pod["total_weight"] + added_weight > MAX_POD_WEIGHT:
                        continue

                    # assign to slot
                    slot["item_code"] = item_code
                    slot["qty_boxes"] = assign_boxes
                    slot["qty_items"] = assign_items
                    slot["orientation"] = cfg["orientation"]
                    slot["cluster"] = cluster

                    pod["total_weight"] += added_weight
                    pod["class_set"].add(cluster)

                    remaining_boxes -= assign_boxes
                    remaining_items_equiv -= assign_items
                    slot_found = True

                    detail_rows.append({
                        "pod_id": pod["pod_id"],
                        "slot_index": slot["slot_index"],
                        "slot_sequence": slot["slot_sequence"],
                        "slot_type": slot["slot_type"],
                        "item_code": item_code,
                        "cluster": cluster,
                        "qty_boxes_assigned": assign_boxes,
                        "qty_items_equivalent": assign_items,
                        "box_weight": box_weight,
                        "weight_assigned": added_weight,
                        "orientation": slot["orientation"],
                        "items_per_box": items_per_box,
                        "initial_inventory_items": initial_items,
                        "initial_inventory_boxes": initial_boxes,
                        "mean_demand": mean_demand,
                        "std_demand": std_demand,
                        "cv_demand": cv_demand,
                        "pod_weight_after": pod["total_weight"],
                        "classes_in_pod_after": "|".join(str(x) for x in sorted(pod["class_set"]))
                    })
                    break

                if slot_found:
                    break

            if not slot_found:
                # no more feasible empty slot
                break

        if remaining_boxes > 0:
            unassigned_rows.append({
                "item_code": item_code,
                "cluster": cluster,
                "initial_inventory_items": initial_items,
                "initial_inventory_boxes": initial_boxes,
                "unassigned_boxes": remaining_boxes,
                "unassigned_items_equivalent": remaining_boxes * items_per_box,
                "reason": "no_available_feasible_slot"
            })

    detail_df = pd.DataFrame(detail_rows)
    unassigned_df = pd.DataFrame(unassigned_rows)
    return detail_df, unassigned_df


# =====================================
# SECONDARY FILL PASS
# =====================================
def secondary_fill_pass(detail_df, unassigned_df, pods, items_df, config_best, compatible_slot_types):
    """
    Try again for remaining SKU boxes using free slots.
    Useful if some SKU still have remaining boxes after first pass.
    """
    if unassigned_df.empty:
        return detail_df, unassigned_df

    item_info = items_df.set_index("item_code").to_dict(orient="index")
    new_detail_rows = []
    still_unassigned = []

    for _, u in unassigned_df.iterrows():
        item_code = str(u["item_code"])
        cluster = int(u["cluster"])
        remaining_boxes = int(u["unassigned_boxes"])

        if remaining_boxes <= 0:
            continue

        info = item_info[item_code]
        box_weight = float(info["box_weight"])
        items_per_box = int(info["number_of_item_in_a_box"])
        initial_items = int(info["initial_inventory_items"])
        initial_boxes = int(info["initial_inventory_boxes"])
        mean_demand = float(info["mean_demand"])
        std_demand = float(info["std_demand"])
        cv_demand = float(info["cv_demand"])

        compatible = compatible_slot_types.get(item_code, set())

        while remaining_boxes > 0:
            slot_found = False

            for pod in pods:
                if remaining_boxes <= 0:
                    break

                for slot in pod["slots"]:
                    if slot["item_code"] is not None:
                        continue

                    slot_type = int(slot["slot_type"])
                    if slot_type not in compatible:
                        continue

                    cfg = config_best.get((item_code, slot_type))
                    if cfg is None:
                        continue

                    max_boxes_in_slot = int(cfg["max_boxes_in_slot"])
                    if max_boxes_in_slot <= 0:
                        continue

                    assign_boxes = min(remaining_boxes, max_boxes_in_slot)
                    assign_items = assign_boxes * items_per_box
                    added_weight = assign_boxes * box_weight

                    if pod["total_weight"] + added_weight > MAX_POD_WEIGHT:
                        continue

                    slot["item_code"] = item_code
                    slot["qty_boxes"] = assign_boxes
                    slot["qty_items"] = assign_items
                    slot["orientation"] = cfg["orientation"]
                    slot["cluster"] = cluster

                    pod["total_weight"] += added_weight
                    pod["class_set"].add(cluster)

                    remaining_boxes -= assign_boxes
                    slot_found = True

                    new_detail_rows.append({
                        "pod_id": pod["pod_id"],
                        "slot_index": slot["slot_index"],
                        "slot_sequence": slot["slot_sequence"],
                        "slot_type": slot["slot_type"],
                        "item_code": item_code,
                        "cluster": cluster,
                        "qty_boxes_assigned": assign_boxes,
                        "qty_items_equivalent": assign_items,
                        "box_weight": box_weight,
                        "weight_assigned": added_weight,
                        "orientation": slot["orientation"],
                        "items_per_box": items_per_box,
                        "initial_inventory_items": initial_items,
                        "initial_inventory_boxes": initial_boxes,
                        "mean_demand": mean_demand,
                        "std_demand": std_demand,
                        "cv_demand": cv_demand,
                        "pod_weight_after": pod["total_weight"],
                        "classes_in_pod_after": "|".join(str(x) for x in sorted(pod["class_set"]))
                    })
                    break

                if slot_found:
                    break

            if not slot_found:
                break

        if remaining_boxes > 0:
            still_unassigned.append({
                "item_code": item_code,
                "cluster": cluster,
                "initial_inventory_items": int(info["initial_inventory_items"]),
                "initial_inventory_boxes": int(info["initial_inventory_boxes"]),
                "unassigned_boxes": remaining_boxes,
                "unassigned_items_equivalent": remaining_boxes * int(info["number_of_item_in_a_box"]),
                "reason": "still_no_available_feasible_slot_after_second_pass"
            })

    if new_detail_rows:
        detail_df = pd.concat([detail_df, pd.DataFrame(new_detail_rows)], ignore_index=True)

    final_unassigned_df = pd.DataFrame(still_unassigned)
    return detail_df, final_unassigned_df


# =====================================
# BUILD OUTPUTS
# =====================================
def build_outputs(detail_df, items_df, pods, unassigned_df):
    if detail_df.empty:
        sku_assignment = items_df.copy()
        sku_assignment["assigned_total_boxes"] = 0
        sku_assignment["assigned_total_items_equivalent"] = 0
        sku_assignment["assigned_total_weight"] = 0
        sku_assignment["pods_used"] = 0
        sku_assignment["slots_used"] = 0
    else:
        sku_stats = (
            detail_df.groupby("item_code", as_index=False)
            .agg(
                assigned_total_boxes=("qty_boxes_assigned", "sum"),
                assigned_total_items_equivalent=("qty_items_equivalent", "sum"),
                assigned_total_weight=("weight_assigned", "sum"),
                pods_used=("pod_id", "nunique"),
                slots_used=("slot_index", "count")
            )
        )

        sku_assignment = items_df.merge(sku_stats, on="item_code", how="left")
        for c in [
            "assigned_total_boxes",
            "assigned_total_items_equivalent",
            "assigned_total_weight",
            "pods_used",
            "slots_used"
        ]:
            sku_assignment[c] = sku_assignment[c].fillna(0)

    pod_rows = []
    for pod in pods:
        used_slots = sum(1 for s in pod["slots"] if s["item_code"] is not None)
        total_boxes = sum(int(s["qty_boxes"]) for s in pod["slots"] if s["item_code"] is not None)
        total_items = sum(int(s["qty_items"]) for s in pod["slots"] if s["item_code"] is not None)

        pod_rows.append({
            "pod_id": pod["pod_id"],
            "used_slots": used_slots,
            "total_slots": len(pod["slots"]),
            "slot_utilization": used_slots / len(pod["slots"]) if len(pod["slots"]) > 0 else 0,
            "total_boxes": total_boxes,
            "total_items_equivalent": total_items,
            "total_weight": pod["total_weight"],
            "weight_utilization": pod["total_weight"] / MAX_POD_WEIGHT,
            "num_classes": len(pod["class_set"]),
            "classes_in_pod": "|".join(str(x) for x in sorted(pod["class_set"]))
        })

    pod_summary = pd.DataFrame(pod_rows)
    return sku_assignment, pod_summary, unassigned_df


# =====================================
# MAIN
# =====================================
def main():
    ITEMS_SLOTS_CONFIG = Path("items_slots_configuration.csv")
    SKU_CLUSTER_FILE = Path("sku_kmeans_clusters_cleaned.csv")
    ITEMS_DICT_FILE = Path("items_dictionary.csv")
    POD_SIZE_FILE = Path("pod_size.csv")

    OUT_DETAIL = Path("sku_assignment_detail.csv")
    OUT_SKU_ASSIGNMENT = Path("sku_assignment.csv")
    OUT_POD_SUMMARY = Path("pod_summary.csv")
    OUT_UNASSIGNED = Path("unassigned_skus.csv")

    config_df = normalize_cols(read_csv_auto_sep(ITEMS_SLOTS_CONFIG))
    cluster_df = normalize_cols(read_csv_auto_sep(SKU_CLUSTER_FILE))
    item_dict_df = normalize_cols(read_csv_auto_sep(ITEMS_DICT_FILE))
    pod_size_df = normalize_cols(read_csv_auto_sep(POD_SIZE_FILE))

    # ---------- validate ----------
    required_cluster = {"item_code", "cluster", "mean_demand"}
    missing = required_cluster - set(cluster_df.columns)
    if missing:
        raise ValueError(f"sku_kmeans_clusters_cleaned.csv missing columns: {missing}")

    if "std_demand" not in cluster_df.columns and "cv_demand" not in cluster_df.columns:
        raise ValueError("Need std_demand or cv_demand in sku_kmeans_clusters_cleaned.csv")

    required_item = {"item_code", "box_weight", "number_of_item_in_a_box"}
    missing = required_item - set(item_dict_df.columns)
    if missing:
        raise ValueError(f"items_dictionary.csv missing columns: {missing}")

    # ---------- clean types ----------
    cluster_df["item_code"] = cluster_df["item_code"].astype(str)
    item_dict_df["item_code"] = item_dict_df["item_code"].astype(str)

    for c in ["mean_demand", "std_demand", "cv_demand"]:
        if c in cluster_df.columns:
            cluster_df[c] = pd.to_numeric(cluster_df[c], errors="coerce")

    item_dict_df["box_weight"] = item_dict_df["box_weight"].apply(parse_dot_comma_float).fillna(0.0)
    item_dict_df["number_of_item_in_a_box"] = (
        item_dict_df["number_of_item_in_a_box"]
        .apply(parse_dot_comma_float)
        .replace(0, np.nan)
        .fillna(1.0)
    )

    # ---------- merge ----------
    items_df = cluster_df.merge(
        item_dict_df[["item_code", "box_weight", "number_of_item_in_a_box"]],
        on="item_code",
        how="left"
    )

    if "std_demand" not in items_df.columns:
        items_df["std_demand"] = np.nan
    if "cv_demand" not in items_df.columns:
        items_df["cv_demand"] = np.nan

    mask_std_na = items_df["std_demand"].isna()
    items_df.loc[mask_std_na, "std_demand"] = (
        items_df.loc[mask_std_na, "cv_demand"] * items_df.loc[mask_std_na, "mean_demand"]
    )

    items_df["mean_demand"] = items_df["mean_demand"].fillna(0.0)
    items_df["std_demand"] = items_df["std_demand"].fillna(0.0)
    items_df["cv_demand"] = items_df["cv_demand"].fillna(0.0)
    items_df["box_weight"] = items_df["box_weight"].fillna(0.0)
    items_df["number_of_item_in_a_box"] = items_df["number_of_item_in_a_box"].fillna(1.0)

    # ---------- initial inventory ----------
    items_df["initial_inventory_items"] = items_df.apply(
        lambda r: compute_initial_inventory_items(r["mean_demand"], r["std_demand"], LEAD_TIME, SERVICE_LEVEL_Z),
        axis=1
    )
    items_df["initial_inventory_boxes"] = items_df.apply(
        lambda r: compute_initial_inventory_boxes(
            r["initial_inventory_items"],
            r["number_of_item_in_a_box"]
        ),
        axis=1
    )

    # ---------- template pod ----------
    template_slots = build_generic_pod_template(pod_size_df)
    pods = create_pod_pool(template_slots, TOTAL_PODS)

    # ---------- config lookup ----------
    config_best, compatible_slot_types = build_config_lookup(config_df)

    # ---------- keep only feasible SKU ----------
    feasible_sku = set(config_best.keys())
    feasible_item_codes = set(x[0] for x in feasible_sku)

    items_df = items_df[items_df["item_code"].isin(feasible_item_codes)].copy()

    # ---------- flexibility ----------
    flex_df = build_flexibility_data(items_df, compatible_slot_types, config_best)

    items_df = items_df.merge(flex_df, on="item_code", how="left")

    # ---------- sort according to greedy pseudocode ----------
    items_df["cluster_rank"] = items_df["cluster"].apply(cluster_rank)
    items_df["num_compatible_slots"] = items_df["num_compatible_slots"].fillna(999)
    items_df["min_slot_type"] = items_df["min_slot_type"].fillna(999)

    items_df = items_df.sort_values(
        by=["cluster_rank", "num_compatible_slots", "min_slot_type", "initial_inventory_boxes"],
        ascending=[True, True, True, False],
        kind="mergesort"
    ).reset_index(drop=True)

    # ---------- assignment ----------
    detail_df, unassigned_df = greedy_assignment(
        items_df=items_df,
        pods=pods,
        config_best=config_best,
        compatible_slot_types=compatible_slot_types
    )

    # ---------- second pass ----------
    detail_df, unassigned_df = secondary_fill_pass(
        detail_df=detail_df,
        unassigned_df=unassigned_df,
        pods=pods,
        items_df=items_df,
        config_best=config_best,
        compatible_slot_types=compatible_slot_types
    )

    # ---------- outputs ----------
    sku_assignment_df, pod_summary_df, unassigned_df = build_outputs(
        detail_df=detail_df,
        items_df=items_df,
        pods=pods,
        unassigned_df=unassigned_df
    )

    detail_df.to_csv(OUT_DETAIL, index=False)
    sku_assignment_df.to_csv(OUT_SKU_ASSIGNMENT, index=False)
    pod_summary_df.to_csv(OUT_POD_SUMMARY, index=False)
    unassigned_df.to_csv(OUT_UNASSIGNED, index=False)

    print("\n=== DONE ===")
    print("Saved:", OUT_DETAIL.resolve())
    print("Saved:", OUT_SKU_ASSIGNMENT.resolve())
    print("Saved:", OUT_POD_SUMMARY.resolve())
    print("Saved:", OUT_UNASSIGNED.resolve())

    print("\nCoverage summary:")
    print("  Feasible SKUs used in assignment :", items_df['item_code'].nunique())
    print("  Assigned SKUs                    :", sku_assignment_df[sku_assignment_df['assigned_total_boxes'] > 0]['item_code'].nunique())
    print("  Unassigned SKUs                  :", unassigned_df['item_code'].nunique() if not unassigned_df.empty else 0)
    print("  Total pods                       :", len(pod_summary_df))
    print("  Pods with >=2 classes            :", (pod_summary_df['num_classes'] >= 2).sum())


if __name__ == "__main__":
    main()
