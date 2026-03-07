"""
Extract clustering features from mid-simulation data (pod_info.csv, assign_order.csv).

Used by run_pipeline.py to re-cluster SKUs at the simulation midpoint.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations


def compute_demand_features_from_picks(pod_info_path, time_window_seconds=3600):
    """
    Compute per-SKU demand features from pick records.

    Args:
        pod_info_path: Path to pod_info.csv
        time_window_seconds: Size of time window for demand bucketing (default 1 hour)

    Returns:
        DataFrame with columns: item_code, sim_mean_demand, sim_std_demand,
                                sim_cv_demand, sim_frequency
    """
    df = pd.read_csv(pod_info_path)
    picks = df[df["task_type"] == 1].copy()

    if picks.empty:
        return pd.DataFrame(columns=[
            "item_code", "sim_mean_demand", "sim_std_demand",
            "sim_cv_demand", "sim_frequency"
        ])

    picks["item_id"] = picks["item_id"].astype(str)
    picks["time_window"] = picks["processed_time"] // time_window_seconds

    min_window = int(picks["time_window"].min())
    max_window = int(picks["time_window"].max())
    total_windows = max_window - min_window + 1

    if total_windows < 1:
        total_windows = 1

    demand_per_window = (
        picks.groupby(["item_id", "time_window"])["qty"]
        .sum()
        .reset_index()
    )

    features = []
    for sku, group in demand_per_window.groupby("item_id"):
        all_demands = np.zeros(total_windows)
        for _, row in group.iterrows():
            idx = int(row["time_window"]) - min_window
            if 0 <= idx < total_windows:
                all_demands[idx] = row["qty"]

        mean_d = np.mean(all_demands)
        std_d = np.std(all_demands, ddof=0)
        cv_d = std_d / mean_d if mean_d > 0 else 0.0
        freq = np.count_nonzero(all_demands) / total_windows

        features.append({
            "item_code": sku,
            "sim_mean_demand": mean_d,
            "sim_std_demand": std_d,
            "sim_cv_demand": cv_d,
            "sim_frequency": freq,
        })

    return pd.DataFrame(features)


def compute_affinity_from_orders(assign_order_path):
    """
    Compute per-SKU affinity features from order co-occurrence.

    Args:
        assign_order_path: Path to assign_order.csv

    Returns:
        DataFrame with columns: item_code, sim_avg_affinity, sim_max_affinity
    """
    df = pd.read_csv(assign_order_path)
    processed = df[df["status"] >= 0].copy()

    if processed.empty:
        return pd.DataFrame(columns=["item_code", "sim_avg_affinity", "sim_max_affinity"])

    processed["item_id"] = processed["item_id"].astype(str)

    order_skus = processed.groupby("order_id")["item_id"].apply(set).reset_index()
    total_orders = len(order_skus)

    if total_orders == 0:
        return pd.DataFrame(columns=["item_code", "sim_avg_affinity", "sim_max_affinity"])

    cooccurrence = defaultdict(lambda: defaultdict(int))
    all_skus = set()

    for _, row in order_skus.iterrows():
        skus = row["item_id"]
        all_skus.update(skus)
        for s1, s2 in combinations(skus, 2):
            cooccurrence[s1][s2] += 1
            cooccurrence[s2][s1] += 1

    features = []
    for sku in all_skus:
        if sku in cooccurrence and len(cooccurrence[sku]) > 0:
            affinities = [v / total_orders for v in cooccurrence[sku].values()]
            avg_aff = float(np.mean(affinities))
            max_aff = float(np.max(affinities))
        else:
            avg_aff = 0.0
            max_aff = 0.0

        features.append({
            "item_code": sku,
            "sim_avg_affinity": avg_aff,
            "sim_max_affinity": max_aff,
        })

    return pd.DataFrame(features)


def extract_all_sim_features(pod_info_path, assign_order_path, time_window_seconds=3600):
    """
    Extract and merge all simulation features.

    Returns:
        DataFrame with columns: item_code, sim_mean_demand, sim_std_demand,
                                sim_cv_demand, sim_frequency,
                                sim_avg_affinity, sim_max_affinity
    """
    import os

    # Guard: if pod_info.csv is missing or empty, return empty
    if not os.path.exists(pod_info_path):
        print(f"WARNING: {pod_info_path} not found — skipping demand features")
        demand_feats = pd.DataFrame()
    else:
        demand_feats = compute_demand_features_from_picks(pod_info_path, time_window_seconds)

    # Guard: if assign_order.csv is missing, return empty affinity
    if not os.path.exists(assign_order_path):
        print(f"WARNING: {assign_order_path} not found — skipping affinity features")
        affinity_feats = pd.DataFrame()
    else:
        affinity_feats = compute_affinity_from_orders(assign_order_path)

    if demand_feats.empty and affinity_feats.empty:
        return pd.DataFrame()

    if demand_feats.empty:
        return affinity_feats

    if affinity_feats.empty:
        return demand_feats

    merged = demand_feats.merge(affinity_feats, on="item_code", how="outer")
    merged = merged.fillna(0.0)
    return merged
