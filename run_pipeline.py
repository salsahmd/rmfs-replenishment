"""
Mid-simulation re-clustering pipeline for RMFS warehouse simulation.

Usage:
    python run_pipeline.py --total-hours 24 --k 5

Flow:
    1. Initial k-means clustering on static features
    2. Run simulation Phase 1 (first half of duration)
    3. Extract features from simulation data
    4. Re-cluster with observed features
    5. Run simulation Phase 2 (second half of duration)
"""

import argparse
import os
import sys
import pickle
import pandas as pd
import numpy as np
from math import ceil, sqrt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from mid_sim_features import extract_all_sim_features

# Replenishment policy constants (from build_sku_dictionary.py)
LEAD_TIME = 1.0
SERVICE_LEVEL_Z = 1.2816  # 90% service level
RANDOM_STATE = 42
N_INIT = 20

CLUSTER_FEATURES = ["mean_demand", "cv_demand", "demand_frequency", "avg_affinity", "max_affinity"]


def compute_initial_inventory(mean_demand, std_demand):
    mu = 0 if pd.isna(mean_demand) else float(mean_demand)
    sigma = 0 if pd.isna(std_demand) else float(std_demand)
    return int(ceil(mu + SERVICE_LEVEL_Z * sigma * sqrt(LEAD_TIME)))


def run_clustering(sku_df, k):
    """
    Run k-means on the 5 clustering features.
    Modifies sku_df in-place by updating the 'cluster' column.

    Args:
        sku_df: DataFrame with CLUSTER_FEATURES columns
        k: Number of clusters

    Returns:
        sku_df with updated 'cluster' column
    """
    X = sku_df[CLUSTER_FEATURES].copy()
    X["mean_demand"] = np.log1p(X["mean_demand"])

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
    sku_df["cluster"] = km.fit_predict(X_scaled)

    print("\nCluster distribution:")
    print(sku_df["cluster"].value_counts().sort_index())
    return sku_df


def run_initial_clustering(sku_sample_path, k):
    """Phase 0: Cluster using static features from sku_sample.csv."""
    print("=" * 60)
    print("PHASE 0: Initial K-Means Clustering")
    print("=" * 60)

    sku_df = pd.read_csv(sku_sample_path)
    sku_df["item_code"] = sku_df["item_code"].astype(str)

    for col in CLUSTER_FEATURES:
        sku_df[col] = pd.to_numeric(sku_df[col], errors="coerce").fillna(0)

    sku_df = run_clustering(sku_df, k)
    sku_df.to_csv(sku_sample_path, index=False)
    print(f"Saved initial clusters to {sku_sample_path}")
    return sku_df


def run_reclustering(sku_sample_path, sim_features, k):
    """
    Mid-simulation re-clustering.
    Replace static features with sim-observed features for SKUs that were picked.
    Keep original features for SKUs with zero picks.
    """
    print("=" * 60)
    print("MID-POINT: Re-Clustering with Simulation Data")
    print("=" * 60)

    sku_df = pd.read_csv(sku_sample_path)
    sku_df["item_code"] = sku_df["item_code"].astype(str)

    # Save original cluster for comparison
    sku_df["cluster_phase1"] = sku_df["cluster"]

    if sim_features.empty:
        print("WARNING: No simulation features extracted. Keeping original clusters.")
        return sku_df

    sim_features["item_code"] = sim_features["item_code"].astype(str)
    observed_skus = set(sim_features["item_code"].values)
    print(f"SKUs observed in simulation: {len(observed_skus)}")
    print(f"Total SKUs: {len(sku_df)}")

    # Merge sim features
    sku_df = sku_df.merge(sim_features, on="item_code", how="left")

    # Replace features for observed SKUs
    for static_col, sim_col in [
        ("mean_demand", "sim_mean_demand"),
        ("cv_demand", "sim_cv_demand"),
        ("demand_frequency", "sim_frequency"),
        ("avg_affinity", "sim_avg_affinity"),
        ("max_affinity", "sim_max_affinity"),
    ]:
        if sim_col in sku_df.columns:
            mask = sku_df["item_code"].isin(observed_skus) & sku_df[sim_col].notna()
            sku_df.loc[mask, static_col] = sku_df.loc[mask, sim_col]

    # Update std_demand for observed SKUs (for replenishment policy)
    if "sim_std_demand" in sku_df.columns:
        mask = sku_df["item_code"].isin(observed_skus) & sku_df["sim_std_demand"].notna()
        sku_df.loc[mask, "std_demand"] = sku_df.loc[mask, "sim_std_demand"]

    # Drop sim columns
    sim_cols = [c for c in sku_df.columns if c.startswith("sim_")]
    sku_df = sku_df.drop(columns=sim_cols)

    # Re-cluster with updated features
    for col in CLUSTER_FEATURES:
        sku_df[col] = pd.to_numeric(sku_df[col], errors="coerce").fillna(0)

    sku_df = run_clustering(sku_df, k)

    # Recompute replenishment policy
    sku_df["item_initial_inventory"] = sku_df.apply(
        lambda row: compute_initial_inventory(row["mean_demand"], row["std_demand"]),
        axis=1
    )
    if "number_of_item_in_a_box" in sku_df.columns:
        sku_df["box_initial_inventory"] = sku_df.apply(
            lambda row: int(ceil(row["item_initial_inventory"] / max(row["number_of_item_in_a_box"], 1)))
            if pd.notna(row["number_of_item_in_a_box"]) and row["number_of_item_in_a_box"] > 0
            else 0,
            axis=1
        )

    # Report changes
    changed = (sku_df["cluster"] != sku_df["cluster_phase1"]).sum()
    print(f"\nSKUs that changed cluster: {changed} / {len(sku_df)}")
    sku_df = sku_df.drop(columns=["cluster_phase1"])

    sku_df.to_csv(sku_sample_path, index=False)
    print(f"Saved re-clustered data to {sku_sample_path}")
    return sku_df


def run_phase(target_tick, phase_name):
    """
    Run tick() loop until universe._tick >= target_tick.
    Collects per-tick metrics from the simulation.

    tick() returns: [object_positions, total_energy, job_queue_len,
                     stop_and_go, total_turning, station_orders, _tick]

    Returns:
        dict with keys: final_tick, tick_count, metrics (DataFrame)
    """
    from netlogo import tick as sim_tick

    print(f"\n{'=' * 60}")
    print(f"Running {phase_name} until tick >= {target_tick:.1f}s")
    print(f"{'=' * 60}")

    tick_count = 0
    current_tick = 0.0
    metrics_log = []

    while current_tick < target_tick:
        result = sim_tick()

        if isinstance(result, str):
            print(f"ERROR during tick: {result}")
            break

        # result: [positions, energy, job_queue_len, stop_and_go, turning, station_orders, _tick]
        current_tick = result[-1]
        tick_count += 1

        metrics_log.append({
            "tick": current_tick,
            "total_energy": result[1],
            "job_queue_len": result[2],
            "stop_and_go": result[3],
            "total_turning": result[4],
        })

        if tick_count % 100 == 0:
            print(f"  tick #{tick_count}: {current_tick:.1f}s / {target_tick:.1f}s "
                  f"| energy={result[1]:.1f} jobs={result[2]} s&g={result[3]} turns={result[4]}")

    metrics_df = pd.DataFrame(metrics_log)
    print(f"{phase_name} complete: {tick_count} ticks, final time = {current_tick:.1f}s")

    return {"final_tick": current_tick, "tick_count": tick_count, "metrics": metrics_df}


def save_results(phase1, phase2, phase1_orders, phase2_orders, args):
    """Save pipeline results summary and per-tick metrics to CSV."""
    print("\n" + "=" * 60)
    print("PIPELINE RESULTS")
    print("=" * 60)

    # --- Per-tick metrics ---
    if not phase1["metrics"].empty:
        phase1["metrics"]["phase"] = 1
    if not phase2["metrics"].empty:
        phase2["metrics"]["phase"] = 2

    all_metrics = pd.concat([phase1["metrics"], phase2["metrics"]], ignore_index=True)
    all_metrics.to_csv("results_tick_metrics.csv", index=False)
    print(f"Saved per-tick metrics to results_tick_metrics.csv ({len(all_metrics)} rows)")

    # --- Phase summaries ---
    for label, phase_data, orders_df in [
        ("Phase 1", phase1, phase1_orders),
        ("Phase 2", phase2, phase2_orders),
    ]:
        m = phase_data["metrics"]
        print(f"\n--- {label} ---")
        print(f"  Ticks:          {phase_data['tick_count']}")
        print(f"  Sim time:       {phase_data['final_tick']:.1f}s")

        if not m.empty:
            print(f"  Total energy:   {m['total_energy'].iloc[-1]:.2f}")
            print(f"  Stop & go:      {m['stop_and_go'].iloc[-1]}")
            print(f"  Total turning:  {m['total_turning'].iloc[-1]}")
            print(f"  Peak job queue: {m['job_queue_len'].max()}")
            print(f"  Avg job queue:  {m['job_queue_len'].mean():.1f}")

        if orders_df is not None and not orders_df.empty:
            completed = len(orders_df)
            if "order_complete_time" in orders_df.columns and "process_start_time" in orders_df.columns:
                orders_df["cycle_time"] = orders_df["order_complete_time"] - orders_df["process_start_time"]
                avg_cycle = orders_df["cycle_time"].mean()
                max_cycle = orders_df["cycle_time"].max()
                print(f"  Orders completed:    {completed}")
                print(f"  Avg order cycle time: {avg_cycle:.1f}s")
                print(f"  Max order cycle time: {max_cycle:.1f}s")
            else:
                print(f"  Orders completed: {completed}")

    # --- Combined summary CSV ---
    summary = {
        "total_hours": args.total_hours,
        "k_clusters": args.k,
        "phase1_ticks": phase1["tick_count"],
        "phase2_ticks": phase2["tick_count"],
        "phase1_final_energy": phase1["metrics"]["total_energy"].iloc[-1] if not phase1["metrics"].empty else 0,
        "phase2_final_energy": phase2["metrics"]["total_energy"].iloc[-1] if not phase2["metrics"].empty else 0,
        "phase1_stop_and_go": phase1["metrics"]["stop_and_go"].iloc[-1] if not phase1["metrics"].empty else 0,
        "phase2_stop_and_go": phase2["metrics"]["stop_and_go"].iloc[-1] if not phase2["metrics"].empty else 0,
        "phase1_orders_completed": len(phase1_orders) if phase1_orders is not None else 0,
        "phase2_orders_completed": len(phase2_orders) if phase2_orders is not None else 0,
    }
    pd.DataFrame([summary]).to_csv("results_summary.csv", index=False)
    print(f"\nSaved summary to results_summary.csv")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="RMFS mid-simulation re-clustering pipeline")
    parser.add_argument("--total-hours", type=float, default=24.0,
                        help="Total simulation duration in hours (default: 24)")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of k-means clusters (default: 5)")
    args = parser.parse_args()

    total_seconds = args.total_hours * 3600
    half_seconds = total_seconds / 2
    half_hours = args.total_hours / 2

    print(f"Pipeline configuration:")
    print(f"  Total duration: {args.total_hours} hours ({total_seconds:.0f}s)")
    print(f"  Half duration:  {half_hours} hours ({half_seconds:.0f}s)")
    print(f"  K clusters:     {args.k}")

    # Save original cwd and switch to netlogo/ for simulation
    original_cwd = os.getcwd()
    netlogo_dir = os.path.join(PROJECT_ROOT, "netlogo")
    sku_sample_path = os.path.join(PROJECT_ROOT, "sku_sample.csv")
    sku_sample_rel = os.path.relpath(sku_sample_path, netlogo_dir)

    # --- Phase 0: Initial Clustering ---
    run_initial_clustering(sku_sample_path, args.k)

    # --- Switch to netlogo/ for simulation ---
    os.chdir(netlogo_dir)
    sys.path.insert(0, netlogo_dir)

    from netlogo import reload_data_for_phase, reload_pods_only

    # --- Setup: Generate ALL orders for the full duration, then start sim ---
    total_order_hours = max(1, int(np.ceil(args.total_hours)))
    backlog_order_hours = total_order_hours + 1
    print(f"\nSetting up simulation (generating orders for {total_order_hours}h)...")
    reload_data_for_phase(
        sku_sample_path=sku_sample_rel,
        order_period_hours=total_order_hours,
        backlog_period_hours=backlog_order_hours
    )

    # --- Phase 1: Run first half ---
    phase1 = run_phase(half_seconds, "Phase 1")

    # --- Save Phase 1 results before pod reload ---
    phase1_order_finished = None
    if os.path.exists("order-finished.csv"):
        phase1_order_finished = pd.read_csv("order-finished.csv")
        phase1_order_finished.to_csv(
            os.path.join(original_cwd, "results_phase1_orders.csv"), index=False
        )

    # --- Mid-point: Extract features and re-cluster ---
    print("\nExtracting mid-simulation features...")
    sim_features = extract_all_sim_features(
        pod_info_path="pod_info.csv",
        assign_order_path="assign_order.csv",
        time_window_seconds=3600
    )
    print(f"Extracted features for {len(sim_features)} SKUs")

    os.chdir(original_cwd)
    run_reclustering(sku_sample_path, sim_features, args.k)
    os.chdir(netlogo_dir)

    # --- Mid-point reload: new pods + replenishment, SAME order stream ---
    print("\nReloading with new clusters (keeping order stream)...")
    reload_pods_only(
        sku_sample_path=sku_sample_rel,
        midpoint_seconds=half_seconds
    )

    # --- Phase 2: Run second half (remaining orders, shifted to start from 0) ---
    phase2 = run_phase(half_seconds, "Phase 2")

    # --- Save Phase 2 results ---
    phase2_order_finished = None
    if os.path.exists("order-finished.csv"):
        phase2_order_finished = pd.read_csv("order-finished.csv")
        phase2_order_finished.to_csv(
            os.path.join(original_cwd, "results_phase2_orders.csv"), index=False
        )

    # --- Compile and save results ---
    os.chdir(original_cwd)
    save_results(phase1, phase2, phase1_order_finished, phase2_order_finished, args)


if __name__ == "__main__":
    main()
