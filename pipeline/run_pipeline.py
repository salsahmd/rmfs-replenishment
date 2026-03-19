"""
Mid-simulation re-clustering pipeline for RMFS warehouse simulation.

Usage:
    python pipeline/run_pipeline.py --total-hours 24 --k 5

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
import io
import contextlib
import pickle
import time
import pandas as pd
import numpy as np
from math import ceil, sqrt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# pipeline/ lives inside the project root
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PIPELINE_DIR)
sys.path.insert(0, PIPELINE_DIR)
sys.path.insert(0, PROJECT_ROOT)

from mid_sim_features import extract_all_sim_features

# Replenishment policy constants (from build_sku_dictionary.py)
LEAD_TIME = 1.0
SERVICE_LEVEL_Z = 1.2816  # 90% service level
RANDOM_STATE = 42
N_INIT = 20

CLUSTER_FEATURES = ["mean_demand", "cv_demand", "demand_frequency", "avg_affinity", "max_affinity"]


BAR_WIDTH = 40


# ── helpers ──────────────────────────────────────────────────

@contextlib.contextmanager
def suppress_stdout():
    """Redirect stdout to devnull to silence simulation debug prints."""
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


def progress_bar(current, total, prefix="", suffix="", decimals=1):
    """Print an in-place progress bar to stderr."""
    pct = 100 * current / total if total > 0 else 100
    filled = int(BAR_WIDTH * current / total) if total > 0 else BAR_WIDTH
    bar = "█" * filled + "░" * (BAR_WIDTH - filled)
    sys.stderr.write(f"\r  {prefix} |{bar}| {pct:.{decimals}f}% {suffix}")
    sys.stderr.flush()


def status(msg):
    """Print a clean status line."""
    sys.stderr.write(f"\n  → {msg}\n")
    sys.stderr.flush()


def header(title):
    sys.stderr.write(f"\n{'─' * 60}\n  {title}\n{'─' * 60}\n")
    sys.stderr.flush()


# ── clustering ───────────────────────────────────────────────

def compute_initial_inventory(mean_demand, std_demand):
    mu = 0 if pd.isna(mean_demand) else float(mean_demand)
    sigma = 0 if pd.isna(std_demand) else float(std_demand)
    return int(ceil(mu + SERVICE_LEVEL_Z * sigma * sqrt(LEAD_TIME)))


def run_clustering(sku_df, k):
    X = sku_df[CLUSTER_FEATURES].copy()
    X["mean_demand"] = np.log1p(X["mean_demand"])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
    sku_df["cluster"] = km.fit_predict(X_scaled)
    return sku_df


def run_initial_clustering(sku_sample_path, k):
    header("PHASE 0  Initial K-Means Clustering")
    sku_df = pd.read_csv(sku_sample_path)
    sku_df["item_code"] = sku_df["item_code"].astype(str)
    for col in CLUSTER_FEATURES:
        sku_df[col] = pd.to_numeric(sku_df[col], errors="coerce").fillna(0)
    sku_df = run_clustering(sku_df, k)
    sku_df.to_csv(sku_sample_path, index=False)

    dist = sku_df["cluster"].value_counts().sort_index()
    status(f"Clustered {len(sku_df)} SKUs into {k} groups: {dict(dist)}")
    return sku_df


def run_reclustering(sku_sample_path, sim_features, k):
    header("MID-POINT  Re-Clustering with Simulation Data")

    sku_df = pd.read_csv(sku_sample_path)
    sku_df["item_code"] = sku_df["item_code"].astype(str)
    sku_df["cluster_phase1"] = sku_df["cluster"]

    if sim_features.empty:
        status("WARNING: No simulation features extracted. Keeping original clusters.")
        return sku_df

    sim_features["item_code"] = sim_features["item_code"].astype(str)
    observed_skus = set(sim_features["item_code"].values)
    status(f"Observed {len(observed_skus)} / {len(sku_df)} SKUs in simulation")

    sku_df = sku_df.merge(sim_features, on="item_code", how="left")

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

    if "sim_std_demand" in sku_df.columns:
        mask = sku_df["item_code"].isin(observed_skus) & sku_df["sim_std_demand"].notna()
        sku_df.loc[mask, "std_demand"] = sku_df.loc[mask, "sim_std_demand"]

    sim_cols = [c for c in sku_df.columns if c.startswith("sim_")]
    sku_df = sku_df.drop(columns=sim_cols)

    for col in CLUSTER_FEATURES:
        sku_df[col] = pd.to_numeric(sku_df[col], errors="coerce").fillna(0)

    sku_df = run_clustering(sku_df, k)

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

    changed = (sku_df["cluster"] != sku_df["cluster_phase1"]).sum()
    sku_df = sku_df.drop(columns=["cluster_phase1"])
    sku_df.to_csv(sku_sample_path, index=False)

    dist = sku_df["cluster"].value_counts().sort_index()
    status(f"Re-clustered: {changed} SKUs changed cluster. Distribution: {dict(dist)}")
    return sku_df


# ── simulation runner ────────────────────────────────────────

def run_phase(target_tick, phase_name):
    """
    Run tick() loop until universe._tick >= target_tick.
    Silences simulation debug output; shows a progress bar instead.

    tick() returns: [positions, energy, job_queue_len, stop_and_go,
                     turning, station_orders, orders_finished, _tick]
    """
    from netlogo import tick as sim_tick

    header(f"{phase_name}  Simulating {target_tick:.0f}s")

    tick_count = 0
    current_tick = 0.0
    metrics_log = []
    t0 = time.time()

    while current_tick < target_tick:
        with suppress_stdout():
            result = sim_tick()

        if isinstance(result, str):
            sys.stderr.write(f"\n  ERROR: {result}\n")
            break

        current_tick = result[-1]
        tick_count += 1

        metrics_log.append({
            "tick": current_tick,
            "total_energy": result[1],
            "job_queue_len": result[2],
            "stop_and_go": result[3],
            "total_turning": result[4],
            "orders_finished": result[6],
        })

        if tick_count % 50 == 0 or current_tick >= target_tick:
            elapsed = time.time() - t0
            finished = result[6]
            progress_bar(
                current_tick, target_tick,
                prefix=phase_name,
                suffix=f" {elapsed:.0f}s elapsed | {finished} orders done"
            )

    # Final bar at 100%
    elapsed = time.time() - t0
    last_finished = metrics_log[-1]["orders_finished"] if metrics_log else 0
    progress_bar(target_tick, target_tick, prefix=phase_name,
                 suffix=f" {elapsed:.0f}s elapsed | {last_finished} orders done")
    sys.stderr.write("\n")

    metrics_df = pd.DataFrame(metrics_log)
    status(f"{phase_name} complete: {tick_count} ticks in {elapsed:.1f}s")

    return {"final_tick": current_tick, "tick_count": tick_count, "metrics": metrics_df}


# ── results ──────────────────────────────────────────────────

def compute_extra_metrics(orders_finished, generated_order_path, pod_info_path):
    """Compute order throughput, replenishment/pick ratio, and pod utilization."""
    metrics = {}

    # Order throughput = orders finished / orders generated
    orders_generated = 0
    if os.path.exists(generated_order_path):
        gen_df = pd.read_csv(generated_order_path, dtype=str)
        orders_generated = gen_df["order_id"].nunique()
    metrics["orders_generated"] = orders_generated
    metrics["order_throughput"] = (
        orders_finished / orders_generated if orders_generated > 0 else 0.0
    )

    # Replenishment/pick ratio and pod utilization from pod_info.csv
    total_picks = 0
    total_replenishments = 0
    total_units_picked = 0
    pod_visits = 0
    if os.path.exists(pod_info_path):
        pod_df = pd.read_csv(pod_info_path)
        if not pod_df.empty and "task_type" in pod_df.columns:
            pick_df = pod_df[pod_df["task_type"] == 1]
            replen_df = pod_df[pod_df["task_type"] == 2]
            total_picks = len(pick_df)
            total_replenishments = len(replen_df)
            total_units_picked = pick_df["qty"].sum() if "qty" in pick_df.columns else 0
            # Pod visit = unique (pod_id, processed_time) for pick tasks
            if not pick_df.empty and "pod_id" in pick_df.columns and "processed_time" in pick_df.columns:
                pod_visits = pick_df.groupby(["pod_id", "processed_time"]).ngroups

    metrics["total_picks"] = total_picks
    metrics["total_replenishments"] = total_replenishments
    metrics["replenishment_pick_ratio"] = (
        total_replenishments / total_picks if total_picks > 0 else 0.0
    )
    metrics["total_units_picked"] = total_units_picked
    metrics["pod_visits"] = pod_visits
    metrics["pod_utilization"] = (
        total_units_picked / pod_visits if pod_visits > 0 else 0.0
    )

    return metrics


def save_results(phase1, phase2, phase1_orders, phase2_orders, args, results_dir,
                 phase1_extra=None, phase2_extra=None):
    header("RESULTS")

    # Per-tick metrics
    if not phase1["metrics"].empty:
        phase1["metrics"]["phase"] = 1
    if not phase2["metrics"].empty:
        phase2["metrics"]["phase"] = 2

    all_metrics = pd.concat([phase1["metrics"], phase2["metrics"]], ignore_index=True)
    metrics_path = os.path.join(results_dir, "tick_metrics.csv")
    all_metrics.to_csv(metrics_path, index=False)

    # Phase summaries
    for label, phase_data, orders_df, extra in [
        ("Phase 1 (before re-cluster)", phase1, phase1_orders, phase1_extra),
        ("Phase 2 (after re-cluster)", phase2, phase2_orders, phase2_extra),
    ]:
        m = phase_data["metrics"]
        sys.stderr.write(f"\n  {label}\n")

        if not m.empty:
            sys.stderr.write(f"    Orders finished:   {m['orders_finished'].iloc[-1]}\n")
            sys.stderr.write(f"    Total energy:      {m['total_energy'].iloc[-1]:.2f}\n")
            sys.stderr.write(f"    Stop & go:         {m['stop_and_go'].iloc[-1]}\n")
            sys.stderr.write(f"    Total turning:     {m['total_turning'].iloc[-1]}\n")
            sys.stderr.write(f"    Peak job queue:    {m['job_queue_len'].max()}\n")
            sys.stderr.write(f"    Avg job queue:     {m['job_queue_len'].mean():.1f}\n")

        if orders_df is not None and not orders_df.empty:
            if "order_complete_time" in orders_df.columns and "process_start_time" in orders_df.columns:
                orders_df = orders_df.copy()
                orders_df["cycle_time"] = orders_df["order_complete_time"] - orders_df["process_start_time"]
                sys.stderr.write(f"    Avg cycle time:    {orders_df['cycle_time'].mean():.1f}s\n")
                sys.stderr.write(f"    Max cycle time:    {orders_df['cycle_time'].max():.1f}s\n")

        if extra:
            sys.stderr.write(f"    Order throughput:  {extra['order_throughput']:.4f} ({extra['orders_generated']} generated)\n")
            sys.stderr.write(f"    Replen/pick ratio: {extra['replenishment_pick_ratio']:.4f} ({extra['total_replenishments']}R / {extra['total_picks']}P)\n")
            sys.stderr.write(f"    Pod utilization:   {extra['pod_utilization']:.4f} ({extra['total_units_picked']} units / {extra['pod_visits']} visits)\n")

    # Combined summary CSV
    def _last(metrics_df, col):
        return metrics_df[col].iloc[-1] if not metrics_df.empty else 0

    summary = {
        "total_hours": args.total_hours,
        "k_clusters": args.k,
        "phase1_ticks": phase1["tick_count"],
        "phase2_ticks": phase2["tick_count"],
        "phase1_orders_finished": _last(phase1["metrics"], "orders_finished"),
        "phase2_orders_finished": _last(phase2["metrics"], "orders_finished"),
        "phase1_final_energy": _last(phase1["metrics"], "total_energy"),
        "phase2_final_energy": _last(phase2["metrics"], "total_energy"),
        "phase1_stop_and_go": _last(phase1["metrics"], "stop_and_go"),
        "phase2_stop_and_go": _last(phase2["metrics"], "stop_and_go"),
        "phase1_total_turning": _last(phase1["metrics"], "total_turning"),
        "phase2_total_turning": _last(phase2["metrics"], "total_turning"),
        "phase1_peak_job_queue": phase1["metrics"]["job_queue_len"].max() if not phase1["metrics"].empty else 0,
        "phase2_peak_job_queue": phase2["metrics"]["job_queue_len"].max() if not phase2["metrics"].empty else 0,
    }

    # Add extra metrics to summary
    for prefix, extra in [("phase1", phase1_extra), ("phase2", phase2_extra)]:
        if extra:
            summary[f"{prefix}_orders_generated"] = extra["orders_generated"]
            summary[f"{prefix}_order_throughput"] = round(extra["order_throughput"], 4)
            summary[f"{prefix}_total_picks"] = extra["total_picks"]
            summary[f"{prefix}_total_replenishments"] = extra["total_replenishments"]
            summary[f"{prefix}_replenishment_pick_ratio"] = round(extra["replenishment_pick_ratio"], 4)
            summary[f"{prefix}_total_units_picked"] = extra["total_units_picked"]
            summary[f"{prefix}_pod_visits"] = extra["pod_visits"]
            summary[f"{prefix}_pod_utilization"] = round(extra["pod_utilization"], 4)

    summary_path = os.path.join(results_dir, "summary.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    sys.stderr.write(f"\n  Saved: {results_dir}/\n")
    sys.stderr.write(f"    tick_metrics.csv   ({len(all_metrics)} rows)\n")
    sys.stderr.write(f"    summary.csv\n")
    sys.stderr.write(f"    phase1_orders.csv\n")
    sys.stderr.write(f"    phase2_orders.csv\n")
    sys.stderr.write(f"{'─' * 60}\n")


# ── main ─────────────────────────────────────────────────────

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

    sys.stderr.write(f"\n  RMFS Pipeline: {args.total_hours}h total, K={args.k}\n")
    sys.stderr.write(f"  Phase 1: {half_hours}h → re-cluster → Phase 2: {half_hours}h\n")

    original_cwd = os.getcwd()
    netlogo_dir = os.path.join(PROJECT_ROOT, "netlogo")
    sku_sample_path = os.path.join(PROJECT_ROOT, "sku_sample.csv")
    sku_sample_rel = os.path.relpath(sku_sample_path, netlogo_dir)

    # ── Phase 0: Initial Clustering ──
    run_initial_clustering(sku_sample_path, args.k)

    # ── Setup: Generate orders for full duration ──
    os.chdir(netlogo_dir)
    sys.path.insert(0, netlogo_dir)

    from netlogo import reload_data_for_phase, reload_pods_only

    total_order_hours = max(1, int(np.ceil(args.total_hours)))
    backlog_order_hours = total_order_hours + 1

    status(f"Generating orders for {total_order_hours}h and initializing simulation...")
    with suppress_stdout():
        reload_data_for_phase(
            sku_sample_path=sku_sample_rel,
            order_period_hours=total_order_hours,
            backlog_period_hours=backlog_order_hours,
        )
    status("Simulation initialized.")

    # ── Phase 1: First half ──
    phase1 = run_phase(half_seconds, "Phase 1")

    # ── Save Phase 1 results ──
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    phase1_order_finished = None
    if os.path.exists("order-finished.csv"):
        phase1_order_finished = pd.read_csv("order-finished.csv")
        phase1_order_finished.to_csv(
            os.path.join(results_dir, "phase1_orders.csv"), index=False
        )

    # ── Phase 1 extra metrics ──
    phase1_finished_count = int(phase1["metrics"]["orders_finished"].iloc[-1]) if not phase1["metrics"].empty else 0
    phase1_extra = compute_extra_metrics(
        phase1_finished_count, "generated_order.csv", "pod_info.csv"
    )

    # ── Mid-point: Extract + Re-cluster ──
    status("Extracting simulation features...")
    sim_features = extract_all_sim_features(
        pod_info_path="pod_info.csv",
        assign_order_path="assign_order.csv",
        time_window_seconds=3600
    )
    status(f"Extracted features for {len(sim_features)} SKUs")

    os.chdir(original_cwd)
    run_reclustering(sku_sample_path, sim_features, args.k)
    os.chdir(netlogo_dir)

    # ── Reload with new clusters (keep order stream) ──
    status("Reloading simulation with new clusters...")
    with suppress_stdout():
        reload_pods_only(
            sku_sample_path=sku_sample_rel,
            midpoint_seconds=half_seconds
        )
    status("Simulation reloaded with new pod assignments.")

    # ── Phase 2: Second half ──
    phase2 = run_phase(half_seconds, "Phase 2")

    # ── Save Phase 2 results ──
    phase2_order_finished = None
    if os.path.exists("order-finished.csv"):
        phase2_order_finished = pd.read_csv("order-finished.csv")
        phase2_order_finished.to_csv(
            os.path.join(results_dir, "phase2_orders.csv"), index=False
        )

    # ── Phase 2 extra metrics ──
    phase2_finished_count = int(phase2["metrics"]["orders_finished"].iloc[-1]) if not phase2["metrics"].empty else 0
    phase2_extra = compute_extra_metrics(
        phase2_finished_count, "generated_order.csv", "pod_info.csv"
    )

    # ── Final results ──
    os.chdir(PROJECT_ROOT)
    save_results(phase1, phase2, phase1_order_finished, phase2_order_finished, args, results_dir,
                 phase1_extra=phase1_extra, phase2_extra=phase2_extra)


if __name__ == "__main__":
    main()
