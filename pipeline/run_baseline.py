"""
Baseline (no re-clustering) pipeline for RMFS warehouse simulation.

Usage:
    python pipeline/run_baseline.py --total-hours 24 --k 5

Flow:
    1. Initial k-means clustering on static features
    2. Run simulation for the full duration (no mid-point split)
"""

import argparse
import os
import sys
import io
import contextlib
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
    pct = 100 * current / total if total > 0 else 100
    filled = int(BAR_WIDTH * current / total) if total > 0 else BAR_WIDTH
    bar = "█" * filled + "░" * (BAR_WIDTH - filled)
    sys.stderr.write(f"\r  {prefix} |{bar}| {pct:.{decimals}f}% {suffix}")
    sys.stderr.flush()


def status(msg):
    sys.stderr.write(f"\n  → {msg}\n")
    sys.stderr.flush()


def header(title):
    sys.stderr.write(f"\n{'─' * 60}\n  {title}\n{'─' * 60}\n")
    sys.stderr.flush()


# ── clustering ───────────────────────────────────────────────

def run_initial_clustering(sku_sample_path, k):
    header("PHASE 0  Initial K-Means Clustering")
    sku_df = pd.read_csv(sku_sample_path)
    sku_df["item_code"] = sku_df["item_code"].astype(str)
    for col in CLUSTER_FEATURES:
        sku_df[col] = pd.to_numeric(sku_df[col], errors="coerce").fillna(0)

    X = sku_df[CLUSTER_FEATURES].copy()
    X["mean_demand"] = np.log1p(X["mean_demand"])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
    sku_df["cluster"] = km.fit_predict(X_scaled)

    sku_df.to_csv(sku_sample_path, index=False)

    dist = sku_df["cluster"].value_counts().sort_index()
    status(f"Clustered {len(sku_df)} SKUs into {k} groups: {dict(dist)}")
    return sku_df


# ── simulation runner ────────────────────────────────────────

def run_phase(target_tick, phase_name):
    """
    Run tick() loop until universe._tick >= target_tick.

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

    elapsed = time.time() - t0
    last_finished = metrics_log[-1]["orders_finished"] if metrics_log else 0
    progress_bar(target_tick, target_tick, prefix=phase_name,
                 suffix=f" {elapsed:.0f}s elapsed | {last_finished} orders done")
    sys.stderr.write("\n")

    metrics_df = pd.DataFrame(metrics_log)
    status(f"{phase_name} complete: {tick_count} ticks in {elapsed:.1f}s")

    return {"final_tick": current_tick, "tick_count": tick_count, "metrics": metrics_df}


# ── results ──────────────────────────────────────────────────

def save_results(run_data, orders_df, args, results_dir):
    header("RESULTS")

    m = run_data["metrics"]
    if not m.empty:
        m["phase"] = "baseline"
        m.to_csv(os.path.join(results_dir, "tick_metrics.csv"), index=False)

        sys.stderr.write(f"\n  Baseline ({args.total_hours}h, k={args.k})\n")
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

    summary = {
        "total_hours": args.total_hours,
        "k_clusters": args.k,
        "pipeline": "baseline",
        "ticks": run_data["tick_count"],
        "orders_finished": m["orders_finished"].iloc[-1] if not m.empty else 0,
        "total_energy": m["total_energy"].iloc[-1] if not m.empty else 0,
        "stop_and_go": m["stop_and_go"].iloc[-1] if not m.empty else 0,
        "total_turning": m["total_turning"].iloc[-1] if not m.empty else 0,
        "peak_job_queue": m["job_queue_len"].max() if not m.empty else 0,
        "avg_job_queue": round(m["job_queue_len"].mean(), 1) if not m.empty else 0,
    }
    summary_path = os.path.join(results_dir, "summary.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    sys.stderr.write(f"\n  Saved: {results_dir}/\n")
    sys.stderr.write(f"    tick_metrics.csv   ({len(m)} rows)\n")
    sys.stderr.write(f"    summary.csv\n")
    sys.stderr.write(f"    orders.csv\n")
    sys.stderr.write(f"{'─' * 60}\n")


# ── main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RMFS baseline pipeline (no re-clustering)")
    parser.add_argument("--total-hours", type=float, default=24.0,
                        help="Total simulation duration in hours (default: 24)")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of k-means clusters (default: 5)")
    args = parser.parse_args()

    total_seconds = args.total_hours * 3600

    sys.stderr.write(f"\n  RMFS Baseline Pipeline: {args.total_hours}h, K={args.k}\n")
    sys.stderr.write(f"  Single run — no mid-point re-clustering\n")

    netlogo_dir = os.path.join(PROJECT_ROOT, "netlogo")
    sku_sample_path = os.path.join(PROJECT_ROOT, "sku_sample.csv")
    sku_sample_rel = os.path.relpath(sku_sample_path, netlogo_dir)

    # ── Phase 0: Initial Clustering ──
    run_initial_clustering(sku_sample_path, args.k)

    # ── Setup: Generate orders and initialize ──
    os.chdir(netlogo_dir)
    sys.path.insert(0, netlogo_dir)

    from netlogo import reload_data_for_phase

    total_order_hours = max(1, int(np.ceil(args.total_hours)))
    backlog_order_hours = total_order_hours + 1

    status(f"Generating orders for {total_order_hours}h and initializing simulation...")
    with suppress_stdout():
        reload_data_for_phase(
            sku_sample_path=sku_sample_rel,
            order_period_hours=total_order_hours,
            backlog_period_hours=backlog_order_hours
        )
    status("Simulation initialized.")

    # ── Run: Full duration ──
    run_data = run_phase(total_seconds, "Baseline")

    # ── Save results ──
    results_dir = os.path.join(PROJECT_ROOT, "results_baseline")
    os.makedirs(results_dir, exist_ok=True)

    order_finished = None
    if os.path.exists("order-finished.csv"):
        order_finished = pd.read_csv("order-finished.csv")
        order_finished.to_csv(os.path.join(results_dir, "orders.csv"), index=False)

    os.chdir(PROJECT_ROOT)
    save_results(run_data, order_finished, args, results_dir)


if __name__ == "__main__":
    main()
