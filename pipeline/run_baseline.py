"""
Baseline pipeline for RMFS warehouse simulation (Rika's baseline).

Usage:
    python pipeline/run_baseline.py [--max-ticks 1000]

Flow:
    1. Run setup() to initialize warehouse, orders, pods, robots
    2. Run tick() loop until simulation ends
    3. Collect and save metrics
"""

import argparse
import os
import sys
import io
import contextlib
import time
import pandas as pd

# pipeline/ lives inside the project root
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PIPELINE_DIR)
sys.path.insert(0, PIPELINE_DIR)
sys.path.insert(0, PROJECT_ROOT)


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


# ── simulation runner ────────────────────────────────────────

def run_simulation(max_ticks):
    """
    Run setup() then tick() loop.

    tick() returns: [positions, energy, job_queue_len, stop_and_go,
                     turning, station_orders]

    Returns IndexError type when universe._tick > 1000 (simulation end).
    """
    from netlogo import setup as sim_setup, tick as sim_tick

    header("SETUP  Initializing simulation")
    status("Running setup()...")
    with suppress_stdout():
        sim_setup()
    status("Setup complete.")

    header(f"SIMULATION  Running (max {max_ticks} ticks)")

    tick_count = 0
    metrics_log = []
    t0 = time.time()

    while tick_count < max_ticks:
        with suppress_stdout():
            result = sim_tick()

        # tick() returns IndexError type when _tick > 1000
        if result is IndexError:
            status(f"Simulation ended at tick {tick_count} (tick limit reached)")
            break

        if isinstance(result, str):
            sys.stderr.write(f"\n  ERROR: {result}\n")
            break

        tick_count += 1

        # result: [positions, energy, job_queue_len, stop_and_go, turning, station_orders]
        metrics_log.append({
            "tick": tick_count,
            "total_energy": result[1],
            "job_queue_len": result[2],
            "stop_and_go": result[3],
            "total_turning": result[4],
        })

        if tick_count % 50 == 0:
            elapsed = time.time() - t0
            progress_bar(
                tick_count, max_ticks,
                prefix="Baseline",
                suffix=f" {elapsed:.0f}s elapsed | tick {tick_count}"
            )

    elapsed = time.time() - t0
    progress_bar(tick_count, max_ticks, prefix="Baseline",
                 suffix=f" {elapsed:.0f}s elapsed | {tick_count} ticks done")
    sys.stderr.write("\n")

    metrics_df = pd.DataFrame(metrics_log)
    status(f"Simulation complete: {tick_count} ticks in {elapsed:.1f}s")

    return {"tick_count": tick_count, "metrics": metrics_df}


# ── results ──────────────────────────────────────────────────

def compute_extra_metrics(generated_order_path, pod_info_path, order_finished_path):
    """Compute order throughput, replenishment/pick ratio, and pod utilization."""
    metrics = {}

    # Orders finished — database orders only (backlog orders have negative order_id)
    orders_finished = 0
    if os.path.exists(order_finished_path):
        finished_df = pd.read_csv(order_finished_path)
        finished_df["order_id"] = pd.to_numeric(finished_df["order_id"], errors="coerce")
        orders_finished = int((finished_df["order_id"] >= 0).sum())
    metrics["orders_finished"] = orders_finished

    # Orders generated — database orders only (generated_order.csv may contain merged backlog)
    orders_generated = 0
    if os.path.exists(generated_order_path):
        gen_df = pd.read_csv(generated_order_path)
        gen_df["order_id"] = pd.to_numeric(gen_df["order_id"], errors="coerce")
        orders_generated = gen_df.loc[gen_df["order_id"] >= 0, "order_id"].nunique()
    metrics["orders_generated"] = orders_generated

    # order_throughput = database orders completed / database orders generated
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


def save_results(run_data, args, results_dir, extra):
    header("RESULTS")

    m = run_data["metrics"]
    if not m.empty:
        m["phase"] = "baseline"
        m.to_csv(os.path.join(results_dir, "tick_metrics.csv"), index=False)

        sys.stderr.write(f"\n  Baseline (max_ticks={args.max_ticks})\n")
        sys.stderr.write(f"    Orders finished:   {extra['orders_finished']}\n")
        sys.stderr.write(f"    Total energy:      {m['total_energy'].iloc[-1]:.2f}\n")
        sys.stderr.write(f"    Stop & go:         {m['stop_and_go'].iloc[-1]}\n")
        sys.stderr.write(f"    Total turning:     {m['total_turning'].iloc[-1]}\n")
        sys.stderr.write(f"    Peak job queue:    {m['job_queue_len'].max()}\n")
        sys.stderr.write(f"    Avg job queue:     {m['job_queue_len'].mean():.1f}\n")

    # Order cycle time from order-finished.csv
    order_finished_path = os.path.join("output", "order-finished.csv")
    if os.path.exists(order_finished_path):
        orders_df = pd.read_csv(order_finished_path)
        orders_df.to_csv(os.path.join(results_dir, "orders.csv"), index=False)
        if "order_complete_time" in orders_df.columns and "process_start_time" in orders_df.columns:
            orders_df["cycle_time"] = orders_df["order_complete_time"] - orders_df["process_start_time"]
            sys.stderr.write(f"    Avg cycle time:    {orders_df['cycle_time'].mean():.1f}s\n")
            sys.stderr.write(f"    Max cycle time:    {orders_df['cycle_time'].max():.1f}s\n")

    sys.stderr.write(f"    Order throughput:  {extra['order_throughput']:.4f} ({extra['orders_generated']} generated)\n")
    sys.stderr.write(f"    Replen/pick ratio: {extra['replenishment_pick_ratio']:.4f} ({extra['total_replenishments']}R / {extra['total_picks']}P)\n")
    sys.stderr.write(f"    Pod utilization:   {extra['pod_utilization']:.4f} ({extra['total_units_picked']} units / {extra['pod_visits']} visits)\n")

    summary = {
        "max_ticks": args.max_ticks,
        "pipeline": "baseline",
        "ticks": run_data["tick_count"],
        "orders_finished": extra["orders_finished"],
        "total_energy": m["total_energy"].iloc[-1] if not m.empty else 0,
        "stop_and_go": m["stop_and_go"].iloc[-1] if not m.empty else 0,
        "total_turning": m["total_turning"].iloc[-1] if not m.empty else 0,
        "peak_job_queue": m["job_queue_len"].max() if not m.empty else 0,
        "avg_job_queue": round(m["job_queue_len"].mean(), 1) if not m.empty else 0,
        "orders_generated": extra["orders_generated"],
        "order_throughput": round(extra["order_throughput"], 4),
        "total_picks": extra["total_picks"],
        "total_replenishments": extra["total_replenishments"],
        "replenishment_pick_ratio": round(extra["replenishment_pick_ratio"], 4),
        "total_units_picked": extra["total_units_picked"],
        "pod_visits": extra["pod_visits"],
        "pod_utilization": round(extra["pod_utilization"], 4),
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
    parser = argparse.ArgumentParser(description="RMFS baseline pipeline (Rika's baseline)")
    parser.add_argument("--max-ticks", type=int, default=10000,
                        help="Maximum number of simulation ticks (default: 10000)")
    args = parser.parse_args()

    sys.stderr.write(f"\n  RMFS Baseline Pipeline (Rika's baseline)\n")
    sys.stderr.write(f"  Max ticks: {args.max_ticks}\n")

    netlogo_dir = os.path.join(PROJECT_ROOT, "netlogo")

    # Switch to netlogo/ directory (simulation expects files in cwd)
    os.chdir(netlogo_dir)
    sys.path.insert(0, netlogo_dir)

    # Run simulation
    run_data = run_simulation(args.max_ticks)

    # Compute extra metrics
    order_finished_path = os.path.join("output", "order-finished.csv")
    extra = compute_extra_metrics("generated_order.csv", "pod_info.csv", order_finished_path)

    # Save results
    results_dir = os.path.join(PROJECT_ROOT, "results_baseline")
    os.makedirs(results_dir, exist_ok=True)

    save_results(run_data, args, results_dir, extra)

    os.chdir(PROJECT_ROOT)


if __name__ == "__main__":
    main()
