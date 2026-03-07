# SALSA-RMFS — Mid-Simulation Re-Clustering for Robotic Mobile Fulfilment Systems

A warehouse simulation pipeline that uses **k-means clustering** to assign SKUs to pods, runs the simulation in two phases, and **re-clusters at the midpoint** using features observed during Phase 1.

## Overview

```
Phase 0   Initial k-means clustering on static demand features
    ↓
Phase 1   Simulate first half of the duration
    ↓
Mid-point Extract simulation features → re-cluster SKUs
    ↓
Phase 2   Simulate second half with updated pod assignments
    ↓
Results   Compare Phase 1 vs Phase 2 metrics
```

## Prerequisites

- **Python 3.10+**
- pip

## Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd salsa-rmfs

# 2. Create a virtual environment (recommended)
python3 -m venv netlogo/.venv
source netlogo/.venv/bin/activate

# 3. Install dependencies
pip install -r netlogo/requirements.txt
```

## Quick Start

Run from the **project root** directory:

```bash
# Short smoke test (≈ 2 min)
python3 pipeline/run_pipeline.py --total-hours 0.25 --k 5

# 1-hour simulation (≈ 10 min)
python3 pipeline/run_pipeline.py --total-hours 1 --k 5

# Full 24-hour simulation
python3 pipeline/run_pipeline.py --total-hours 24 --k 5
```

### CLI Options

| Flag | Default | Description |
|---|---|---|
| `--total-hours` | `24` | Total simulation duration in hours. Split equally between Phase 1 and Phase 2. |
| `--k` | `5` | Number of k-means clusters for SKU grouping. |

## Output

Results are saved to the `results/` directory:

| File | Description |
|---|---|
| `summary.csv` | One-row summary: orders finished, energy, stop-and-go, turning, peak/avg job queue for each phase. |
| `tick_metrics.csv` | Per-tick time-series of energy, job queue length, stop-and-go count, and turning. |
| `phase1_orders.csv` | Completed orders from Phase 1. |
| `phase2_orders.csv` | Completed orders from Phase 2. |

Console output shows a live progress bar and a side-by-side comparison of Phase 1 vs Phase 2 at the end:

```
  Phase 1 (before re-cluster)
    Orders finished:   73
    Total energy:      3987205.61
    Peak job queue:    49
    Avg job queue:     31.4

  Phase 2 (after re-cluster)
    Orders finished:   43
    Total energy:      3285225.30
    Peak job queue:    25
    Avg job queue:     4.2
```

## Project Structure

```
salsa-rmfs/
├── pipeline/
│   ├── run_pipeline.py        # Main entry point — orchestrates both phases
│   └── mid_sim_features.py    # Extracts features from simulation data
├── netlogo/
│   ├── netlogo.py             # Simulation setup, tick loop, pod/order config
│   ├── model/
│   │   ├── inventory.py       # Core simulation universe (order processing, metrics)
│   │   ├── robot.py           # Robot movement, energy, job execution
│   │   ├── order.py           # Order data model
│   │   ├── order_generator.py # Generates synthetic orders from demand data
│   │   └── ...
│   ├── engine/
│   │   └── universe.py        # Base simulation universe class
│   ├── requirements.txt       # Python dependencies
│   └── config.dictionary      # Simulation parameters (robots, pods, stations)
├── clustering/                # Standalone clustering experiments
├── results/                   # Output from pipeline runs
└── README.md
```

## Data Requirements

The pipeline expects these CSV files in the project root:

- `sku_sample.csv` — SKU catalogue with item IDs
- `daily_qty_by_item.csv` — Daily demand quantities per SKU
- `raw_order.csv` — Historical order data for demand features
- `pod_size.csv` / `pod_type_quota.csv` / `pods_dictionary.csv` — Pod layout configuration
- `items_dictionary.csv` — Item metadata

These are included in the repository.

## How It Works

1. **Clustering (Phase 0):** SKUs are clustered by demand features (mean demand, CV, frequency, affinity scores) using k-means. Each cluster maps to a pod zone.

2. **Phase 1:** The simulation runs for the first half of the duration. Robots fetch pods to picker stations to fulfil incoming orders. Metrics (energy, turning, stop-and-go, job queue) are tracked per tick.

3. **Mid-point Re-clustering:** Simulation features (actual throughput, congestion) are extracted and merged with static features. SKUs are re-clustered and pods are reassigned.

4. **Phase 2:** The simulation restarts with updated pod assignments and processes the remaining orders. All metrics are tracked independently from Phase 1.
