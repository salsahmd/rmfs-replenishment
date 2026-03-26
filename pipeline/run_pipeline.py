"""
RMFS warehouse simulation pipeline (Rika's baseline).

This is the same as run_baseline.py — re-clustering is not available
in the baseline configuration. Use this as the main entry point.

Usage:
    python pipeline/run_pipeline.py [--max-ticks 1000]
"""

from run_baseline import main

if __name__ == "__main__":
    main()
